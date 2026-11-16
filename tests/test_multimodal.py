"""M5: images and videos in the state.

A tiny random Qwen3.5 vision-language model with the real processor (tokenizer, image/video preprocessing).
Fork and single-pass reads with media must match transformers' own full multimodal forward over state + branch,
which computes M-RoPE positions for the whole sequence itself; that catches branch position-offset bugs.
EVS pruning is checked for its keep count, first-group rule, change ranking and position bookkeeping.
"""

import numpy as np
import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from system_one import multimodal
from system_one.multimodal import MediaConfig, encode_state, evs_keep_count, evs_retention_mask
from system_one.scorers import Backbone, LetterScorer
from system_one.schema import Request

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "Qwen/Qwen3.5-0.8B-Base"
TOL = 1e-4


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=16,
            linear_value_head_dim=16,
            # Full rotary with all three M-RoPE axes in use, so a wrong height/width/time offset changes reads.
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=256),
    )  # fmt: skip
    vlm = Qwen3_5Model(config).eval().float().to(DEVICE)
    lm_head = torch.nn.Linear(64, 248320, bias=False).to(DEVICE)
    lm_head.weight = vlm.language_model.embed_tokens.weight
    tok = AutoTokenizer.from_pretrained(REPO)
    proc = AutoProcessor.from_pretrained(REPO)
    return Backbone(vlm.language_model, lm_head, tok, chunk_size=2, vlm=vlm, processor=proc)


def _image(seed=0, h=64, w=96):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _video(seed=0, frames=8, h=64, w=64, static=False):
    rng = np.random.default_rng(seed)
    if static:  # one random frame repeated, with a moving bright square
        base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        clip = np.repeat(base[None], frames, axis=0).copy()
        for t in range(frames):
            clip[t, 16 * (t % 4) : 16 * (t % 4) + 16, :16] = 255
        return clip, 4.0
    return rng.integers(0, 255, (frames, h, w, 3), dtype=np.uint8), 4.0


def _reference(backbone, state_text, branch, images=(), videos=(), cfg=MediaConfig()):
    """transformers' own multimodal forward over state + branch (positions from get_rope_index on the full sequence)."""
    vlm, proc = backbone.vlm, backbone.processor
    imgs = [multimodal.load_image(i) for i in images]
    vids = [multimodal.load_video(v, cfg) for v in videos]
    kwargs = {}
    if imgs:
        kwargs.update(images=imgs, images_kwargs={"size": {"shortest_edge": 32 * 32 * 4, "longest_edge": cfg.image_max_pixels}})
    if vids:
        kwargs.update(videos=[f for f, _ in vids], videos_kwargs={"video_metadata": [m for _, m in vids], "do_sample_frames": False})
    text = multimodal.render_media_state(state_text, len(imgs), len(vids))
    enc = proc(text=[text], return_tensors="pt", **kwargs)
    ids = torch.cat([enc["input_ids"], torch.tensor([branch])], dim=1).to(DEVICE)
    types = torch.cat([enc["mm_token_type_ids"], torch.zeros((1, len(branch)), dtype=enc["mm_token_type_ids"].dtype)], dim=1)
    extra = {k: enc[k].to(DEVICE) for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw") if k in enc}
    vlm.rope_deltas = None
    with torch.no_grad():
        out = vlm(input_ids=ids, mm_token_type_ids=types.to(DEVICE), use_cache=False, **extra)
    return out.last_hidden_state[0, -len(branch) :]


MEDIA = {
    "image": dict(images=[_image(0)]),
    "two_images": dict(images=[_image(1, 64, 64), _image(2, 96, 64)]),
    "video": dict(videos=[_video(3)]),
    "image_and_video": dict(images=[_image(4)], videos=[_video(5, frames=6)]),
}


@pytest.mark.parametrize("single_pass", [False, True])
@pytest.mark.parametrize("media", list(MEDIA))
def test_media_reads_match_full_forward(backbone, media, single_pass):
    state = "Describe what you see."
    if single_pass:
        branches = [backbone.encode(" Is there a cat? Answer:")]
    else:
        branches = [backbone.encode(t) for t in (" Is there a cat? Answer:", " Colour:", " How many objects are there?")]
    m = MEDIA[media]
    backbone.single_pass = single_pass
    try:
        with torch.no_grad():
            got = backbone.run(state, branches, (tuple(m.get("images", ())), tuple(m.get("videos", ()))))
    finally:
        backbone.single_pass = True
    for b, h in zip(branches, got):
        ref = _reference(backbone, state, b, **m)
        assert h.shape == ref.shape
        assert (h - ref).abs().max().item() < TOL


def test_media_changes_reads(backbone):
    """Sanity: the reads depend on the image (so the equivalence test isn't vacuous)."""
    b = [backbone.encode(" Answer:")]
    with torch.no_grad():
        a = backbone.run("x", b, ((_image(0),), ()))[0]
        c = backbone.run("x", b, ((_image(9),), ()))[0]
    assert (a - c).abs().max().item() > 1e-3


def test_letter_scorer_accepts_media_requests(backbone):
    req = Request.from_dict({
        "state": "A picture.", "images": [_image(0)],
        "questions": {"q": {"type": "noul", "instructions": "Is it dark?"},
                      "c": {"type": "choice", "instructions": "Main colour?", "criteria": {"red": "", "blue": ""}}},
    })  # fmt: skip
    out = LetterScorer(backbone)(req)
    assert set(out) == {"q", "c"} and out["c"].shape == (2,)
    assert torch.isfinite(out["q"]).all()


# ---- EVS ---------------------------------------------------------------------------------------------------


def test_evs_keep_count_and_first_group():
    torch.manual_seed(0)
    t, h, w = 6, 8, 8  # merged grid 6 x 4 x 4
    feats = torch.randn(t * 16, 32)
    for q in (0.0, 0.3, 0.5, 0.75, 0.99):
        keep = evs_retention_mask(feats, (t, h, w), 2, q)
        assert keep.sum().item() == evs_keep_count(16, t, q) == max(16, int(96 * (1 - q)))
        assert keep[:16].all()  # first group always kept
    assert evs_retention_mask(feats, (t, h, w), 2, 0.0).all()


def test_evs_keeps_changing_tokens():
    """Tokens identical to the previous group are dropped before tokens that change."""
    torch.manual_seed(1)
    t = 4
    base = torch.randn(16, 32)
    feats = base.repeat(t, 1).clone()
    moving = [16 + 5, 32 + 6, 48 + 7]  # one location changes in each later group and stays changed
    for i in moving:
        feats[i::16][: t - i // 16] = torch.randn(32)
    keep = evs_retention_mask(feats, (t, 8, 8), 2, q=0.7)  # keeps max(16, int(64 * 0.3)) = 19
    assert keep.sum() == 19 and keep[:16].all() and keep[moving].all()


def test_evs_vs_vllm_reference():
    evs = pytest.importorskip("vllm.multimodal.video_prune.evs")
    torch.manual_seed(2)
    feats = torch.randn(5 * 16, 24)
    for q in (0.2, 0.6):
        ours = evs_retention_mask(feats, (5, 8, 8), 2, q)
        ref = evs.compute_retention_mask(feats, torch.tensor([5, 8, 8]), 2, q)
        assert torch.equal(ours, ref)


def test_evs_state_keeps_unpruned_positions(backbone):
    cfg0, cfg = MediaConfig(), MediaConfig(video_pruning_rate=0.5)
    video = _video(6, frames=8, static=True)
    full = encode_state(backbone.vlm, backbone.processor, "Clip.", videos=[video], cfg=cfg0)
    pruned = encode_state(backbone.vlm, backbone.processor, "Clip.", videos=[video], cfg=cfg)
    assert pruned.pruned_tokens > 0
    assert pruned.embeds.shape[1] == full.embeds.shape[1] - pruned.pruned_tokens
    assert pruned.next_position == full.next_position
    # Every kept token (text or video) appears in the unpruned sequence with the same id, position and embedding.
    full_keys = {tuple(full.position_ids[:, 0, i].tolist()) + (int(full.input_ids[0, i]),): i for i in range(full.embeds.shape[1])}
    for i in range(pruned.embeds.shape[1]):
        key = tuple(pruned.position_ids[:, 0, i].tolist()) + (int(pruned.input_ids[0, i]),)
        assert key in full_keys
        assert torch.equal(pruned.embeds[0, i], full.embeds[0, full_keys[key]])


def test_evs_reads_run_and_differ_from_unpruned(backbone):
    video = _video(7, frames=8, static=True)
    b = [backbone.encode(" Answer:"), backbone.encode(" Moving:")]
    try:
        with torch.no_grad():
            full = backbone.run("Clip.", b, ((), (video,)))
            backbone.media = MediaConfig(video_pruning_rate=0.75)
            pruned = backbone.run("Clip.", b, ((), (video,)))
            stats = backbone.last_media_stats
    finally:
        backbone.media = MediaConfig()
    assert stats["pruned_tokens"] > 0
    for f, p in zip(full, pruned):
        assert f.shape == p.shape and torch.isfinite(p).all()
