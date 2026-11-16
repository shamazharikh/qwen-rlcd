"""Images and videos in the state (PLAN.md M5), with optional EVS video-token pruning.

Media go into the **state** only. The state is embedded once: text tokens through the embedding table, image and
video placeholders replaced by vision-tower features. Positions are Qwen3.5's 3-axis M-RoPE (`get_rope_index`),
and every branch continues at the state's next text position on all three axes. The causal mask follows cache
indices, not positions, so position jumps inside the state (an image of N tokens advances positions by only its
side length) don't affect which tokens a branch sees.

EVS (Efficient Video Sampling, vLLM's `--video-pruning-rate`) drops video tokens that barely change from the
same patch in the previous temporal group: rank by 1 - cosine similarity of the merged vision features, always
keep the first group, keep `max(tokens_per_group, int(total * (1 - q)))` tokens. Retained tokens keep their
unpruned M-RoPE positions and the text after the video keeps its own, as in vLLM.

The state must be prefilled with 3-axis positions: a 2D or 4-axis text position track that jumps would be read
by transformers as packed sequences and split the attention.
"""

from __future__ import annotations

import base64
import io
import math
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass
class MediaConfig:
    """Bounds on vision tokens. Images: at most `image_max_pixels` (~1 token per 32x32 px after merging).
    Videos: `fps` frames per second, `min_frames`..`max_frames` (2 frames make one temporal group), each frame
    resized to at most `video_max_side` px on its longer side. `video_pruning_rate`: EVS q in [0, 1)."""

    image_max_pixels: int = 1024 * 1024
    fps: float = 2.0
    min_frames: int = 2
    max_frames: int = 32
    video_max_side: int = 448
    video_pruning_rate: float = 0.0


@dataclass
class MultimodalState:
    embeds: torch.Tensor  # [1, S, d] after pruning
    position_ids: torch.Tensor  # [3, 1, S] M-RoPE positions
    next_position: int  # first position for text after the state
    input_ids: torch.Tensor  # [1, S] (pruned video tokens removed)
    vision_tokens: int  # image + video tokens kept
    pruned_tokens: int


# ---- loading ---------------------------------------------------------------------------------------------


def _read_bytes(src: str) -> bytes:
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as r:
            return r.read()
    with open(src, "rb") as f:
        return f.read()


def load_image(src):
    """PIL image from a PIL image, path, URL or data URL, as RGB."""
    from PIL import Image

    img = src if isinstance(src, Image.Image) else Image.open(io.BytesIO(_read_bytes(src)))
    return img.convert("RGB")


def _resize_frames(frames: np.ndarray, max_side: int) -> np.ndarray:
    t, h, w, _ = frames.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return frames
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bilinear", antialias=True)
    return x.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def load_video(src, cfg: MediaConfig) -> tuple[np.ndarray, dict]:
    """Frames [T, H, W, 3] uint8 and metadata for the processor's timestamps.

    Sampling matches vLLM's video loader so both systems see the same frames: `n = min(max_frames,
    floor(duration * fps))` (at least `min_frames`), at `linspace(0, total - 1, n)` truncated to int, with
    timestamps taken from the source frame indices. Frames larger than `video_max_side` are downscaled.
    `src`: a path/URL/data URL, or an already-decoded `(frames, source_fps)` tuple."""
    if isinstance(src, tuple):
        all_frames, src_fps = src
        to_rgb = np.asarray
    else:
        import av

        # Decode every frame (the count fixes the sampling, and container headers can be wrong) but keep them
        # in their native format: only the sampled frames are converted to RGB, which is most of the CPU cost.
        with av.open(io.BytesIO(_read_bytes(src))) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            src_fps = float(stream.average_rate or 24)
            all_frames = list(container.decode(stream))

        def to_rgb(frames):
            return np.stack([f.to_ndarray(format="rgb24") for f in frames])

    total = len(all_frames)
    duration = total / src_fps
    n = min(total, max(cfg.min_frames, min(cfg.max_frames, math.floor(duration * cfg.fps))))
    idx = np.linspace(0, total - 1, n, dtype=int)
    frames = _resize_frames(to_rgb([all_frames[i] for i in idx]), cfg.video_max_side)
    meta = {"fps": src_fps, "total_num_frames": total, "duration": duration, "frames_indices": idx.tolist()}
    return frames, meta


# ---- EVS -------------------------------------------------------------------------------------------------


def evs_keep_count(tokens_per_group: int, groups: int, q: float) -> int:
    return max(tokens_per_group, int(tokens_per_group * groups * (1 - q)))


def evs_retention_mask(features: torch.Tensor, grid_thw, merge_size: int, q: float) -> torch.Tensor:
    """Boolean mask over one video's merged features ([T*h*w, d], T temporal groups) keeping the tokens that
    change most from the same location in the previous group; the first group is always kept."""
    t, h, w = (int(x) for x in grid_thw)
    h, w = h // merge_size, w // merge_size
    x = features.reshape(t, h, w, -1).float()
    change = 1 - torch.nn.functional.cosine_similarity(x[1:], x[:-1], dim=-1)  # [t-1, h, w]
    change = torch.cat([torch.full_like(x[:1, ..., 0], float("inf")), change]).flatten()
    keep = torch.zeros_like(change, dtype=torch.bool)
    keep[torch.argsort(change, descending=True, stable=True)[: evs_keep_count(h * w, t, q)]] = True
    return keep


# ---- state encoding --------------------------------------------------------------------------------------


def render_media_state(state_text: str, n_images: int, n_videos: int) -> str:
    from system_one import templates

    return templates.render_state(IMAGE_PLACEHOLDER * n_images + VIDEO_PLACEHOLDER * n_videos + "\n" + state_text)


@torch.no_grad()
def encode_state(vlm, processor, state_text: str, images=(), videos=(), cfg: MediaConfig | None = None) -> MultimodalState:
    """Embed a state with media. `vlm` is the `Qwen3_5Model` (vision tower + language model + `get_rope_index`)."""
    cfg = cfg or MediaConfig()
    lm = vlm.language_model
    device = lm.embed_tokens.weight.device
    imgs = [load_image(i) for i in images]
    vids = [load_video(v, cfg) for v in videos]
    kwargs = {}
    if imgs:
        kwargs["images"] = imgs
        kwargs["images_kwargs"] = {"size": {"shortest_edge": 32 * 32 * 4, "longest_edge": cfg.image_max_pixels}}
    if vids:
        kwargs["videos"] = [f for f, _ in vids]
        kwargs["videos_kwargs"] = {"video_metadata": [m for _, m in vids], "do_sample_frames": False}
    text = render_media_state(state_text, len(imgs), len(vids))
    enc = processor(text=[text], return_tensors="pt", **kwargs)
    input_ids = enc["input_ids"].to(device)
    embeds = lm.embed_tokens(input_ids)
    cfg_vlm = vlm.config
    merge = vlm.visual.spatial_merge_size
    keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
    vision_tokens = pruned = 0

    if imgs:
        feats = vlm.get_image_features(enc["pixel_values"].to(device), enc["image_grid_thw"].to(device)).pooler_output
        feats = torch.cat(feats).to(embeds.dtype)
        slots = (input_ids[0] == cfg_vlm.image_token_id).nonzero().squeeze(1)
        embeds[0, slots] = feats
        vision_tokens += len(slots)
    if vids:
        grids = enc["video_grid_thw"].to(device)
        per_video = vlm.get_video_features(enc["pixel_values_videos"].to(device), grids).pooler_output
        slots = (input_ids[0] == cfg_vlm.video_token_id).nonzero().squeeze(1)
        masks = []
        for feats, grid in zip(per_video, grids):
            if cfg.video_pruning_rate > 0:
                masks.append(evs_retention_mask(feats, grid, merge, cfg.video_pruning_rate))
            else:
                masks.append(torch.ones(len(feats), dtype=torch.bool, device=device))
        embeds[0, slots] = torch.cat(per_video).to(embeds.dtype)
        video_keep = torch.cat(masks)
        keep[slots] = video_keep
        vision_tokens += int(video_keep.sum())
        pruned = int((~video_keep).sum())

    position_ids, _ = vlm.get_rope_index(
        input_ids, enc["mm_token_type_ids"].to(device),
        image_grid_thw=enc.get("image_grid_thw", None), video_grid_thw=enc.get("video_grid_thw", None),
    )  # fmt: skip
    next_position = int(position_ids.max()) + 1
    return MultimodalState(
        embeds=embeds[:, keep], position_ids=position_ids[:, :, keep], next_position=next_position,
        input_ids=input_ids[:, keep], vision_tokens=vision_tokens, pruned_tokens=pruned,
    )  # fmt: skip


def prefill(text_model, state: MultimodalState):
    """Run the state once and return its cache."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=text_model.config)
    text_model(inputs_embeds=state.embeds, position_ids=state.position_ids, past_key_values=cache, use_cache=True)
    return cache


def forward_with_branch(text_model, state: MultimodalState, branch: list[int]) -> torch.Tensor:
    """Single pass: state + one branch in one uncached forward. Returns the branch's hidden states [len, d]."""
    device = state.embeds.device
    ids = torch.tensor([branch], device=device)
    embeds = torch.cat([state.embeds, text_model.embed_tokens(ids)], dim=1)
    pos = torch.arange(state.next_position, state.next_position + len(branch), device=device)
    position_ids = torch.cat([state.position_ids, pos.view(1, 1, -1).expand(3, 1, -1)], dim=2)
    out = text_model(inputs_embeds=embeds, position_ids=position_ids, use_cache=False)
    return out.last_hidden_state[0, -len(branch) :]
