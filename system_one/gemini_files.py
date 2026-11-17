"""Upload media to the Gemini Files API once and reuse the file URI across calls.

A chat baseline asks one question per call, so inline video would be re-uploaded with every question (several GB
for a benchmark run). Uploaded files live for 48 hours; the cache (`~/.cache/qwen_rlcd/gemini_files.json`) maps a
local file (path, size, mtime) to its URI and expiry, and re-uploads when the entry is stale.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

BASE = "https://generativelanguage.googleapis.com"
CACHE = os.path.expanduser("~/.cache/qwen_rlcd/gemini_files.json")
_lock = threading.Lock()


def _key(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def _load() -> dict:
    try:
        with open(CACHE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE)


def _request(url, api_key, data=None, headers=None, method="GET", timeout=600):
    req = urllib.request.Request(url, data=data, method=method, headers={"x-goog-api-key": api_key, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.headers, resp.read()


def upload(path: str, mime_type: str, api_key: str, retries: int = 4) -> str:
    """Files API URI for `path`, uploading it (resumable protocol) unless a fresh cached URI exists."""
    key = _key(path)
    with _lock:
        hit = _load().get(key)
    if hit and hit["expires"] > time.time() + 3600:
        return hit["uri"]
    size = os.path.getsize(path)
    for attempt in range(retries + 1):
        try:
            headers, _ = _request(
                f"{BASE}/upload/v1beta/files", api_key, method="POST",
                data=json.dumps({"file": {"display_name": os.path.basename(path)}}).encode(),
                headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                         "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime_type,
                         "Content-Type": "application/json"},
            )  # fmt: skip
            with open(path, "rb") as f:
                _, body = _request(
                    headers["x-goog-upload-url"], api_key, method="POST", data=f.read(),
                    headers={"X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0",
                             "Content-Length": str(size)},
                )  # fmt: skip
            info = json.loads(body)["file"]
            while info.get("state") == "PROCESSING":  # video is processed before it can be used
                time.sleep(2)
                info = json.loads(_request(f"{BASE}/v1beta/{info['name']}", api_key)[1])
            if info.get("state") != "ACTIVE":
                raise RuntimeError(f"Gemini file {info.get('name')} is {info.get('state')}")
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt * 2, 30))
    with _lock:
        cache = _load()
        cache[key] = {"uri": info["uri"], "expires": time.time() + 47 * 3600}
        _save(cache)
    return info["uri"]
