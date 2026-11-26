"""Потоковая конвертация slovo_mediapipe.json (1.2+ ГБ) в компактный кэш .npz.

Файл не грузится целиком: верхнеуровневые ключи — UUID видео, каждое видео
парсится отдельно. Результат: data/cache/landmarks.npz
  ids      — attachment_id (str)
  offsets  — начало кадров каждого видео в общем массиве (len = n_videos + 1)
  frames   — float16 [N_frames, 2, 21, 3], отсутствующая рука = NaN
  nhands   — uint8 [N_frames], сколько рук было в исходном кадре (0..3)

Использование: python training/prepare_data.py [path/to/slovo_mediapipe.json]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "slovo_mediapipe.json"
OUT = ROOT / "data" / "cache" / "landmarks.npz"
KEY_RE = re.compile(rb'"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})":\s*\[')
CHUNK = 64 * 1024 * 1024


def iter_videos(path):
    """Yield (video_id, frames_list) without loading the whole file."""
    buf = b""
    cur_id, cur_start = None, None
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            buf += chunk
            pos = 0
            while True:
                m = KEY_RE.search(buf, pos)
                if not m:
                    break
                if cur_id is not None:
                    seg = buf[cur_start:m.start()].rstrip().rstrip(b",")
                    yield cur_id, json.loads(seg)
                cur_id, cur_start = m.group(1).decode(), m.end() - 1
                pos = m.end()
            if not chunk:
                break
            # keep only the unfinished current video in the buffer
            if cur_start is not None:
                buf = buf[cur_start:]
                cur_start = 0
    if cur_id is not None:
        seg = buf[cur_start:].rstrip().rstrip(b"}").rstrip().rstrip(b",")
        yield cur_id, json.loads(seg)


def frame_to_array(fr):
    arr = np.full((2, 21, 3), np.nan, dtype=np.float32)
    hands = [k for k in sorted(fr) if k.startswith("hand")]
    for i, k in enumerate(hands[:2]):
        pts = fr[k]
        if len(pts) == 21:
            arr[i] = [[p["x"], p["y"], p["z"]] for p in pts]
    return arr, len(hands)


def main():
    ids, offsets, frames, nhands = [], [0], [], []
    frame_lens, hand_counts, other_keys = [], Counter(), Counter()
    for n, (vid, vframes) in enumerate(iter_videos(SRC)):
        ids.append(vid)
        for fr in vframes:
            for k in fr:
                if not k.startswith("hand"):
                    other_keys[k] += 1
            a, h = frame_to_array(fr)
            frames.append(a)
            nhands.append(h)
            hand_counts[h] += 1
        frame_lens.append(len(vframes))
        offsets.append(len(frames))
        if n % 2000 == 0:
            print(f"  {n} videos, {len(frames)} frames", flush=True)

    fl = np.array(frame_lens)
    print(f"\nvideos: {len(ids)}, frames: {len(frames)}")
    print(f"frames/video: min {fl.min()}, median {np.median(fl):.0f}, mean {fl.mean():.1f}, max {fl.max()}")
    print(f"videos with 0 frames: {(fl == 0).sum()}")
    print("hands per frame:", dict(sorted(hand_counts.items())))
    print("non-hand keys:", dict(other_keys) or "none")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        ids=np.array(ids),
        offsets=np.array(offsets, dtype=np.int64),
        frames=np.stack(frames).astype(np.float16) if frames else np.zeros((0, 2, 21, 3), np.float16),
        nhands=np.array(nhands, dtype=np.uint8),
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
