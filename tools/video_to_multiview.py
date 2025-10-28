from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean squared error between two images downscaled to 64x64."""
    a_small = cv2.resize(a, (64, 64))
    b_small = cv2.resize(b, (64, 64))
    diff = a_small.astype(np.float32) - b_small.astype(np.float32)
    return float(np.mean(diff * diff))


def extract_multiview(
    video_path: str,
    out_dir: str,
    target_views: int = 24,
    min_delta: float = 150.0,
    max_side: int = 896,
    fps_cap: float = 8.0,
) -> List[str]:
    """Extract representative frames from a video to approximate multi-view coverage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(src_fps / fps_cap)))

    saved_paths: List[str] = []
    last_frame: np.ndarray | None = None
    grabbed = 0
    write_idx = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        grabbed += 1
        if grabbed % stride != 0:
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if last_frame is not None and _frame_diff(frame, last_frame) < min_delta:
            continue

        height, width = frame.shape[:2]
        scale = max_side / max(height, width)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        out_path = out / f"view_{write_idx:03d}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        saved_paths.append(str(out_path))
        last_frame = frame
        write_idx += 1

        if len(saved_paths) >= target_views:
            break

    cap.release()
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--out", required=True, help="Directory to save extracted frames")
    parser.add_argument("--views", type=int, default=24, help="Maximum number of frames to extract")
    parser.add_argument(
        "--min_delta",
        type=float,
        default=150.0,
        help="Minimum MSE between consecutive saved frames to keep the new frame",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=896,
        help="Resize so that the longer side is at most this many pixels",
    )
    parser.add_argument(
        "--fps_cap",
        type=float,
        default=8.0,
        help="Upper bound on extraction FPS to avoid near-duplicates",
    )
    args = parser.parse_args()

    paths = extract_multiview(
        video_path=args.video,
        out_dir=args.out,
        target_views=args.views,
        min_delta=args.min_delta,
        max_side=args.max_side,
        fps_cap=args.fps_cap,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
