
#!/usr/bin/env python3
"""
infer_video.py — Run RT-DETR inference over a video.

- Loads best.pt (Ultralytics RT-DETR) by default
- Decomposes an input video into frames
- Runs inference on each frame
- Stitches annotated frames back into an output video

Usage:
  python infer_video.py \
    --weights basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt \
    --source /path/to/input.mp4 \
    --output /path/to/output_annotated.mp4 \
    --imgsz 640 --conf 0.25 --iou 0.45 --device auto
"""

import os
import sys
import cv2
import shutil
import argparse
import tempfile
from typing import Optional
from tqdm import tqdm
import numpy as np

# Ultralytics RT-DETR
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR video inference and re-stitch")
    parser.add_argument(
        "--weights",
        type=str,
        default="basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt",
        help="Path to best.pt weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output annotated video (.mp4). Default: <input>_annotated.mp4",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--keep_frames",
        action="store_true",
        help="Keep decomposed frames in a temp directory (for debugging)",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=2,
        help="Line width for boxes/labels in visualization",
    )
    parser.add_argument(
        "--show_conf",
        action="store_true",
        help="Overlay confidence scores on labels",
    )
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help="Overlay class labels",
    )
    return parser.parse_args()


def ensure_output_path(in_path: str, out_path: Optional[str]) -> str:
    if out_path:
        return out_path
    stem, ext = os.path.splitext(in_path)
    return f"{stem}_annotated.mp4"


def open_video_reader(src_path: str):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)

    return cap, fps, width, height, frame_count


def open_video_writer(dst_path: str, fps: float, width: int, height: int):
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    # mp4v works well in Colab and most local setups
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {dst_path}")
    return writer


def main():
    args = parse_args()

    # Load model
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    model = RTDETR(args.weights)

    # Device handling (Ultralytics handles 'device' kw)
    device = None if args.device == "auto" else args.device

    # Open input video
    cap, fps, width, height, frame_count = open_video_reader(args.source)
    out_path = ensure_output_path(args.source, args.output)
    writer = open_video_writer(out_path, fps, width, height)

    # Optional temp frames directory (decompose → process → restitch)
    tmp_dir = tempfile.mkdtemp(prefix="rtdetr_frames_")
    if not args.keep_frames:
        # we’ll clean this up at the end
        pass

    try:
        pbar = tqdm(total=frame_count if frame_count > 0 else None, desc="Processing frames", unit="frame")
        idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Save raw frame if requested/for debugging
            raw_path = os.path.join(tmp_dir, f"frame_{idx:08d}.jpg")
            cv2.imwrite(raw_path, frame_bgr)

            # Run inference on this single frame (numpy array)
            # Ultralytics accepts numpy arrays in BGR; it will handle internal conversions.
            results = model.predict(
                source=frame_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
                show=False,
                save=False,
                agnostic_nms=False,
                max_det=300,
                # Visualization toggles applied when plotting below
            )

            # results is a list; one result per image
            res = results[0]

            # Get annotated frame (BGR)
            annotated = res.plot(
                line_width=args.line_width,
                labels=args.show_labels,
                conf=args.show_conf,
            )

            # (Optional) you could also save annotated frames to disk here if needed:
            # ann_path = os.path.join(tmp_dir, f"frame_{idx:08d}_annot.jpg")
            # cv2.imwrite(ann_path, annotated)

            # Write to output video
            writer.write(annotated)

            idx += 1
            pbar.update(1)

        pbar.close()
        print(f"\n✅ Done. Saved annotated video to: {out_path}")

        if args.keep_frames:
            print(f"📁 Kept decomposed frames at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    main()
