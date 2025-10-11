#!/usr/bin/env python3
# infer_lastshot.py
# Uses ./best.pt and ./lastshot.mp4, saves ./lastshot_annotated.mp4

import os
import cv2
from ultralytics import RTDETR
from tqdm import tqdm

WEIGHTS = "./best.pt"
INPUT_VIDEO = "./lastshot.mp4"
OUTPUT_VIDEO = "./lastshot_annotated.mp4"

IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = "auto"   # 'cuda', 'cpu', or 'auto'
LINE_WIDTH = 2
SHOW_LABELS = True
SHOW_CONF = True

def open_reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
    return cap, fps, w, h, n

def open_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output writer: {path}")
    return out

def main():
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")

    print(f"Loading model: {WEIGHTS}")
    model = RTDETR(WEIGHTS)

    print(f"Reading video: {INPUT_VIDEO}")
    cap, fps, w, h, n = open_reader(INPUT_VIDEO)
    out = open_writer(OUTPUT_VIDEO, fps, w, h)

    try:
        pbar = tqdm(total=n if n > 0 else None, desc="Annotating", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=None if DEVICE == "auto" else DEVICE,
                verbose=False,
                save=False,
                show=False,
                agnostic_nms=False,
                max_det=300,
            )
            annotated = results[0].plot(
                line_width=LINE_WIDTH,
                labels=SHOW_LABELS,
                conf=SHOW_CONF,
            )
            out.write(annotated)
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()
        out.release()

    print(f"✅ Done. Saved: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
