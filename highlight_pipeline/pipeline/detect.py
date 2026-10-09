"""YOLO object detection on the extracted frames, aggregated per shot."""
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def detect_frames(frames_dir, cfg):
    """Run YOLO (batched) over every frame. Returns {frame_index: [(cls_name, conf, (x1,y1,x2,y2)), ...]}."""
    from ultralytics import YOLO
    model = YOLO(cfg.yolo_weights)
    paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    dets = {}
    for i in range(0, len(paths), cfg.yolo_batch):
        batch = paths[i:i + cfg.yolo_batch]
        results = model.predict([str(p) for p in batch], conf=cfg.yolo_conf, verbose=False)
        for p, r in zip(batch, results):
            idx = int(p.stem.split("_")[1]) - 1           # frame_00001 -> index 0 -> t = 0 s
            dets[idx] = [(r.names[int(b.cls)], float(b.conf), tuple(map(int, b.xyxy[0].tolist())))
                         for b in r.boxes]
    log.info("YOLO processed %d frames", len(paths))
    return dets, paths


def aggregate_shot(shot, dets, cfg):
    """Frames belonging to the shot -> avg people, top objects, representative frame index."""
    idxs = [i for i in dets if shot.start <= i / cfg.frame_fps < shot.end]
    if not idxs:   # very short shot: fall back to the nearest frame
        mid = (shot.start + shot.end) / 2 * cfg.frame_fps
        idxs = [min(dets, key=lambda i: abs(i - mid))]
    people = [sum(1 for c, _, _ in dets[i] if c == "person") for i in idxs]
    objects = Counter(c for i in idxs for c, _, _ in dets[i] if c != "person")
    # keyframe = frame with the most people; ties -> closest to the middle of the shot
    mid = (shot.start + shot.end) / 2 * cfg.frame_fps
    key = max(idxs, key=lambda i: (sum(1 for c, _, _ in dets[i] if c == "person"), -abs(i - mid)))
    return {
        "avg_people": round(float(np.mean(people)), 2),
        "objects": [c for c, _ in objects.most_common(cfg.top_objects)],
        "key_idx": key,
    }


def save_keyframe(frame_path, detections, out_path):
    """Draw boxes + labels with OpenCV."""
    img = cv2.imread(str(frame_path))
    for cls, conf, (x1, y1, x2, y2) in detections:
        color = (0, 200, 0) if cls == "person" else (0, 140, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
