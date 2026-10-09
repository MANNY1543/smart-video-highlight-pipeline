"""OpenCV: shot detection (HSV histograms) + motion score (Farneback optical flow)."""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Shot:
    id: int
    start: float
    end: float
    motion: float = 0.0   # raw mean optical-flow magnitude (pixels per analysed step, at small resolution)


def _sample_video(path, cfg):
    """One pass over the video. For every sampled frame (after the first) return
    (time, histogram distance to previous sample, mean flow magnitude vs previous sample)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / cfg.analysis_fps))
    samples, prev_hist, prev_gray, i = [], None, None, 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (cfg.analysis_width, int(cfg.analysis_width * h / w)))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                samples.append((i / fps, float(dist), float(np.linalg.norm(flow, axis=2).mean())))
            prev_hist, prev_gray = hist, gray
        i += 1
    cap.release()
    return samples


def detect_shots(video_path, duration, cfg):
    samples = _sample_video(video_path, cfg)

    # 1) cuts = samples whose histogram distance exceeds the threshold
    boundaries = [0.0]
    for t, dist, _ in samples:
        if dist > cfg.cut_threshold and t - boundaries[-1] >= cfg.min_shot_sec:
            boundaries.append(t)          # (too-close cuts are ignored = merged into previous shot)
    boundaries.append(duration)

    # 2) split very long shots so a clip with few cuts still yields several candidates
    final = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        parts = max(1, int(np.ceil((b - a) / cfg.max_shot_sec)))
        edges = np.linspace(a, b, parts + 1)
        final += list(zip(edges[:-1], edges[1:]))

    # 3) motion = mean flow magnitude over the shot, ignoring the pair that straddles a cut
    shots = []
    for k, (a, b) in enumerate(final):
        mags = [m for t, d, m in samples if a < t <= b and d <= cfg.cut_threshold]
        shots.append(Shot(id=k, start=round(float(a), 2), end=round(float(b), 2),
                          motion=round(float(np.mean(mags)) if mags else 0.0, 4)))
    log.info("Detected %d shots (%d histogram cuts, threshold %.2f)",
             len(shots), len(boundaries) - 2, cfg.cut_threshold)
    return shots
