"""Optional person tracking on the EXISTING YOLO detections (no second inference pass).

* Backend: Ultralytics' ByteTrack (`BYTETracker`) fed with the detections `detect.detect_frames` already
  produced. If it cannot be imported/created, a small greedy IoU tracker (SORT-like, no Kalman filter) is used
  and the backend name is recorded in the output. Tracking never raises into the main pipeline (main.py wraps it).
* Frames are processed sequentially, and the tracker is RESET AT EVERY SHOT BOUNDARY: a cut changes the camera,
  so carrying IDs across a cut would only produce false associations. Track IDs are therefore SHOT-LOCAL; the
  pipeline does not re-identify people across shots and makes no claim about identity or unique-person counts.
* Only `person` is tracked. At the pipeline's 2 FPS a ball (or any small fast object) moves too far between
  sampled frames for tracking to be reliable, so other classes are not tracked.
* 2 FPS caveat: people can move a long way in 0.5 s, so ID switches and broken tracks are expected in fast play.
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- tracker backends
class _Dets:
    """Minimal results-like object for BYTETracker: .conf, .cls, .xywh, boolean indexing, len()."""
    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Dets(self.xywh[i], self.conf[i], self.cls[i])


class _ByteTrackBackend:
    name = "bytetrack"

    def __init__(self, cfg):
        import inspect
        from types import SimpleNamespace
        from ultralytics.trackers.byte_tracker import BYTETracker
        self._cls = BYTETracker
        self.args = SimpleNamespace(
            track_high_thresh=cfg.track_high_thresh,   # first association: confident detections
            track_low_thresh=cfg.yolo_conf,            # second association: remaining detections kept by YOLO
            new_track_thresh=cfg.track_high_thresh,    # only confident detections may start a track
            track_buffer=cfg.track_buffer,             # lost tracks kept this many SAMPLED frames
            match_thresh=cfg.track_match_thresh, fuse_score=True)
        self._takes_fps = "frame_rate" in inspect.signature(BYTETracker.__init__).parameters
        self.reset()

    def reset(self):
        # frame_rate=30 disables the old versions' fps scaling, so track_buffer means sampled frames in all versions
        self.t = self._cls(self.args, frame_rate=30) if self._takes_fps else self._cls(self.args)

    def update(self, boxes, conf):
        if len(boxes) == 0:
            rows = self.t.update(_Dets(np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
        else:
            b = np.asarray(boxes, float)
            xywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            rows = self.t.update(_Dets(xywh, np.asarray(conf, float), np.zeros(len(b))))
        return [(int(r[4]), tuple(float(v) for v in r[:4]), float(r[5])) for r in rows]   # [x1,y1,x2,y2,id,score,...]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _IoUBackend:
    """Fallback: greedy highest-IoU matching to the previous box of each live track. No motion model."""
    name = "iou_greedy"

    def __init__(self, cfg):
        self.min_iou = 1 - cfg.track_match_thresh      # same meaning as ByteTrack's match_thresh (cost = 1-IoU)
        self.max_lost = cfg.track_buffer
        self.reset()

    def reset(self):
        self.tracks, self.next_id = {}, 1               # id -> {"box":..., "lost": n}

    def update(self, boxes, conf):
        pairs = sorted(((_iou(t["box"], b), tid, j) for tid, t in self.tracks.items() for j, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_d, out = set(), set(), []
        for v, tid, j in pairs:
            if v < self.min_iou or tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            self.tracks[tid] = {"box": tuple(boxes[j]), "lost": 0}
            out.append((tid, tuple(float(x) for x in boxes[j]), float(conf[j])))
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        for j, b in enumerate(boxes):
            if j not in used_d and conf[j] >= 0:
                self.tracks[self.next_id] = {"box": tuple(b), "lost": 0}
                out.append((self.next_id, tuple(float(x) for x in b), float(conf[j])))
                self.next_id += 1
        return out


def make_backend(cfg):
    try:
        return _ByteTrackBackend(cfg)
    except Exception as e:                       # ultralytics missing / API changed
        log.warning("ByteTrack unavailable (%s: %s); using the greedy IoU fallback tracker", type(e).__name__, e)
        return _IoUBackend(cfg)


# ---------------------------------------------------------------- aggregation (pure functions, unit-tested)
def aggregate_tracks(observations, fps):
    """observations: {track_id: [(frame_idx, (x1,y1,x2,y2), conf), ...]} -> list of per-track statistics.
    observed_duration_sec = (last - first) + one sampling interval, i.e. the time span the track covers."""
    out = []
    for tid, obs in sorted(observations.items()):
        obs = sorted(obs)
        first, last = obs[0][0], obs[-1][0]
        span = last - first + 1
        centres = [[round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)] for _, b, _ in obs]
        out.append({
            "track_id": tid, "class": "person",
            "first_seen_sec": round(first / fps, 2), "last_seen_sec": round(last / fps, 2),
            "frames_detected": len(obs), "frames_missed_within_span": span - len(obs),
            "observed_duration_sec": round(span / fps, 2),
            "mean_confidence": round(float(np.mean([c for _, _, c in obs])), 3),
            "path_centres_px": centres})
    return out


def shot_tracking_summary(tracks, frame_counts):
    """Compact per-shot record that goes into summary.json."""
    return {
        "num_people_tracks": len(tracks),
        "track_ids": [t["track_id"] for t in tracks],
        "max_concurrent_tracked": max(frame_counts) if frame_counts else 0,
        "mean_observed_duration_sec": round(float(np.mean([t["observed_duration_sec"] for t in tracks])), 2) if tracks else 0.0,
        "longest_track": (lambda t: {"track_id": t["track_id"], "observed_duration_sec": t["observed_duration_sec"],
                                     "frames_detected": t["frames_detected"]})(
            max(tracks, key=lambda t: t["observed_duration_sec"])) if tracks else None,
    }


def frames_of_shot(shot, dets, fps):
    """Same frame-to-shot rule as detect.aggregate_shot."""
    idxs = sorted(i for i in dets if shot.start <= i / fps < shot.end)
    if not idxs:
        mid = (shot.start + shot.end) / 2 * fps
        idxs = [min(dets, key=lambda i: abs(i - mid))]
    return idxs


def track_shots(dets, shot_list, cfg, backend=None):
    """Sequential tracking of persons; tracker reset per shot. Returns (tracking_dict, per_frame, per_shot_summary)
    where per_frame = {frame_idx: [(track_id, box, conf), ...]} (IDs shot-local)."""
    backend = backend or make_backend(cfg)
    fps = cfg.frame_fps
    shots_out, per_frame, per_shot = [], {}, {}
    for s in shot_list:
        backend.reset()
        obs, counts = {}, []
        for i in frames_of_shot(s, dets, fps):
            people = [(c, b) for cls, c, b in dets[i] if cls == "person"]
            rows = backend.update([b for _, b in people], [c for c, _ in people])
            per_frame[i] = rows
            counts.append(len(rows))
            for tid, box, conf in rows:
                obs.setdefault(tid, []).append((i, box, conf))
        tracks = aggregate_tracks(obs, fps)
        per_shot[s.id] = shot_tracking_summary(tracks, counts)
        shots_out.append({"shot_id": s.id, "start": s.start, "end": s.end, **per_shot[s.id], "tracks": tracks})
    result = {
        "backend": backend.name, "fps_sampled": fps, "classes_tracked": ["person"],
        "parameters": {"track_high_thresh": cfg.track_high_thresh, "track_low_thresh": cfg.yolo_conf,
                       "track_buffer_frames": cfg.track_buffer, "match_thresh": cfg.track_match_thresh},
        "total_tracks": sum(len(x["tracks"]) for x in shots_out),
        "notes": ["Track IDs are SHOT-LOCAL: the tracker is reset at every shot boundary and nothing is re-identified "
                  "across shots, so total_tracks is NOT a count of unique people.",
                  "Frames are sampled at 2 FPS; people can move far between samples, so ID switches and "
                  "fragmented tracks are expected in fast play. IDs are associations by box overlap/motion, "
                  "not identity recognition.",
                  "Only the 'person' class is tracked (a ball is too fast/small at 2 FPS)."],
        "shots": shots_out}
    return result, per_frame, per_shot


# ---------------------------------------------------------------- drawing (cheap: no inference)
def _colour(tid):
    return tuple(int(x) for x in np.random.RandomState(tid * 7919 % 2**31).randint(60, 255, 3))


def draw_frame(img, rows, trails, other_dets=()):
    """Boxes + '#id' labels + short movement trails for the tracked people; plain boxes for other objects."""
    for cls, conf, (x1, y1, x2, y2) in other_dets:
        if cls != "person":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    for tid, (x1, y1, x2, y2), conf in rows:
        col = _colour(tid)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"#{tid}", (int(x1), max(int(y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        pts = trails.get(tid, [])
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
    return img


def _trail_points(shot_frames, per_frame, upto, trail_len):
    """Centre points of each track over the last `trail_len` sampled frames of the SAME shot up to `upto`."""
    trails = {}
    for j in [f for f in shot_frames if f <= upto][-trail_len:]:
        for tid, b, _ in per_frame.get(j, []):
            trails.setdefault(tid, []).append((int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)))
    return trails


def render_outputs(frame_paths, dets, shot_list, per_frame, key_map, out_dir, cfg, make_video=True):
    """Overwrite the per-shot annotated keyframes with track IDs/trails and encode results/tracking.mp4
    from the already-extracted frames (drawing only, no model inference)."""
    out_dir = Path(out_dir)
    fps = cfg.frame_fps
    shot_frames = {s.id: frames_of_shot(s, dets, fps) for s in shot_list}
    for s in shot_list:                                                       # annotated keyframes
        k = key_map.get(s.id)
        if k is None or k not in per_frame:
            continue
        img = cv2.imread(str(frame_paths[k]))
        draw_frame(img, per_frame[k], _trail_points(shot_frames[s.id], per_frame, k, cfg.trail_len), dets[k])
        cv2.imwrite(str(out_dir / "keyframes" / f"shot_{s.id:03d}.jpg"), img)
    if not make_video:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for s in shot_list:
            for i in shot_frames[s.id]:
                img = cv2.imread(str(frame_paths[i]))
                draw_frame(img, per_frame.get(i, []), _trail_points(shot_frames[s.id], per_frame, i, cfg.trail_len), dets[i])
                cv2.imwrite(f"{tmp}/f_{n:05d}.jpg", img); n += 1
        out = out_dir / "tracking.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f_%05d.jpg",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def run_tracking(dets, shot_list, frame_paths, key_map, cfg, out_dir):
    """Entry point used by main.py. Writes results/tracking.json (+ tracking.mp4, + ID-annotated keyframes)."""
    result, per_frame, per_shot = track_shots(dets, shot_list, cfg)
    try:
        video = render_outputs(frame_paths, dets, shot_list, per_frame, key_map, out_dir, cfg, cfg.tracking_video)
        result["visualization"] = video.name if video else None
    except Exception as e:                       # drawing/encoding must not lose the tracking data
        log.warning("Tracking visualization failed (%s); tracking.json is still written", e)
        result["visualization"] = None
    Path(out_dir, "tracking.json").write_text(json.dumps(result, indent=2))
    log.info("Tracking: backend=%s, %d shot-local person tracks -> tracking.json", result["backend"], result["total_tracks"])
    return result, per_shot
