"""Heuristic highlight score, top-k selection and reel building."""
import logging
from pathlib import Path

from . import media

log = logging.getLogger(__name__)


def _norm(values):
    """Min-max normalise to 0..1 (all-equal list -> zeros)."""
    lo, hi = min(values), max(values)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def score_shots(shots, cfg):
    """score = w_motion*motion_norm + w_people*people_norm + w_vlm*(1 if exciting else 0)
    motion is min-max normalised across shots; people is divided by the max average across shots.
    The weights are heuristics, not learned."""
    motion = _norm([s["motion_score"] for s in shots])
    max_p = max(s["avg_people"] for s in shots) or 1.0
    for s, m in zip(shots, motion):
        p = s["avg_people"] / max_p
        v = 1.0 if s["vlm_exciting"] else 0.0
        s["final_score"] = round(cfg.w_motion * m + cfg.w_people * p + cfg.w_vlm * v, 4)


def select_top(shots, cfg):
    """Highest scores first; skip any shot whose time range overlaps one already chosen."""
    best = []
    for s in sorted(shots, key=lambda s: s["final_score"], reverse=True):
        if all(s["end"] <= b["start"] or s["start"] >= b["end"] for b in best):
            best.append(s)
        if len(best) == cfg.top_k:
            break
    return sorted(best, key=lambda s: s["start"])        # play in chronological order


def build_reel(video_path, selected, duration, out_dir, cfg):
    tmp = Path(out_dir) / "clips"
    tmp.mkdir(exist_ok=True)
    clips = []
    for s in selected:
        length = min(s["end"] - s["start"], cfg.max_clip_sec)
        centre = (s["keyframe_time"])
        start = min(max(s["start"], centre - length / 2), s["end"] - length)   # window centred on keyframe, inside shot
        start = max(0.0, start)
        length = min(length, duration - start)
        clip = tmp / f"clip_shot{s['shot_id']:03d}.mp4"
        media.cut_clip(video_path, start, length, s["vlm_caption"], clip, cfg)
        clips.append(clip)
        log.info("Cut shot %d: %.1fs -> %.1fs", s["shot_id"], start, start + length)
    out = Path(out_dir) / "highlights.mp4"
    media.concat(clips, out)
    log.info("Wrote %s", out)
    return out
