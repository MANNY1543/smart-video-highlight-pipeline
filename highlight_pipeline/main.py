"""Smart Video Highlight & Insight Pipeline.

Usage: python main.py --input video.mp4 --out results/
"""
import argparse
import json
import logging
import subprocess
import time
from dataclasses import fields
from pathlib import Path

from pipeline import audio as audio_mod, detect, highlights, media, shots as shots_mod, tracking, vlm as vlm_mod
from pipeline.config import Config

log = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Smart Video Highlight & Insight Pipeline")
    p.add_argument("--input", required=True, help="input video file")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--vlm-model", help='Gemini model id (default: env GEMINI_MODEL), or "none" to skip the VLM')
    p.add_argument("--yolo-weights", help="e.g. yolo11n.pt or yolov8n.pt")
    p.add_argument("--cut-threshold", type=float, help="HSV histogram distance for a shot cut")
    p.add_argument("--debug", action="store_true", help="log raw VLM output")
    p.add_argument("--top-k", type=int, help="number of highlight shots")
    p.add_argument("--tracking", action="store_true", help="bonus: ByteTrack person tracking -> tracking.json (+ tracking.mp4)")
    p.add_argument("--audio-analysis", action="store_true", help="bonus: audio loudness peaks -> audio_analysis.json")
    p.add_argument("--experimental-audio-score", action="store_true",
                   help="with audio analysis: also report a separate, optional audio-weighted ranking (default ranking unchanged)")
    p.add_argument("--no-tracking-video", action="store_true", help="with --tracking: skip tracking.mp4")
    return p.parse_args()


def make_config(args):
    cfg = Config()
    overrides = {"vlm_model": args.vlm_model, "yolo_weights": args.yolo_weights,
                 "cut_threshold": args.cut_threshold, "top_k": args.top_k}
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.tracking:
        cfg.tracking = True
    if args.audio_analysis or args.experimental_audio_score:
        cfg.audio_analysis = True
    if args.experimental_audio_score:
        cfg.audio_experimental_score = True
    if args.no_tracking_video:
        cfg.tracking_video = False
    return cfg


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)   # keep request details out of logs
    cfg, out = make_config(args), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keyframes").mkdir(exist_ok=True)
    timings, t0 = {}, time.time()

    def lap(name):
        nonlocal t0
        timings[name] = round(time.time() - t0, 1)
        t0 = time.time()

    media.check_tools()
    if not Path(args.input).exists():
        raise SystemExit(f"Input not found: {args.input}")

    # 1. FFmpeg: probe + prepare
    log.info("STAGE 1/5 FFmpeg: probe + prepare")
    try:
        info = media.probe(args.input)
        video_720p, frames_dir, audio_wav = media.prepare(args.input, out, cfg, info["has_audio"])
    except (ValueError, KeyError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Invalid video or FFmpeg failure: {e}")
    lap("ffmpeg_prepare")

    # 2. OpenCV: shots + motion
    log.info("STAGE 2/5 OpenCV: shot detection + motion")
    shot_list = shots_mod.detect_shots(video_720p, info["duration_sec"], cfg)
    lap("shots_motion")

    # 3. YOLO: detections per frame, aggregated per shot, annotated keyframes
    log.info("STAGE 3/5 YOLO: object detection")
    try:
        dets, frame_paths = detect.detect_frames(frames_dir, cfg)
    except Exception as e:
        raise SystemExit(f"YOLO detection failed ({e}). Check network for weight download / torch install.")
    for s in shot_list:
        agg = detect.aggregate_shot(s, dets, cfg)
        s.agg = agg
        kf = out / "keyframes" / f"shot_{s.id:03d}.jpg"
        detect.save_keyframe(frame_paths[agg["key_idx"]], dets[agg["key_idx"]], kf)
        s.keyframe = kf
    lap("yolo")

    # 3b. (optional bonus) tracking on the existing detections; any failure is logged and ignored
    features = {"tracking": "off", "audio_analysis": "off"}
    tracking_info, tracking_per_shot = None, {}
    if cfg.tracking:
        log.info("STAGE 3b Tracking: ByteTrack on existing detections")
        try:
            key_map = {s.id: s.agg["key_idx"] for s in shot_list}
            tracking_info, tracking_per_shot = tracking.run_tracking(dets, shot_list, frame_paths, key_map, cfg, out)
            features["tracking"] = "ok"
        except Exception as e:
            log.warning("Tracking failed and was skipped (%s: %s). Highlight generation continues.", type(e).__name__, e)
            features["tracking"] = f"failed: {type(e).__name__}"
        lap("tracking")

    # 3c. (optional bonus) audio loudness on the already-extracted WAV; any failure is logged and ignored
    audio_info, audio_per_shot = None, {}
    if cfg.audio_analysis:
        log.info("STAGE 3c Audio: loudness peaks")
        if audio_wav is None or not Path(audio_wav).exists():
            features["audio_analysis"] = "skipped: no audio track"
            log.warning("Audio analysis skipped: the input has no audio track")
        else:
            try:
                audio_info, audio_per_shot = audio_mod.run_audio_analysis(audio_wav, shot_list, cfg, out)
                features["audio_analysis"] = "ok"
            except Exception as e:
                log.warning("Audio analysis failed and was skipped (%s: %s). Highlight generation continues.", type(e).__name__, e)
                features["audio_analysis"] = f"failed: {type(e).__name__}"
        lap("audio")

    # 4. VLM on each keyframe
    log.info("STAGE 4/5 VLM: %s", cfg.vlm_model)
    vlm = vlm_mod.VLM(cfg)
    records = []
    for s in shot_list:
        a = s.agg
        clean = frame_paths[a["key_idx"]]          # clean extracted frame (no boxes) goes to the VLM
        res, status = vlm.analyse(clean)
        log.info("VLM %d/%d shot %d [%s] %s/%s exciting=%s | %s", s.id + 1, len(shot_list), s.id, status, res.sport,
                 res.event_type, res.exciting, res.caption)
        records.append({
            "shot_id": s.id, "start": s.start, "end": s.end,
            "motion_score": s.motion, "avg_people": a["avg_people"], "common_objects": a["objects"],
            "keyframe": str(s.keyframe.relative_to(out)),                 # annotated (YOLO boxes)
            "keyframe_clean": str(Path(clean).relative_to(out)),          # what the VLM saw
            "keyframe_time": round(a["key_idx"] / cfg.frame_fps, 2),
            "vlm_caption": res.caption, "vlm_exciting": res.exciting, "vlm_reason": res.reason,
            "vlm_sport": res.sport, "vlm_event_type": res.event_type,
            "vlm_event_description": res.event_description, "vlm_players_and_actions": res.players_and_actions,
            "vlm_status": status, "vlm_ok": status != "fallback",
        })
    if tracking_per_shot:                                   # extra per-shot field; existing fields untouched
        for r in records:
            r["tracking"] = tracking_per_shot.get(r["shot_id"])
    if audio_per_shot:
        for r in records:
            r["audio"] = audio_per_shot.get(r["shot_id"])
    n = len(records)
    first = sum(r["vlm_status"] == "ok" for r in records)
    retry = sum(r["vlm_status"] == "retry_ok" for r in records)
    fb = sum(r["vlm_status"] == "fallback" for r in records)
    vlm_stats = {"total": n, "first_attempt_success": first, "retry_success": retry,
                 "fallback_count": fb, "success_rate": round(100 * (first + retry) / max(n, 1), 1)}
    log.info("VLM stats: total=%d first_success=%d retry_success=%d fallback=%d success_rate=%.1f%%",
             n, first, retry, fb, vlm_stats["success_rate"])
    if fb:
        log.warning("VLM fell back on %d/%d shots", fb, n)
    lap("vlm")

    # 5. Score, select, build reel (cut from the ORIGINAL video, original audio kept)
    log.info("STAGE 5/5 Highlights: score, cut, caption, concat")
    highlights.score_shots(records, cfg)
    selected = highlights.select_top(records, cfg)
    experimental = None
    if audio_info and cfg.audio_experimental_score:
        try:
            experimental = audio_mod.experimental_scores(records, cfg, highlights.select_top)
        except Exception as e:
            log.warning("Experimental audio score failed and was skipped (%s)", e)
    try:
        highlights.build_reel(args.input, selected, info["duration_sec"], out, cfg)
    except subprocess.CalledProcessError:
        raise SystemExit("FFmpeg failed while building highlights.mp4 (see error above)")
    lap("highlight_reel")

    summary = {
        "video": info,
        "scoring": {"formula": "w_motion*motion_norm + w_people*people_norm + w_vlm*vlm_yes",
                    "weights": {"motion": cfg.w_motion, "people": cfg.w_people, "vlm": cfg.w_vlm},
                    "note": "heuristic weights, not learned"},
        "vlm_model": cfg.vlm_model,
        "vlm_stats": vlm_stats,
        "highlight_shot_ids": [s["shot_id"] for s in selected],
        "highlight_events": [{k: s[k] for k in ("shot_id", "vlm_sport", "vlm_event_type", "vlm_event_description")}
                             for s in selected],
        "shots": records,
        "runtime_sec": timings,
        "features": features,                                # optional bonus features: off / ok / failed
    }
    if audio_info:
        summary["audio_analysis"] = {"file": "audio_analysis.json", "num_peaks": len(audio_info["peaks"]),
                                     "peak_times_sec": [p["time_sec"] for p in audio_info["peaks"]],
                                     "note": audio_info["notes"][1]}
    if experimental:
        summary["experimental_audio_ranking"] = experimental
    if tracking_info:
        summary["tracking"] = {"file": "tracking.json", "backend": tracking_info["backend"],
                               "total_shot_local_tracks": tracking_info["total_tracks"],
                               "visualization": tracking_info.get("visualization"),
                               "note": tracking_info["notes"][0]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Highlight shots: %s | timings: %s", summary["highlight_shot_ids"], timings)


if __name__ == "__main__":
    main()
