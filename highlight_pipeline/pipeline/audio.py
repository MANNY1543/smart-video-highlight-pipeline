"""Optional audio loudness analysis on the 16 kHz mono WAV that FFmpeg already extracted (NumPy + stdlib only).

Short-window RMS level in dBFS -> relative peaks (clip median + k * robust spread) -> peaks associated with shots by
timestamp. Peaks are SUPPORTING EVIDENCE ONLY: loudness can come from crowd noise, commentary, music or graphics
stingers, so it is never interpreted as a goal/wicket/celebration. The default highlight score is NOT changed; an
optional, separately reported experimental score is provided (see experimental_scores)."""
import json
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
SILENCE_DB = -80.0     # windows quieter than this are treated as digital silence


def read_wav(path):
    """-> (float32 mono samples in [-1, 1], sample_rate). Supports 8/16/32-bit PCM WAV."""
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, "u1").astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
    return x, sr


def window_levels(samples, sr, window_sec, hop_sec):
    """Short-window RMS in dBFS. Returns (centre_times_sec, db) as arrays; empty if the audio is shorter than a window."""
    win, hop = max(1, int(round(window_sec * sr))), max(1, int(round(hop_sec * sr)))
    if len(samples) < win:
        return np.zeros(0), np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(samples.astype(np.float64) ** 2)])
    starts = np.arange(0, len(samples) - win + 1, hop)
    rms = np.sqrt((c[starts + win] - c[starts]) / win)
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    return (starts + win / 2) / sr, db


def find_peaks(times, db, k, min_prominence_db, min_gap_sec):
    """Relative peaks: local maxima above median + max(k * 1.4826 * MAD, min_prominence_db), at least min_gap apart
    (the loudest peak wins). Digital silence never produces peaks."""
    if len(db) < 3:
        return [], {}
    loud = db[db > SILENCE_DB]
    if len(loud) < 3:
        return [], {"median_db": round(float(np.median(db)), 2)}
    med = float(np.median(loud))
    spread = 1.4826 * float(np.median(np.abs(loud - med)))
    thr = med + max(k * spread, min_prominence_db)
    cand = [i for i in range(len(db))
            if db[i] >= thr and db[i] >= (db[i - 1] if i else -1e9) and db[i] >= (db[i + 1] if i + 1 < len(db) else -1e9)]
    chosen = []
    for i in sorted(cand, key=lambda j: db[j], reverse=True):
        if all(abs(times[i] - times[j]) >= min_gap_sec for j in chosen):
            chosen.append(i)
    peaks = [{"time_sec": round(float(times[i]), 2), "level_db": round(float(db[i]), 2),
              "above_clip_median_db": round(float(db[i] - med), 2),
              "relative_strength": round(float((db[i] - med) / max(spread, 0.5)), 2)} for i in sorted(chosen)]
    return peaks, {"median_db": round(med, 2), "robust_spread_db": round(spread, 2), "peak_threshold_db": round(thr, 2)}


def analyse_audio(wav_path, cfg):
    x, sr = read_wav(wav_path)
    t, db = window_levels(x, sr, cfg.audio_window_sec, cfg.audio_hop_sec)
    peaks, stats = find_peaks(t, db, cfg.audio_peak_k, cfg.audio_min_prominence_db, cfg.audio_min_peak_gap_sec)
    return {
        "source": Path(wav_path).name, "sample_rate_hz": sr, "duration_sec": round(len(x) / sr, 2),
        "method": "short-window RMS level (dBFS); peak = local maximum above clip median + max(k*robust spread, min prominence)",
        "parameters": {"window_sec": cfg.audio_window_sec, "hop_sec": cfg.audio_hop_sec, "k": cfg.audio_peak_k,
                       "min_prominence_db": cfg.audio_min_prominence_db, "min_peak_gap_sec": cfg.audio_min_peak_gap_sec},
        "clip_stats": {**stats, "max_db": round(float(db.max()), 2) if len(db) else None,
                       "p95_db": round(float(np.percentile(db, 95)), 2) if len(db) else None},
        "peaks": peaks,
        "notes": ["Peaks are relative to this clip's own loudness distribution.",
                  "Peaks are supporting evidence only: loud crowd noise, commentary or music is not proof of a goal, "
                  "wicket, celebration or any other event. The audio is NOT interpreted.",
                  "Audio is not part of the default highlight score."],
        "loudness_series": [[round(float(a), 2), round(float(b), 1)] for a, b in zip(t, db)],
    }


def associate_with_shots(analysis, shot_list):
    """{shot_id: {...}} - loudness statistics and peaks whose timestamp falls inside [shot.start, shot.end)."""
    series = analysis["loudness_series"]
    out = {}
    for s in shot_list:
        levels = [d for t, d in series if s.start <= t < s.end]
        pk = [p for p in analysis["peaks"] if s.start <= p["time_sec"] < s.end]
        out[s.id] = {
            "mean_level_db": round(float(np.mean(levels)), 2) if levels else None,
            "max_level_db": round(float(max(levels)), 2) if levels else None,
            "peak_count": len(pk), "peak_times_sec": [p["time_sec"] for p in pk],
            "max_peak_above_median_db": max((p["above_clip_median_db"] for p in pk), default=0.0)}
    return out


def experimental_scores(records, cfg, select):
    """OPTIONAL, separate from the default ranking (which stays 0.5 motion + 0.2 people + 0.3 VLM).
    experimental = (1 - w_audio_exp) * final_score + w_audio_exp * audio_norm, where audio_norm is the shot's strongest
    peak above the clip median divided by the maximum over shots (0 if the shot has no peak). Adds
    `experimental_audio_score` to each record and returns a report with the alternative top-k; `final_score` and the
    default selection are never modified."""
    w = cfg.w_audio_exp
    mx = max((r["audio"]["max_peak_above_median_db"] for r in records if r.get("audio")), default=0.0) or 1.0
    for r in records:
        a = (r.get("audio") or {}).get("max_peak_above_median_db", 0.0) / mx
        r["experimental_audio_score"] = round((1 - w) * r["final_score"] + w * a, 4)
    alt = select([dict(r, final_score=r["experimental_audio_score"]) for r in records], cfg)
    default = select(records, cfg)
    return {"formula": f"(1 - {w}) * final_score + {w} * audio_peak_norm",
            "note": "Experimental and optional. The default ranking/highlight reel above is unchanged; this is reported "
                    "only for comparison. Loud audio is not proof of a notable event.",
            "top_shot_ids": [r["shot_id"] for r in alt], "default_top_shot_ids": [r["shot_id"] for r in default],
            "differs_from_default": [r["shot_id"] for r in alt] != [r["shot_id"] for r in default]}


def run_audio_analysis(wav_path, shot_list, cfg, out_dir):
    """Entry point used by main.py: writes results/audio_analysis.json, returns (analysis, per_shot)."""
    analysis = analyse_audio(wav_path, cfg)
    per_shot = associate_with_shots(analysis, shot_list)
    analysis["shots"] = [{"shot_id": sid, **v} for sid, v in per_shot.items()]
    Path(out_dir, "audio_analysis.json").write_text(json.dumps(analysis, indent=2))
    log.info("Audio: %d relative loudness peaks -> audio_analysis.json", len(analysis["peaks"]))
    return analysis, per_shot
