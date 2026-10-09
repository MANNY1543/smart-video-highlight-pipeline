"""Tests for the optional bonus features (tracking, audio, dashboard helpers). No Gemini key or model download needed.
Run: python tests/test_bonus.py   (or python -m pytest tests -q)"""
import json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import detect, tracking
from pipeline.config import Config
from pipeline.shots import Shot


# ------------------------------------------------------------------ helpers
def moving_people(n_frames, start_idx=0):
    """Two people walking in different places at 15 px/frame -> {frame_idx: detections}."""
    d = {}
    for k in range(n_frames):
        a = (100 + 15 * k, 200, 180 + 15 * k, 400)
        b = (700 - 15 * k, 220, 780 - 15 * k, 420)
        d[start_idx + k] = [("person", 0.9, a), ("person", 0.8, b), ("sports ball", 0.7, (10, 10, 30, 30))]
    return d


# ------------------------------------------------------------------ aggregation
def test_aggregate_tracks_stats():
    obs = {1: [(2, (0, 0, 10, 10), 0.8), (3, (5, 0, 15, 10), 0.6), (6, (20, 0, 30, 10), 0.7)],   # gap at frames 4, 5
           2: [(4, (0, 0, 10, 10), 0.9)]}
    t1, t2 = tracking.aggregate_tracks(obs, fps=2.0)
    assert (t1["track_id"], t1["first_seen_sec"], t1["last_seen_sec"]) == (1, 1.0, 3.0)
    assert t1["frames_detected"] == 3 and t1["frames_missed_within_span"] == 2
    assert t1["observed_duration_sec"] == 2.5                       # (6-2+1) frames / 2 FPS
    assert t1["mean_confidence"] == 0.7 and t1["path_centres_px"][0] == [5.0, 5.0]
    assert t2["frames_detected"] == 1 and t2["observed_duration_sec"] == 0.5 and t2["frames_missed_within_span"] == 0


def test_shot_tracking_summary():
    tr = tracking.aggregate_tracks({1: [(0, (0, 0, 1, 1), .9), (1, (0, 0, 1, 1), .9)], 2: [(0, (5, 5, 6, 6), .9)]}, 2.0)
    s = tracking.shot_tracking_summary(tr, [2, 1])
    assert s["num_people_tracks"] == 2 and s["track_ids"] == [1, 2] and s["max_concurrent_tracked"] == 2
    assert s["longest_track"]["track_id"] == 1 and s["mean_observed_duration_sec"] == 0.75
    assert tracking.shot_tracking_summary([], [])["longest_track"] is None


# ------------------------------------------------------------------ track IDs (real ByteTrack if installed, else fallback)
def _ids_persist(backend):
    cfg = Config(); dets = moving_people(6)
    shots = [Shot(0, 0.0, 3.0)]
    result, per_frame, per_shot = tracking.track_shots(dets, shots, cfg, backend=backend)
    tr = result["shots"][0]["tracks"]
    assert len(tr) == 2, tr                                          # two people -> exactly two tracks
    assert all(t["frames_detected"] == 6 and t["first_seen_sec"] == 0.0 and t["last_seen_sec"] == 2.5 for t in tr)
    ids_by_frame = [sorted(r[0] for r in per_frame[i]) for i in range(6)]
    assert all(x == ids_by_frame[0] for x in ids_by_frame)           # same IDs on every frame
    assert per_shot[0]["max_concurrent_tracked"] == 2 and result["classes_tracked"] == ["person"]
    return result


def test_track_ids_persist_bytetrack():
    cfg = Config(); b = tracking.make_backend(cfg)
    if b.name != "bytetrack":
        print("  (ultralytics not installed: ByteTrack backend not exercised here)"); return
    assert _ids_persist(b)["backend"] == "bytetrack"


def test_track_ids_persist_fallback_tracker():
    assert _ids_persist(tracking._IoUBackend(Config()))["backend"] == "iou_greedy"


def test_tracker_resets_at_shot_boundary_and_ids_are_shot_local():
    cfg = Config(); dets = moving_people(4); dets.update(moving_people(4, start_idx=4))   # frames 0-3 | 4-7
    shots = [Shot(0, 0.0, 2.0), Shot(1, 2.0, 4.0)]
    for backend in (tracking.make_backend(cfg), tracking._IoUBackend(cfg)):
        result, _, _ = tracking.track_shots(dets, shots, cfg, backend=backend)
        a, b = result["shots"]
        assert sorted(t["track_id"] for t in a["tracks"]) == sorted(t["track_id"] for t in b["tracks"]) == [1, 2]  # restart at 1
        assert b["tracks"][0]["first_seen_sec"] == 2.0                                                              # new shot, new tracks
        assert "NOT a count of unique people" in " ".join(result["notes"])


def test_non_person_objects_are_not_tracked():
    cfg = Config(); dets = {i: [("sports ball", 0.9, (10 * i, 10, 10 * i + 20, 30))] for i in range(4)}
    result, per_frame, per_shot = tracking.track_shots(dets, [Shot(0, 0.0, 2.0)], cfg, backend=tracking._IoUBackend(cfg))
    assert result["total_tracks"] == 0 and per_shot[0]["longest_track"] is None


def test_low_confidence_person_cannot_start_a_track_but_can_extend_one():
    cfg = Config()
    b = tracking.make_backend(cfg)
    if b.name != "bytetrack":
        return
    box = lambda k, conf: [("person", conf, (100 + 5 * k, 100, 160 + 5 * k, 300))]
    dets = {0: box(0, 0.4), 1: box(1, 0.4)}                                   # below track_high_thresh 0.5: no track starts
    r, _, _ = tracking.track_shots(dets, [Shot(0, 0.0, 1.0)], cfg, backend=b)
    assert r["total_tracks"] == 0
    dets = {0: box(0, 0.9), 1: box(1, 0.4), 2: box(2, 0.4)}                   # confident start, low-conf continuation
    r, _, _ = tracking.track_shots(dets, [Shot(0, 0.0, 2.0)], cfg, backend=b)
    assert r["total_tracks"] == 1 and r["shots"][0]["tracks"][0]["frames_detected"] == 3


# ------------------------------------------------------------------ pipeline integration (fake YOLO, tiny synthetic video)
def _make_video(path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=4:r=25",
                    "-f", "lavfi", "-i", "smptebars=s=640x360:d=4:r=25", "-f", "lavfi", "-i", "rgbtestsrc=s=640x360:d=4:r=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a", "-pix_fmt", "yuv420p",
                    "-shortest", str(path)], check=True)


def _fake_detect(frames_dir, cfg):
    paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    d = {}
    for i in range(len(paths)):
        k = i % 8
        d[i] = [("person", 0.9, (50 + 10 * k, 50, 150 + 10 * k, 300)), ("person", 0.8, (400 - 10 * k, 60, 500 - 10 * k, 310)),
                ("sports ball", .8, (10, 10, 40, 40))]
    return d, paths


def _run_main(extra, tmp, patch=None):
    import importlib
    detect.detect_frames = _fake_detect
    main = importlib.import_module("main")
    importlib.reload(main)
    src = Path(tmp) / "in.mp4"
    if not src.exists():
        _make_video(src)
    out = Path(tmp) / ("res_" + str(abs(hash(tuple(extra)))))
    sys.argv = ["main.py", "--input", str(src), "--out", str(out), "--vlm-model", "none"] + extra
    if patch:
        patch(main)
    main.main()
    return out, json.loads((out / "summary.json").read_text())


def test_baseline_cli_unchanged_when_bonuses_off():
    with tempfile.TemporaryDirectory() as tmp:
        out, s = _run_main([], tmp)
        assert s["features"] == {"tracking": "off", "audio_analysis": "off"} and "tracking" not in s and "audio_analysis" not in s
        assert not (out / "audio_analysis.json").exists() and all("audio" not in r for r in s["shots"])
        assert not (out / "tracking.json").exists() and not (out / "tracking.mp4").exists()
        assert all("tracking" not in r for r in s["shots"]) and len(s["highlight_shot_ids"]) >= 1
        assert (out / "highlights.mp4").exists()


def test_tracking_enabled_on_small_sample():
    with tempfile.TemporaryDirectory() as tmp:
        out, s = _run_main(["--tracking"], tmp)
        assert s["features"]["tracking"] == "ok" and s["tracking"]["file"] == "tracking.json"
        t = json.loads((out / "tracking.json").read_text())
        assert t["total_tracks"] >= 2 and t["shots"] and (out / "tracking.mp4").exists() and (out / "highlights.mp4").exists()
        assert all(r["tracking"] and r["tracking"]["num_people_tracks"] >= 1 for r in s["shots"])
        # existing per-shot fields are all still there
        assert {"motion_score", "avg_people", "vlm_caption", "final_score", "keyframe"} <= set(s["shots"][0])
        assert s["scoring"]["weights"] == {"motion": 0.5, "people": 0.2, "vlm": 0.3}


def test_tracking_failure_does_not_break_highlights():
    orig = tracking.run_tracking
    tracking.run_tracking = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tracker exploded"))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out, s = _run_main(["--tracking"], tmp)
            assert s["features"]["tracking"].startswith("failed") and (out / "highlights.mp4").exists()
            assert "tracking" not in s and not (out / "tracking.json").exists()
    finally:
        tracking.run_tracking = orig


def test_tracking_video_can_be_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        out, s = _run_main(["--tracking", "--no-tracking-video"], tmp)
        assert (out / "tracking.json").exists() and not (out / "tracking.mp4").exists()


# ------------------------------------------------------------------ audio loudness
from pipeline import audio as A


def _sig(sr=16000, secs=12, bursts=((5.0, 6.0, 0.5),), base=0.01, seed=0):
    rng = np.random.RandomState(seed)
    x = (base * rng.randn(sr * secs)).astype(np.float32)
    for a, b, amp in bursts:
        x[int(a * sr):int(b * sr)] += (amp * np.sin(2 * np.pi * 440 * np.arange(int((b - a) * sr)) / sr)).astype(np.float32)
    return x


def test_window_levels_known_values():
    sr = 16000
    x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 4) / sr)).astype(np.float32)
    t, db = A.window_levels(x, sr, 0.5, 0.25)
    assert len(t) == (4 * sr - 8000) // 4000 + 1 and abs(t[0] - 0.25) < 1e-6 and abs(t[1] - t[0] - 0.25) < 1e-6
    assert np.allclose(db, 20 * np.log10(0.5 / np.sqrt(2)), atol=0.05)             # RMS of a 0.5-amplitude sine = -9.03 dBFS
    t, db = A.window_levels(np.zeros(sr * 2, np.float32), sr, 0.5, 0.25)
    assert db.max() <= -100                                                         # digital silence
    assert len(A.window_levels(np.zeros(100, np.float32), sr, 0.5, 0.25)[0]) == 0   # shorter than one window


def test_peak_found_at_burst_time_and_silence_gives_none():
    t, db = A.window_levels(_sig(), 16000, 0.5, 0.25)
    peaks, stats = A.find_peaks(t, db, 3.0, 3.0, 1.0)
    assert len(peaks) == 1 and 5.0 <= peaks[0]["time_sec"] <= 6.0 and peaks[0]["above_clip_median_db"] > 20
    t, db = A.window_levels(np.zeros(16000 * 5, np.float32), 16000, 0.5, 0.25)
    assert A.find_peaks(t, db, 3.0, 3.0, 1.0)[0] == []
    t, db = A.window_levels(_sig(bursts=()), 16000, 0.5, 0.25)                      # steady noise: nothing stands out
    assert A.find_peaks(t, db, 3.0, 3.0, 1.0)[0] == []


def test_min_gap_keeps_only_loudest_of_close_peaks():
    t, db = A.window_levels(_sig(bursts=((3.0, 3.5, 0.3), (4.0, 4.5, 0.6), (9.0, 9.5, 0.4))), 16000, 0.5, 0.25)
    peaks, _ = A.find_peaks(t, db, 3.0, 3.0, 2.0)
    times = [p["time_sec"] for p in peaks]
    assert len(peaks) == 2 and any(3.5 <= x <= 5 for x in times) and any(9 <= x <= 10 for x in times)


def test_peaks_associated_with_shots_by_timestamp():
    analysis = {"loudness_series": [[0.25, -40.0], [1.25, -30.0], [2.25, -20.0], [3.25, -35.0]],
                "peaks": [{"time_sec": 2.25, "level_db": -20.0, "above_clip_median_db": 15.0, "relative_strength": 5.0}]}
    shots = [Shot(0, 0.0, 2.0), Shot(1, 2.0, 4.0)]
    r = A.associate_with_shots(analysis, shots)
    assert r[0]["peak_count"] == 0 and r[0]["max_level_db"] == -30.0 and r[0]["mean_level_db"] == -35.0
    assert r[1]["peak_count"] == 1 and r[1]["peak_times_sec"] == [2.25] and r[1]["max_peak_above_median_db"] == 15.0
    edge = A.associate_with_shots({"loudness_series": [], "peaks": [{"time_sec": 2.0, "above_clip_median_db": 9.0}]}, shots)
    assert edge[1]["peak_count"] == 1 and edge[0]["peak_count"] == 0                # [start, end) boundary
    assert edge[0]["mean_level_db"] is None


def test_experimental_score_is_separate_and_default_untouched():
    from pipeline import highlights
    cfg = Config(); cfg.top_k = 1
    mk = lambda i, sc, peak: dict(shot_id=i, start=i * 10, end=i * 10 + 5, final_score=sc,
                                  audio={"max_peak_above_median_db": peak})
    recs = [mk(0, 0.80, 0.0), mk(1, 0.75, 20.0)]
    rep = A.experimental_scores(recs, cfg, highlights.select_top)
    assert [r["final_score"] for r in recs] == [0.80, 0.75]                         # default scores untouched
    assert rep["default_top_shot_ids"] == [0] and rep["top_shot_ids"] == [1] and rep["differs_from_default"] is True
    assert recs[1]["experimental_audio_score"] == round(0.85 * 0.75 + 0.15 * 1.0, 4)
    assert "not proof" in rep["note"]


def _make_video_with_burst(path, audio=True):
    af = ["-f", "lavfi", "-i", "sine=frequency=440:duration=12,volume=0.02,volume=volume=40:enable='between(t,5,6)'"] if audio else []
    maps = ["-map", "3:a", "-shortest"] if audio else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=4:r=25",
                    "-f", "lavfi", "-i", "smptebars=s=640x360:d=4:r=25", "-f", "lavfi", "-i", "rgbtestsrc=s=640x360:d=4:r=25", *af,
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]", "-map", "[v]", *maps, "-pix_fmt", "yuv420p", str(path)], check=True)


def test_audio_analysis_enabled_on_small_sample():
    with tempfile.TemporaryDirectory() as tmp:
        _make_video_with_burst(Path(tmp) / "in.mp4")
        out, s = _run_main(["--audio-analysis"], tmp)
        a = json.loads((out / "audio_analysis.json").read_text())
        assert s["features"]["audio_analysis"] == "ok" and s["audio_analysis"]["file"] == "audio_analysis.json"
        assert a["sample_rate_hz"] == 16000 and a["peaks"] and any(5 <= p["time_sec"] <= 6.5 for p in a["peaks"])
        assert all("audio" in r and {"peak_count", "mean_level_db", "max_level_db"} <= set(r["audio"]) for r in s["shots"])
        assert sum(r["audio"]["peak_count"] for r in s["shots"]) == len(a["peaks"])
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                                "-of", "csv=p=0", str(out / "highlights.mp4")], capture_output=True, text=True).stdout
        assert probe.strip(), "original audio must be preserved in highlights.mp4"
        assert "experimental_audio_ranking" not in s and "experimental_audio_score" not in s["shots"][0]


def test_default_ranking_identical_with_and_without_audio():
    with tempfile.TemporaryDirectory() as tmp:
        _make_video_with_burst(Path(tmp) / "in.mp4")
        _, base = _run_main([], tmp)
        _, aud = _run_main(["--experimental-audio-score"], tmp)
        assert base["highlight_shot_ids"] == aud["highlight_shot_ids"]
        assert [r["final_score"] for r in base["shots"]] == [r["final_score"] for r in aud["shots"]]
        assert aud["scoring"] == base["scoring"] and aud["experimental_audio_ranking"]["formula"]
        assert all("experimental_audio_score" in r for r in aud["shots"])


def test_audio_skipped_without_audio_track_and_failure_is_graceful():
    with tempfile.TemporaryDirectory() as tmp:
        _make_video_with_burst(Path(tmp) / "in.mp4", audio=False)
        out, s = _run_main(["--audio-analysis"], tmp)
        assert s["features"]["audio_analysis"].startswith("skipped") and (out / "highlights.mp4").exists()
        assert not (out / "audio_analysis.json").exists()
    orig = A.run_audio_analysis
    A.run_audio_analysis = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad wav"))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _make_video_with_burst(Path(tmp) / "in.mp4")
            out, s = _run_main(["--audio-analysis"], tmp)
            assert s["features"]["audio_analysis"].startswith("failed") and (out / "highlights.mp4").exists()
            assert "audio_analysis" not in s and all("audio" not in r for r in s["shots"])
    finally:
        A.run_audio_analysis = orig


# ------------------------------------------------------------------ dashboard helpers + app (Streamlit optional)
import dashboard_utils as U


def test_dashboard_upload_validation_and_safe_dirs():
    assert U.validate_upload("clip.MP4", 1000) is None and U.validate_upload("x.mov", 5) is None
    assert "Unsupported" in U.validate_upload("notes.txt", 10) and "Unsupported" in U.validate_upload("virus.exe", 10)
    assert "empty" in U.validate_upload("a.mp4", 0) and "larger" in U.validate_upload("a.mp4", 600 * 1024 * 1024)
    with tempfile.TemporaryDirectory() as root:
        d = U.safe_run_dir("my_run-1", runs_root=root)
        assert d.parent == Path(root) and d.exists()
        for bad in ["../evil", "a/b", "C:\\x", "", "x" * 65, "sp ace"]:
            if bad == "":
                assert U.safe_run_dir("", runs_root=root).name.startswith("highlights_")      # empty -> temp dir
                continue
            try:
                U.safe_run_dir(bad, runs_root=root); raise AssertionError(bad)
            except ValueError:
                pass


def test_dashboard_command_reuses_cli_and_never_leaks_key():
    cmd = U.build_command("in.mp4", "out", tracking=True, audio=True, experimental_audio=True, vlm_model="none", top_k=3)
    assert cmd[1].endswith("main.py") and "--tracking" in cmd and "--audio-analysis" in cmd and cmd[cmd.index("--vlm-model") + 1] == "none"
    assert not any("secret" in c for c in cmd) and U.child_env("secret-key")["GEMINI_API_KEY"] == "secret-key"
    assert U.redact("auth secret-key failed", "secret-key", "") == "auth *** failed"
    assert "--tracking" not in U.build_command("in.mp4", "out")


def _sample_results(tmp, extra):
    out, s = _run_main(extra, tmp)
    return out, s


def test_streamlit_app_renders_without_and_with_results():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("  (streamlit not installed: app test skipped)"); return
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not at.exception and at.title[0].value.startswith("Smart Video")
    assert [b.label for b in at.button] == ["Run pipeline"] and at.button[0].disabled        # nothing uploaded yet
    with tempfile.TemporaryDirectory() as tmp:
        _make_video_with_burst(Path(tmp) / "in.mp4")
        out, s = _sample_results(tmp, ["--tracking", "--audio-analysis", "--experimental-audio-score"])
        at2 = AppTest.from_file(str(root / "app.py"), default_timeout=60)
        at2.session_state["results"] = str(out)
        at2.run()
        assert not at2.exception, at2.exception
        assert len(at2.tabs) == 5 and any("Done" in x.value or "Results in" in x.value for x in at2.success)
        rows = U.shots_table(s)
        assert len(rows) == len(s["shots"]) and all(r["audio_peaks"] is not None and r["people_tracks"] is not None for r in rows)
        # a baseline run (no optional features) renders too, with the 'not available' hints
        out0, _ = _sample_results(tmp, [])
        at3 = AppTest.from_file(str(root / "app.py"), default_timeout=60)
        at3.session_state["results"] = str(out0)
        at3.run()
        assert not at3.exception, at3.exception

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
