"""Run with: python -m pytest tests -q   (or python tests/test_pipeline.py)
Tests that need no model downloads: VLM two-line parsing, scoring/selection, and a full main.py run with
YOLO replaced by fake detections (so the glue code, summary.json and the reel are exercised)."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import detect, highlights
from pipeline.config import Config
from pipeline.vlm import VLM


import json as _json
FIELDS = {"caption", "sport", "event_type", "event_description", "players_and_actions", "exciting", "reason"}


def _resp(**kw):
    d = dict(caption="A batter plays a pull shot as fielders watch.", sport="cricket", event_type="batting_shot",
             event_description="The batter is mid-swing at a short delivery; the outcome is not visible in this frame.",
             players_and_actions="Batter swings horizontally; wicket-keeper crouches; fielders stand deep.",
             exciting=True, reason="A shot is being played in a clear competitive moment.")
    d.update(kw); return _json.dumps(d)


def test_vlm_schema_and_prompts():
    from pipeline import vlm as V
    from pydantic import ValidationError
    ok = V.VLMResult.model_validate_json(_resp(sport="Cricket ", event_type="Batting Shot"))
    assert ok.exciting is True and ok.sport == "cricket" and ok.event_type == "batting_shot"   # normalised labels
    good = _json.loads(_resp())
    for bad in [dict(good, exciting="yes"),                                       # string bool
                dict(good, caption="x"),                                          # caption too short
                dict(good, event_type="  "),                                      # empty label
                dict(good, event_description="Unclear."),                         # too short
                {k: v for k, v in good.items() if k != "event_type"},             # missing field
                dict(good, extra=1)]:                                             # extra key
        try:
            V.VLMResult.model_validate_json(_json.dumps(bad)); raise AssertionError(bad)
        except ValidationError:
            pass
    assert set(V.GeminiSchema.model_json_schema()["properties"]) == FIELDS
    assert all(f.description for f in V.GeminiSchema.model_fields.values())
    assert all(f.description for f in V.VLMResult.model_fields.values())
    assert V.PROMPT.startswith("You are an expert sports-video event analyst") and V.PROMPT != V.PROMPT_RETRY
    assert "event_type" in V.PROMPT_RETRY and "general_play" in V.PROMPT_RETRY


def test_vlm_specific_event_and_uncertain_event():
    cfg = Config(); cfg.vlm_model = "none"; cfg.vlm_retry_delay_sec = 0
    v = VLM(cfg); v.enabled = True
    v._generate = lambda img, prompt: _resp(                                       # specific event (cricket, graphic-supported)
        caption="The batter finishes a drive as the ball races to the rope.", event_type="boundary_four",
        event_description="A readable FOUR graphic and the ball near the boundary rope support a boundary four.")
    r, st = v.analyse("x.jpg")
    assert st == "ok" and r.sport == "cricket" and r.event_type == "boundary_four" and r.exciting is True
    v._generate = lambda img, prompt: _resp(                                       # generic scene -> uncertain, honest limitation
        caption="A fielder in a dark uniform stands on the outfield.", event_type="uncertain",
        event_description="No ball or clear action is visible, so the specific event cannot be determined from this frame.",
        players_and_actions="One fielder stands still.", exciting=False, reason="Nothing notable is visible.")
    r, st = v.analyse("x.jpg")
    assert st == "ok" and r.event_type == "uncertain" and r.exciting is False and "cannot be determined" in r.event_description
    v._generate = lambda img, prompt: _resp(sport="basketball", event_type="dunk", caption="A player rises to dunk at the rim.",
                                            event_description="A player is above the rim with the ball near the basket.")
    r, st = v.analyse("x.jpg")                                                    # another sport, same taxonomy
    assert (r.sport, r.event_type) == ("basketball", "dunk")


def test_vlm_retry_then_fallback():
    from pipeline import vlm as V
    cfg = Config(); cfg.vlm_model = "none"; cfg.vlm_retry_delay_sec = 0
    v = VLM(cfg); v.enabled = True; calls = []
    def boom(img, prompt): calls.append(prompt); raise TimeoutError("timeout")
    v._generate = boom                                                            # API error twice -> fallback
    res, status = v.analyse("x.jpg")
    assert status == "fallback" and calls == [V.USER_TEXT, V.PROMPT_RETRY]        # exactly one retry
    assert res.exciting is False and res.caption.startswith("[fallback]") and res.reason.startswith("[fallback]")
    assert res.sport == "unknown" and res.event_type == "uncertain" and res.event_description.startswith("[fallback]")
    out = {V.USER_TEXT: "not json", V.PROMPT_RETRY: _resp()}
    v._generate = lambda img, prompt: out[prompt]                                 # invalid first, retry works
    r, st = v.analyse("x.jpg")
    assert st == "retry_ok" and r.event_type == "batting_shot"
    v._generate = lambda img, prompt: _resp(exciting=False)
    r, st = v.analyse("x.jpg")
    assert st == "ok" and r.exciting is False
    assert v.stats == {"ok": 1, "retry_ok": 1, "fallback": 1}
    assert VLMResult_fields(r) == FIELDS


def test_vlm_missing_key_falls_back(monkeypatch=None):
    import os
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        v = VLM(Config())
        assert v.enabled is False and v.analyse("x.jpg")[1] == "fallback"
    finally:
        if old: os.environ["GEMINI_API_KEY"] = old


def VLMResult_fields(r):
    return set(r.model_dump())


def test_vlm_boolean_changes_score():
    cfg = Config()
    mk = lambda e: [dict(shot_id=i, start=i, end=i + 1, motion_score=1.0, avg_people=1, vlm_exciting=(e if i == 0 else False)) for i in range(2)]
    a, b = mk(True), mk(False)
    highlights.score_shots(a, cfg); highlights.score_shots(b, cfg)
    assert a[0]["final_score"] - b[0]["final_score"] == round(cfg.w_vlm, 4)


def test_selection_no_overlap():
    cfg = Config()
    shots = [dict(shot_id=i, start=i * 5, end=i * 5 + 5, motion_score=m, avg_people=1, vlm_exciting=e)
             for i, (m, e) in enumerate([(1, True), (5, False), (3, True), (2, False), (4, True)])]
    highlights.score_shots(shots, cfg)
    sel = highlights.select_top(shots, cfg)
    assert len(sel) == 3 and [s["start"] for s in sel] == sorted(s["start"] for s in sel)


def _fake_detect(frames_dir, cfg):
    paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    return {i: [("person", 0.9, (50, 50, 300, 600))] * (1 + i % 3) + [("sports ball", .8, (10, 10, 40, 40))]
            for i in range(len(paths))}, paths


def test_full_run_with_fake_yolo(tmp_path=None):
    import tempfile, importlib
    out = Path(tmp_path or tempfile.mkdtemp())
    src = out / "in.mp4"          # no-audio 3-scene video made with ffmpeg
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=4:r=25",
                    "-f", "lavfi", "-i", "smptebars=s=640x360:d=4:r=25", "-f", "lavfi", "-i", "rgbtestsrc=s=640x360:d=4:r=25",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", src], check=True)
    detect.detect_frames = _fake_detect
    main = importlib.import_module("main")
    sys.argv = ["main.py", "--input", str(src), "--out", str(out / "res"), "--vlm-model", "none"]
    main.main()
    s = json.loads((out / "res" / "summary.json").read_text())
    assert s["vlm_stats"]["fallback_count"] == s["vlm_stats"]["total"] and s["vlm_stats"]["success_rate"] == 0 and s["video"]["has_audio"] is False and len(s["highlight_shot_ids"]) >= 1
    assert all({"vlm_caption", "vlm_exciting", "vlm_reason", "vlm_status", "vlm_ok", "final_score", "keyframe", "keyframe_clean", "common_objects", "vlm_sport", "vlm_event_type", "vlm_event_description", "vlm_players_and_actions"} <= set(r) for r in s["shots"])
    assert all(h["vlm_event_type"] == "uncertain" for h in s["highlight_events"]) and len(s["highlight_events"]) == len(s["highlight_shot_ids"])
    assert (out / "res" / "highlights.mp4").exists() and list((out / "res" / "keyframes").glob("*.jpg"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
