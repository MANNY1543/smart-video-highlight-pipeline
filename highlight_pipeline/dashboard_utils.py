"""Streamlit-free helpers for app.py (so they can be unit-tested and the CLI never needs Streamlit)."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD_MB = 500


def validate_upload(name, size_bytes):
    """-> error string, or None if the upload looks acceptable (the real check is ffprobe after saving)."""
    ext = Path(name or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        return f"Unsupported file type '{ext or name}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
    if size_bytes <= 0:
        return "The uploaded file is empty."
    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        return f"File is larger than {MAX_UPLOAD_MB} MB."
    return None


def safe_run_dir(folder_name=None, runs_root=None):
    """Temporary directory by default; otherwise ONLY a plain folder name inside ./runs/ (no paths, no '..')."""
    if not folder_name:
        return Path(tempfile.mkdtemp(prefix="highlights_"))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", folder_name):
        raise ValueError("Folder name may only contain letters, digits, '_' and '-' (it is created inside ./runs/).")
    root = Path(runs_root) if runs_root else ROOT / "runs"
    d = root / folder_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_command(video, out_dir, tracking=False, audio=False, experimental_audio=False, vlm_model=None, top_k=None):
    """Command line for the EXISTING CLI (main.py) - the dashboard does not duplicate any pipeline logic."""
    cmd = [sys.executable, str(ROOT / "main.py"), "--input", str(video), "--out", str(out_dir)]
    if tracking:
        cmd.append("--tracking")
    if audio:
        cmd.append("--audio-analysis")
    if experimental_audio:
        cmd.append("--experimental-audio-score")
    if vlm_model:
        cmd += ["--vlm-model", vlm_model]
    if top_k:
        cmd += ["--top-k", str(int(top_k))]
    return cmd


def child_env(api_key=None):
    """Environment for the subprocess. A key typed into the dashboard is passed only via the environment."""
    env = dict(os.environ)
    if api_key:
        env["GEMINI_API_KEY"] = api_key
    return env


def redact(line, *secrets):
    for s in secrets:
        if s:
            line = line.replace(s, "***")
    return line


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def shots_table(summary):
    """Flat rows for a table view of summary.json (works whether or not optional features were enabled)."""
    rows = []
    for r in summary.get("shots", []):
        rows.append({
            "shot": r["shot_id"], "start": r["start"], "end": r["end"], "motion": r["motion_score"], "avg_people": r["avg_people"],
            "objects": ", ".join(r.get("common_objects", [])), "sport": r.get("vlm_sport"), "event_type": r.get("vlm_event_type"),
            "exciting": r["vlm_exciting"], "score": r["final_score"],
            "selected": r["shot_id"] in summary.get("highlight_shot_ids", []),
            "audio_peaks": (r.get("audio") or {}).get("peak_count"),
            "people_tracks": (r.get("tracking") or {}).get("num_people_tracks"),
            "vlm_status": r.get("vlm_status")})
    return rows
