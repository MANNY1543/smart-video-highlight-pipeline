"""FFmpeg / ffprobe helpers: probing, preparation, cutting, concatenation."""
import json
import logging
import shutil
import subprocess
import textwrap
from pathlib import Path

log = logging.getLogger(__name__)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "C:/Windows/Fonts/arial.ttf",                         # Windows
]


def run(cmd):
    """Run a command; on failure log FFmpeg's stderr tail and raise."""
    log.debug("CMD: %s", " ".join(str(c) for c in cmd))
    try:
        return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s\n%s", cmd[0], e.stderr[-1500:])
        raise


def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH. Install FFmpeg first.")


def probe(path):
    """ffprobe -> duration, fps, resolution, codec (+ whether audio exists)."""
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", path]).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("No video stream found in input")
    num, den = video["r_frame_rate"].split("/")
    info = {
        "duration_sec": round(float(data["format"]["duration"]), 2),
        "fps": round(float(num) / float(den), 3),
        "width": video["width"],
        "height": video["height"],
        "video_codec": video["codec_name"],
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }
    log.info("ffprobe: %s", info)
    return info


def prepare(input_path, out_dir, cfg, has_audio):
    """Resize to 720p, extract frames at 2 FPS, extract 16 kHz mono WAV."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.jpg"):   # avoid mixing frames from a previous run
        old.unlink()

    video_720p = out_dir / "video_720p.mp4"
    run(["ffmpeg", "-y", "-i", input_path, "-vf", f"scale=-2:{cfg.target_height}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", video_720p])
    log.info("Wrote %s", video_720p)

    # frame i (1-based) sits at t = (i-1)/frame_fps
    run(["ffmpeg", "-y", "-i", video_720p, "-vf", f"fps={cfg.frame_fps}", "-q:v", "2",
         frames_dir / "frame_%05d.jpg"])
    n = len(list(frames_dir.glob("frame_*.jpg")))
    log.info("Extracted %d frames at %s FPS", n, cfg.frame_fps)

    audio = None
    if has_audio:
        audio = out_dir / "audio_16k_mono.wav"
        run(["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", cfg.audio_rate, audio])
        log.info("Wrote %s", audio)
    else:
        log.warning("Input has no audio track; skipping WAV extraction")
    return video_720p, frames_dir, audio


def _font():
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    log.warning("No known font file found; relying on FFmpeg/fontconfig default")
    return None


def _esc(path):
    """Escape a path for use inside an FFmpeg filter argument."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def cut_clip(src, start, duration, caption, out_path, cfg):
    """Cut [start, start+duration] from src, burn caption with drawtext, keep audio."""
    out_path = Path(out_path)
    txt = out_path.with_suffix(".txt")                       # caption goes in a file -> no quote escaping
    txt.write_text(textwrap.fill(caption, 55), encoding="utf-8")
    font = _font()
    font_opt = f"fontfile='{_esc(font)}':" if font else ""
    vf = (f"scale=-2:{cfg.target_height},"
          f"drawtext={font_opt}textfile='{_esc(txt.resolve())}':fontcolor=white:fontsize=26:"
          f"box=1:boxcolor=black@0.6:boxborderw=10:x=(w-text_w)/2:y=h-text_h-40")
    # Same codec settings for every clip so they can be concatenated without re-encoding.
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", src, "-t", f"{duration:.2f}",
         "-vf", vf, "-map", "0:v:0", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "30",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", out_path])


def concat(clips, out_path):
    lst = Path(out_path).with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in clips))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out_path])
