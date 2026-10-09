"""Isolated Gemini VLM check on real keyframes. Run this BEFORE the full pipeline.

   export GEMINI_API_KEY=...        (Windows PowerShell: $env:GEMINI_API_KEY="...")
   python tests/vlm_isolated.py results/frames/frame_00020.jpg results/frames/frame_00100.jpg results/frames/frame_00300.jpg
   python tests/vlm_isolated.py results/frames          # a directory: picks 3 evenly spread frames

Calls the Gemini analyzer on exactly 3 different keyframes and prints, for each, the raw structured Gemini
response, then the validated caption / exciting / reason. Exit code 0 only if none fell back."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.basicConfig(level=logging.DEBUG if "--debug" in sys.argv else logging.WARNING,
                                    format="%(levelname)s %(message)s")
from pipeline.config import Config
from pipeline.vlm import VLM

N = 3


def pick(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.jpg")) if p.is_dir() else [p]
    files = list({f.resolve(): f for f in files}.values())                  # de-duplicate
    if len(files) > N:
        files = [files[round(i * (len(files) - 1) / (N - 1))] for i in range(N)]    # evenly spread
    return files


ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+", help="keyframe image paths, or a directory of .jpg frames")
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()
files = pick(args.paths)
missing = [f for f in files if not f.exists()]
if missing or len(files) != N:
    sys.exit(f"Need exactly {N} different existing keyframes (got {len(files)}; missing: {[str(m) for m in missing]})")

cfg = Config()
vlm = VLM(cfg)
if not vlm.enabled:
    sys.exit("Gemini VLM did not start - set GEMINI_API_KEY (see the error above).")
print(f"model: {cfg.vlm_model}")
bad = 0
for f in files:
    res, status = vlm.analyse(f)
    print(f"\n=== {f}\nstatus: {status}   (Gemini calls: {len(vlm.last_raw) or 'failed'})")
    for i, raw in enumerate(vlm.last_raw, 1):
        print(f"raw Gemini structured response [{i}]: {raw}")
    print(f"caption : {res.caption}\nsport   : {res.sport}\nevent   : {res.event_type}\nevent_description  : {res.event_description}"
          f"\nplayers_and_actions: {res.players_and_actions}\nexciting: {res.exciting}\nreason  : {res.reason!r}")
    bad += status == "fallback"
print("\nstats:", vlm.stats)
print("RESULT:", "PASS" if not bad else "CHECK OUTPUT ABOVE")
sys.exit(1 if bad else 0)
