# Smart Video Highlight & Insight Pipeline

A Python project that takes a short sports or event video and creates:

- A short highlight reel (`highlights.mp4`)
- A JSON report describing the shots, objects, actions, and possible events (`summary.json`)
- Keyframes and optional tracking/audio-analysis files
- An optional Streamlit dashboard to run the pipeline and view its results

The pipeline uses FFmpeg, OpenCV, YOLO11n, and the Gemini API. A GPU is optional; CPU processing is supported but may be slower.

## 1. What the pipeline does

1. **Reads the video:** FFprobe reads the video details, such as duration, FPS, resolution, and codec.
2. **Prepares the media:** FFmpeg creates a 720p working video, extracts frames at 2 FPS, and converts audio to 16 kHz mono WAV.
3. **Finds shots:** OpenCV compares HSV colour histograms to estimate where shots change.
4. **Measures movement:** Farneback optical flow estimates movement in each shot. Camera movement can also increase this score.
5. **Detects people and objects:** pretrained YOLO11n detects people and common COCO objects. The pipeline saves annotated keyframes.
6. **Describes possible events:** Gemini analyses one clean keyframe per shot and returns structured fields such as sport, event type, event description, player actions, caption, excitement decision, and reason. Pydantic validates the response. The pipeline retries once and uses a fallback result if the request still fails.
7. **Ranks highlights:** a simple score combines movement, people count, and Gemini's excitement decision.
8. **Creates the reel:** the pipeline selects up to three non-overlapping clips, each up to six seconds long, joins them, burns captions into the video, and preserves the original audio.

**Important:** Gemini receives one keyframe per shot, not the full action over time. Treat event labels as visual estimates, not guaranteed proof that a goal, wicket, boundary, or other outcome happened.

## 2. Requirements

- Python 3.9 or newer
- FFmpeg and FFprobe installed and available on your system PATH
- A Gemini API key for event descriptions
- Internet access for Gemini requests and the initial YOLO weights download
- GPU optional

## 3. Install the project

Open a terminal in the project folder—the folder containing `main.py` and `requirements.txt`.

### Install FFmpeg

Install FFmpeg for your operating system and make sure both `ffmpeg` and `ffprobe` work in a new terminal. Check with:

```bash
ffmpeg -version
ffprobe -version
```

If either command is not recognised, install FFmpeg or add its `bin` folder to PATH, then reopen the terminal.

### Create a Python environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or follow your organisation's Python environment policy.

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure the Gemini API key

The API key is needed for Gemini's event-description step. Create a key through Google AI Studio, then set it as an environment variable. **Never put your real key in this README, source code, screenshots, or GitHub.**

**Windows PowerShell** (run this in the same terminal you will use to start the project):

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
$env:GEMINI_MODEL = "gemini-3.1-flash-lite"
```

**Linux/macOS:**

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
export GEMINI_MODEL="gemini-3.1-flash-lite"
```

`GEMINI_MODEL` is optional; the code's documented default is `gemini-3.1-flash-lite`. Use a model that is enabled for your Google account. Model names, quotas, and free-tier availability can change.

Optional setting: `GEMINI_MIN_INTERVAL` controls the delay between Gemini calls. The documented default is `4.5` seconds. It helps reduce rate-limit errors but can make processing slower.

The environment variables above are temporary for that terminal session. Set them again in a new terminal. In Google Colab, store the key in Colab Secrets as `GEMINI_API_KEY`, then read it into the environment from the notebook; do not paste the key into a notebook that you share.

## 5. Add a video and run the command-line pipeline

Put your input video in the `input` folder. For example:

```text
input/video.mp4
```

Run the baseline pipeline from the project root:

```bash
python main.py --input input/video.mp4 --out results
```

Run with optional player tracking and audio loudness analysis:

```bash
python main.py --input input/video.mp4 --out results --tracking --audio-analysis
```

To check all command-line options supported by your local copy, run:

```bash
python main.py --help
```

The output folder is created if needed. Processing time depends on video duration, number of shots, hardware, Gemini response time, and API limits.

## 6. Run the Streamlit dashboard

The Streamlit dashboard is optional. It provides a browser interface for uploading a video, running the pipeline, and viewing the generated results. The command-line pipeline works without Streamlit.

First, activate the same Python environment used above, then install the dashboard dependencies:

```bash
pip install -r requirements-app.txt
```

Start the dashboard from the project root:

```bash
streamlit run app.py
```

Streamlit normally prints a local URL such as `http://localhost:8501`. Open that address in your browser. Upload a video in the app and follow the on-screen controls to start processing. When processing finishes, use the dashboard to view the highlight reel, event information, shot table, keyframes, tracking details, audio peaks, and available JSON downloads.

Stop the dashboard by returning to the terminal and pressing `Ctrl+C`.

The dashboard runs `main.py` as a subprocess rather than maintaining a separate copy of the processing logic. It handles one job at a time and writes run data to a temporary folder or `./runs/<name>`, depending on the app configuration.

## 7. Run the tests

From the project root, with the Python environment activated:

```bash
python tests/test_pipeline.py
python tests/test_bonus.py
```

These are project test scripts. A test may need dependencies or YOLO weights to be available. Read the output and fix any reported errors before submitting. Also run the full pipeline on a real, appropriately licensed video; passing tests alone does not prove the complete video workflow works.

## 8. Output files

The standard output folder is `results/`.

| File/folder | What it contains |
|---|---|
| `highlights.mp4` | Final joined highlight reel with captions and original audio |
| `summary.json` | Video details, shot details, event descriptions, scores, chosen highlights, runtime, and feature status |
| `frames/` | Frames extracted during video preparation |
| `keyframes/` | Keyframes with object detections; clean keyframes are also used for Gemini analysis |
| `tracking.json` | Tracking statistics, when tracking is enabled |
| `tracking.mp4` | Video with tracking boxes, IDs, and short trails, when supported by the tracking run |
| `audio_analysis.json` | Audio loudness series and detected peaks, when audio analysis is enabled |
|

`summary.json` contains top-level information such as `video`, `scoring`, `vlm_model`, `vlm_stats`, `highlight_shot_ids`, `highlight_events`, `shots`, `runtime_sec`, and `features`. Each shot can include its start/end times, motion score, average people count, common objects, keyframe paths, Gemini caption/event fields, excitement decision, reason, status, and final score. Optional tracking and audio fields appear when those features are enabled.

## 9. How highlight scores work

The default score is:

```text
0.5 × normalized motion + 0.2 × normalized people count + 0.3 × Gemini excitement
```

- **Motion (50%):** more estimated movement can raise the score.
- **People count (20%):** more detected people can raise the score.
- **Gemini excitement (30%):** Gemini's yes/no excitement result contributes to the score.

This is a simple heuristic, not a model trained to identify the best sports moments. A camera pan, crowd shot, or generic play can rank highly. The default selection chooses up to three non-overlapping clips, each no longer than six seconds.

## 10. Optional features

### Player tracking

Enable with `--tracking`:

```bash
python main.py --input input/video.mp4 --out results --tracking
```

The tracker reuses YOLO detections rather than running a second object-detection model. ByteTrack is used when available; otherwise, a simpler IoU-based tracker may be used. Tracking is limited to people. IDs reset at each shot cut, so they are **shot-local tracking IDs**, not permanent identities. At 2 FPS, fast movement can cause missed detections, broken tracks, and ID switches.

### Audio loudness analysis

Enable with `--audio-analysis`:

```bash
python main.py --input input/video.mp4 --out results --audio-analysis
```

The pipeline measures audio RMS loudness in short windows and records possible loudness peaks. A peak might be caused by commentary, crowd noise, or music; it is not proof that an important event occurred. The original audio is retained in the final reel.

The optional `--experimental-audio-score` flag reports a separate comparison ranking that adds a small audio-peak contribution. It does not replace the default score or the default highlight selection. Check `python main.py --help` for the exact combination of flags supported by your copy.

### Other command-line options

The project documents these options; use `python main.py --help` to confirm their exact syntax:

- `--vlm-model`: choose or disable the VLM; `none` skips Gemini.
- `--yolo-weights`: choose the YOLO weights file.
- `--cut-threshold`: change the shot-change threshold.
- `--top-k`: change how many highlight clips are selected.
- `--tracking`: enable tracking.
- `--no-tracking-video`: disable writing the tracking-annotated video.
- `--audio-analysis`: enable audio loudness analysis.
- `--experimental-audio-score`: report an experimental audio-adjusted ranking.

Thresholds and other settings are configured in `pipeline/config.py`.

## 11. Docker (optional)

Docker support is optional; you do not need Docker to run the project with Python. If Docker is installed and the repository's `Dockerfile` is present, build the image from the project root:

```bash
docker build -t highlights .
```

Example CLI run from a shell where `input/video.mp4` exists:

```bash
docker run --rm -e GEMINI_API_KEY -v "$PWD/input:/data/in:ro" -v "$PWD/results:/data/out" highlights --input /data/in/video.mp4 --out /data/out --tracking --audio-analysis
```

To launch the dashboard inside the container:

```bash
docker run --rm -p 8501:8501 -e GEMINI_API_KEY --entrypoint streamlit highlights run app.py --server.address 0.0.0.0
```

Open `http://localhost:8501` in your browser. Docker GPU support has not been verified, and the Docker image should be built and tested on your own system before relying on these commands. The volume syntax above is for Unix-like shells; Windows PowerShell may require adjusted volume paths.

## 12. Settings and current thresholds

- Working video: 720p
- Extracted frames: 2 FPS
- Extracted audio: 16 kHz mono WAV
- Shot detection: HSV histogram distance threshold `0.45`
- Short shots under 1 second are merged; shots over 8 seconds are split
- YOLO11n: pretrained COCO model, confidence threshold `0.35`, batches of 16
- Gemini: one clean keyframe per shot, structured response validation, one retry, then fallback
- Highlight selection: up to 3 non-overlapping clips, up to 6 seconds each

These values are implementation settings, not universal best values. Shot detection was checked on a limited video sample and may need adjustment for other footage.

## 13. Runtime and limitations

On the 239-second development cricket clip, a baseline run took about 11.5 minutes in Google Colab with GPU-assisted YOLO. Gemini API calls were the slowest part. Runtime will vary with the number of shots, API rate limits, retries, and hardware. Tracking and audio analysis were tested only on a short synthetic clip and have not been benchmarked on every full-length video.

Known limitations:

- A single image per shot cannot show the full sequence of an event or reliably prove its outcome.
- Gemini's excitement decision can be overused and should be checked against the generated summary and reel.
- Histogram shot detection can miss gradual transitions or add cuts during camera pans/zooms.
- Optical flow also measures camera movement, not just player movement.
- Tracking IDs are short-lived box associations, not person recognition.
- Audio peaks can be caused by non-event sounds.
- Gemini model availability and API quotas can change.

## 14. Video source and licence

For the final demonstration, use a sports/event clip that you are allowed to use, such as a royalty-free clip from Pexels or Pixabay. In your submission, record the **exact video URL, creator, and licence** for the clip you actually tested. Do not describe the ICC broadcast used during development as royalty-free; it is copyrighted and is not included in this repository.

## 15. Before submitting

- [ ] Install dependencies and verify `ffmpeg -version` and `ffprobe -version`.
- [ ] Set `GEMINI_API_KEY` without putting it in the repository.
- [ ] Run `python main.py --help` and confirm the documented options.
- [ ] Run both test scripts.
- [ ] Run the pipeline on a royalty-free/licensed video and inspect `highlights.mp4` and `summary.json`.
- [ ] Try `pip install -r requirements-app.txt` and `streamlit run app.py` if submitting the dashboard.
- [ ] Check `.gitignore` excludes API keys, `.env`, virtual environments, large input videos, model weights, and generated results unless you intentionally want to include permitted demo outputs.
