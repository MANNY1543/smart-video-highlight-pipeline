"""All tunable values live here, so thresholds are easy to find and change."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Media (FFmpeg) ---
    frame_fps: float = 2.0            # frames extracted for YOLO / keyframes
    target_height: int = 720
    audio_rate: int = 16000

    # --- Shot detection + motion (OpenCV) ---
    analysis_fps: float = 5.0         # video is sampled at this rate for histograms / flow
    analysis_width: int = 160         # frames are shrunk to this width (speed)
    cut_threshold: float = 0.45       # Bhattacharyya distance between HSV hists; above = new shot
    min_shot_sec: float = 1.0         # shorter shots are merged into the previous one
    max_shot_sec: float = 8.0         # longer shots are split (handles clips with few/no cuts)

    # --- Detection (YOLO) ---
    yolo_weights: str = "yolo11n.pt"  # auto-downloaded by ultralytics; "yolov8n.pt" also works
    yolo_conf: float = 0.35
    yolo_batch: int = 16
    top_objects: int = 3              # most common non-person classes kept per shot

    # --- VLM (hosted Gemini; key only from env GEMINI_API_KEY) ---
    vlm_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"))  # or "none"
    vlm_timeout_sec: float = 30.0
    vlm_max_output_tokens: int = 800
    vlm_retry_delay_sec: float = 3.0   # pause before the single retry (helps with 429s)
    vlm_min_interval_sec: float = field(default_factory=lambda: float(os.environ.get("GEMINI_MIN_INTERVAL", "4.5")))  # free-tier RPM guard

    # --- Optional bonus: person tracking (ByteTrack on the existing detections) ---
    tracking: bool = False            # off by default; enable with --tracking
    tracking_video: bool = True       # results/tracking.mp4 (drawing only, no extra inference)
    track_high_thresh: float = 0.5    # confident detections: first association + may start a track
    track_buffer: int = 4             # SAMPLED frames (= 2 s at 2 FPS) a lost track is kept
    track_match_thresh: float = 0.9   # ByteTrack cost threshold (1-IoU); loose because people move far in 0.5 s
    trail_len: int = 6                # sampled frames of movement trail drawn

    # --- Optional bonus: audio loudness (NumPy on the extracted 16 kHz WAV) ---
    audio_analysis: bool = False      # off by default; enable with --audio-analysis
    audio_window_sec: float = 0.5
    audio_hop_sec: float = 0.25
    audio_peak_k: float = 3.0         # peak = window level > median + k * (1.4826 * MAD) of the clip
    audio_min_peak_gap_sec: float = 1.0
    audio_min_prominence_db: float = 3.0   # ...and at least this many dB above the clip median
    audio_experimental_score: bool = False # separate, optional score; default ranking is unchanged
    w_audio_exp: float = 0.15

    # --- Highlight score (heuristic weights, sum to 1) ---
    w_motion: float = 0.5
    w_people: float = 0.2
    w_vlm: float = 0.3
    top_k: int = 3
    max_clip_sec: float = 6.0         # each highlight segment is at most this long
