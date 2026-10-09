"""Optional Streamlit dashboard:  pip install streamlit  &&  streamlit run app.py
It only launches the existing CLI (main.py) as a subprocess and displays results/ - no pipeline logic lives here."""
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st

import dashboard_utils as U

st.set_page_config(page_title="Smart Video Highlight Pipeline", layout="wide")


@st.cache_resource
def _job_lock():
    return threading.Lock()          # one expensive job at a time across all browser sessions of this server


def run_job(cmd, env, secret):
    """Run main.py, stream its log into the page, return (returncode, log_text)."""
    lines, box = [], st.empty()
    proc = subprocess.Popen(cmd, cwd=str(U.ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(U.redact(line.rstrip(), secret))
        box.code("\n".join(lines[-14:]), language="text")
    return proc.wait(), "\n".join(lines)


st.title("Smart Video Highlight & Insight Pipeline")
with st.sidebar:
    st.header("Run settings")
    use_gemini = st.checkbox("Use Gemini VLM", value=True, help="Unchecked = skip the VLM (all shots get the fallback).")
    api_key = st.text_input("GEMINI_API_KEY (optional)", type="password",
                            help="Leave empty to use the GEMINI_API_KEY environment variable. Never stored or shown.")
    tracking = st.checkbox("Player tracking (ByteTrack)", value=False)
    audio = st.checkbox("Audio loudness analysis", value=False)
    exp_audio = st.checkbox("Experimental audio-weighted ranking (reported separately)", value=False, disabled=not audio)
    folder = st.text_input("Output folder name inside ./runs/ (empty = temporary directory)", value="")

up = st.file_uploader("Upload a 2-5 minute video", type=[e.strip(".") for e in sorted(U.ALLOWED_EXT)])
lock = _job_lock()

if st.button("Run pipeline", type="primary", disabled=up is None):
    err = U.validate_upload(up.name, up.size)
    if err:
        st.error(err)
    elif not lock.acquire(blocking=False):
        st.warning("Another pipeline job is already running. Please wait for it to finish.")
    else:
        try:
            out_dir = U.safe_run_dir(folder.strip() or None)
            video = out_dir / ("input" + Path(up.name).suffix.lower())
            video.write_bytes(up.getbuffer())
            cmd = U.build_command(video, out_dir / "results", tracking, audio, exp_audio, None if use_gemini else "none")
            t0 = time.time()
            with st.status("Running the pipeline (FFmpeg -> OpenCV -> YOLO -> Gemini -> highlights)...", expanded=True) as status:
                code, log = run_job(cmd, U.child_env(api_key.strip() or None), api_key.strip())
                if code == 0:
                    status.update(label=f"Done in {time.time() - t0:.0f}s", state="complete")
                    st.session_state["results"] = str(out_dir / "results")
                else:
                    status.update(label=f"Pipeline failed (exit code {code})", state="error")
                    st.error("The pipeline failed. See the log above (the last lines usually name the cause).")
        except Exception as e:                      # visible, never a blank page
            st.error(f"Could not start the pipeline: {e}")
        finally:
            lock.release()

res = st.session_state.get("results")
if res and (Path(res) / "summary.json").exists():
    R = Path(res)
    S = U.load_json(R / "summary.json")
    st.success(f"Results in: {R}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Highlight reel")
        if (R / "highlights.mp4").exists():
            st.video(str((R / "highlights.mp4").resolve()))
    with right:
        st.subheader("Video & VLM")
        st.json({"video": S["video"], "vlm_model": S["vlm_model"], "vlm_stats": S["vlm_stats"], "features": S.get("features"),
                 "scoring": S["scoring"]["formula"]}, expanded=False)

    tabs = st.tabs(["Highlight events", "All shots", "Audio peaks", "Tracking", "Files & keyframes"])
    with tabs[0]:
        for r in [x for x in S["shots"] if x["shot_id"] in S["highlight_shot_ids"]]:
            c1, c2 = st.columns([1, 2])
            kf = R / r["keyframe"]
            if kf.exists():
                c1.image(str(kf), caption=f"shot {r['shot_id']} ({r['start']:.1f}-{r['end']:.1f}s)")
            c2.markdown(f"**{r['vlm_sport']} / {r['vlm_event_type']}**  |  exciting: **{r['vlm_exciting']}**  |  score {r['final_score']}")
            c2.write(r["vlm_caption"]); c2.caption(r["vlm_event_description"]); c2.caption("Reason: " + r["vlm_reason"])
            if r.get("vlm_players_and_actions"):
                c2.caption("Players/actions: " + r["vlm_players_and_actions"])
        if S.get("experimental_audio_ranking"):
            st.info("Experimental audio-weighted ranking (reported only; the reel above uses the default score): "
                    f"{S['experimental_audio_ranking']['top_shot_ids']} vs default {S['experimental_audio_ranking']['default_top_shot_ids']}")
    with tabs[1]:
        st.dataframe(U.shots_table(S), use_container_width=True)
    with tabs[2]:
        A = U.load_json(R / "audio_analysis.json")
        if not A:
            st.info(f"Audio analysis not available ({S.get('features', {}).get('audio_analysis', 'off')}). Enable it in the sidebar.")
        else:
            st.caption(A["notes"][1])
            st.line_chart({"level (dBFS)": {t: d for t, d in A["loudness_series"]}})
            st.dataframe(A["peaks"], use_container_width=True)
    with tabs[3]:
        T = U.load_json(R / "tracking.json")
        if not T:
            st.info(f"Tracking not available ({S.get('features', {}).get('tracking', 'off')}). Enable it in the sidebar.")
        else:
            st.caption(T["notes"][0]); st.caption(T["notes"][1])
            st.write(f"Backend: **{T['backend']}** | shot-local person tracks: **{T['total_tracks']}**")
            if (R / "tracking.mp4").exists():
                st.video(str((R / "tracking.mp4").resolve()))
            st.dataframe([{"shot": s["shot_id"], "tracks": s["num_people_tracks"], "max_concurrent": s["max_concurrent_tracked"],
                           "mean_observed_s": s["mean_observed_duration_sec"]} for s in T["shots"]], use_container_width=True)
    with tabs[4]:
        for name in ("summary.json", "tracking.json", "audio_analysis.json", "highlights.mp4"):
            p = R / name
            if p.exists():
                st.download_button(f"Download {name}", p.read_bytes(), file_name=name, key=name)
        kfs = sorted((R / "keyframes").glob("*.jpg"))
        st.write(f"{len(kfs)} annotated keyframes")
        if kfs:
            st.image([str(k) for k in kfs[:24]], caption=[k.stem for k in kfs[:24]], width=200)
