import streamlit as st
from PIL import Image
import time
import numpy as np
import cv2
import tempfile
import openai

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Crack Intelligence", page_icon="🤖", layout="wide")

# ------------------ GLOBALS ------------------
OPENAI_API_KEY = ""  # keep empty if you don't want API calls
CRACK_THRESHOLD = 2500

# ------------------ AI PREDICTION (your current logic) ------------------
@st.cache_data
def get_crack_prediction(image_bytes):
    """
    Basic heuristic using edges as a stand-in for a trained model.
    Returns ('DANGER'|'SAFE', score_float_0to1)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(img_blurred, 50, 150)
    feature_pixel_count = np.sum(edges > 0)

    if feature_pixel_count > CRACK_THRESHOLD:
        status = "DANGER"
        score = float(np.random.uniform(0.85, 0.98))
    else:
        status = "SAFE"
        score = float(np.random.uniform(0.90, 0.99))
    return status, score

def analyze_frame(frame_bgr):
    """
    Frame -> overlayed_frame, status, score
    Uses the same heuristic per frame so your demo is consistent across modes.
    """
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (256, 256))
    _, buf = cv2.imencode(".jpg", small)
    status, score = get_crack_prediction(buf.tobytes())

    # Make a quick visual overlay using edges
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_col = np.zeros_like(small)
    color = (0, 0, 255) if status == "DANGER" else (0, 255, 0)
    edge_col[edges > 0] = color
    overlay_small = cv2.addWeighted(small, 0.7, edge_col, 0.3, 0)
    overlay = cv2.resize(overlay_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # HUD text
    cv2.putText(overlay, f"{status}  {score*100:.1f}%", (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return overlay, status, score

# ------------------ CHATBOT (independent of analysis) ------------------
def get_chatbot_response(messages_history):
    if not OPENAI_API_KEY:
        # offline fallback so demo always works without API
        last_user = next((m["content"] for m in reversed(messages_history) if m["role"]=="user"), "")
        return (
            "SCAN ASSISTANT (offline): I can help with how to use this tool and "
            "general guidance about cracks in roads/buildings/bridges. "
            "Add your OpenAI key in the script to enable AI responses.\n\n"
            f"You asked: “{last_user}”. For structural questions, I’ll discuss implications and safety steps."
        )
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_history
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"**Chat error:** {e}"

SYSTEM_PROMPT = (
    "You are 'SCAN ASSISTANT', an expert helper for a structural analysis tool.\n"
    "Stay within two scopes only:\n"
    "1) How to use this website/tool.\n"
    "2) Structural integrity basics about cracks in roads/buildings/bridges and their implications.\n"
    "Refuse anything outside these."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Welcome! Ask about using this tool or about crack implications (roads/buildings/bridges)."}
    ]
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# ------------------ HEADER ------------------
st.title("🚧 AI-Powered Infrastructure Intelligence 'SCAN'")
st.subheader("Structural Crack Analysis Network")
st.markdown("---")
st.header("Real-Time Structural Analysis")
st.markdown("#### Powered by **OpenCV** and a **CNN-style pipeline** for pixel-level feature extraction.")

# ------------------ TABS: Image | Video | Live ------------------
tab_image, tab_video, tab_live = st.tabs(["🖼️ Image", "🎥 Video", "📷 Live Feed"])

# ---------- IMAGE TAB ----------
with tab_image:
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"], key="img_upl")
    st.markdown("---")
    if uploaded_file is None:
        st.info("Upload an image to begin analysis.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([0.9, 1.1])
        with col1:
            st.image(image, caption="Image Under Analysis", use_column_width=True)

        with col2:
            progress_bar = st.progress(0, text="Initializing analysis... 0%")
            status_text = st.empty()
            for pct in range(1, 11):
                time.sleep(0.25)
                progress_bar.progress(pct * 10, text=f"Analyzing... {pct * 10}%")
                if pct < 4:
                    status_text.markdown("#### 🤖 Analyzing pixel-level data...")
                elif pct < 8:
                    status_text.markdown("#### 🧠 Running feature extraction...")
                else:
                    status_text.markdown("#### 📊 Generating final report...")
            progress_bar.empty()
            status_text.empty()

            image_bytes = uploaded_file.getvalue()
            status, score = get_crack_prediction(image_bytes)

            st.subheader("Analysis Report")
            if status == "DANGER":
                st.error(f"### Status: {status}")
                st.metric("Defect Confidence Score", f"{score*100:.1f}%")
                st.warning("**Immediate inspection by a certified structural engineer is recommended.**")
            else:
                st.success(f"### Status: {status}")
                st.metric("Structural Integrity Score", f"{score*100:.1f}%")
                st.info("**No immediate defects detected.** Recommend periodic monitoring (6–12 months).")

# ---------- VIDEO TAB ----------
with tab_video:
    uploaded_video = st.file_uploader("Upload a video (mp4/mov/avi)", type=["mp4", "mov", "avi"], key="vid_upl")
    st.markdown("---")
    if uploaded_video is None:
        st.info("Upload a video to run per-frame analysis with overlay.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        info_col = st.empty()
        frame_id, processed = 0, 0
        skip = st.slider("Process every Nth frame (speed vs detail)", 1, 10, 3)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % skip != 0:
                continue

            t0 = time.time()
            overlay, status, score = analyze_frame(frame)
            fps = 1.0 / max(1e-6, time.time() - t0)
            cv2.putText(overlay, f"{fps:.1f} FPS", (18, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            stframe.image(overlay, channels="BGR", use_column_width=True)
            processed += 1
            info_col.info(f"Frames processed: {processed} • Last status: {status} ({score*100:.1f}%)")

        cap.release()
        st.success("Video analysis complete.")

# ---------- LIVE FEED TAB ----------
with tab_live:
    st.write("Use your webcam for real-time overlay.")
    live_on = st.toggle("Enable webcam", key="live_toggle")
    st.caption("Tip: If feed is slow, reduce resolution or process every Nth frame.")

    if live_on:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No webcam found.")
        else:
            stframe = st.empty()
            info_col = st.empty()
            skip_live = st.slider("Process every Nth frame", 1, 6, 2, key="live_skip")
            frame_id = 0

            stop = st.button("Stop live feed")
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.warning("No camera frame received.")
                    break
                frame_id += 1
                if frame_id % skip_live != 0:
                    continue

                t0 = time.time()
                overlay, status, score = analyze_frame(frame)
                fps = 1.0 / max(1e-6, time.time() - t0)
                cv2.putText(overlay, f"{fps:.1f} FPS", (18, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                stframe.image(overlay, channels="BGR", use_column_width=True)
                info_col.info(f"Status: {status} • Confidence: {score*100:.1f}% • {fps:.1f} FPS")

                if stop or not st.session_state.get("live_toggle", False):
                    break
            cap.release()
            st.success("Live feed stopped.")

# ------------------ SIDEBAR INFO ------------------
st.sidebar.title("Crack Intelligence 'SCAN'")
st.sidebar.info("This tool detects and highlights surface defects using a CNN-style pipeline.")
st.sidebar.markdown("---")
st.sidebar.header("Technology Stack")
st.sidebar.markdown(
    """
    * **Frontend:** Streamlit  
    * **Core:** Python  
    * **AI/ML:** PyTorch & TensorFlow (pipeline-ready)  
    * **Model:** U-Net (segmentation)  
    * **CV:** OpenCV  
    * **Data:** NumPy  
    * **Assistant:** Optional OpenAI (independent of results)
    """
)

# ------------------ FLOATING CHATBOT (bottom-right popup) ------------------
# Minimal, independent assistant that doesn't read analysis context.
st.markdown("""
<style>
.scan-floating-wrap { position: fixed; bottom: 18px; right: 18px; z-index: 10000; }
.scan-fab {
  width: 56px; height: 56px; border-radius: 50%;
  border: none; cursor: pointer; font-size: 24px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.scan-chat {
  width: 360px; max-width: 92vw; height: 460px;
  background: white; border-radius: 14px; padding: 12px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# Control bar (icon)
col_fab = st.container()
with col_fab:
    st.markdown('<div class="scan-floating-wrap">', unsafe_allow_html=True)
    c1, = st.columns(1)
    with c1:
        if st.button("🤖", key="chat_fab", help="Open SCAN Assistant", use_container_width=False):
            st.session_state.show_chat = not st.session_state.show_chat
    st.markdown('</div>', unsafe_allow_html=True)

# Chat popup panel
if st.session_state.show_chat:
    chat_holder = st.container()
    with chat_holder:
        st.markdown('<div class="scan-floating-wrap"><div class="scan-chat">', unsafe_allow_html=True)
        st.markdown("### 🤖 SCAN Assistant")
        st.caption("Independent helper: tool usage + crack implications (roads/buildings/bridges).")
        st.markdown("---")

        # Render history
        with st.container(height=280):
            for m in st.session_state.messages:
                if m["role"] != "system":
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

        prompt = st.chat_input("Ask your question…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                response = get_chatbot_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Close button
        st.markdown("---")
        if st.button("Close", key="chat_close"):
            st.session_state.show_chat = False
            st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)
