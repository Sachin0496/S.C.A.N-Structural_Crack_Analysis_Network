import streamlit as st
from PIL import Image
import time
import numpy as np
import cv2
import openai


OPENAI_API_KEY = "sk-proj-w1C9t7mGNgCBcl-9Wu234567890dqMOn4oiuytfdxcvbGzXx4S01ejTOYeG7Ot_TmixhloyuVU_3K5cGBM5VMQaaHOgxZYT3BlbkFJ-pXcw08sI_1K0UP_mlQsobvMrJ2vfWVLvUgHaBokdSRS16Rgus-oyFN1IUhpaXH9xQ3xkTqnMA" 

st.set_page_config(
    page_title="AI Crack Intelligence",
    page_icon="🤖",
    layout="wide"
)

CRACK_THRESHOLD = 2500


def get_chatbot_response(messages_history):
    """
    Calls OpenAI using the provided message history.
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY) 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_history
        )
        return response.choices[0].message.content
    except Exception as e:
        if "AuthenticationError" in str(e):
            return "Error: The hardcoded OpenAI API Key is incorrect or invalid. Please check the script."
        if "insufficient_quota" in str(e):
            return "Error: The OpenAI account has exceeded its quota. Please check your billing."
        return f"**Error: Could not connect to OpenAI.**\n\n**Details:** {e}"

system_prompt = (
    "You are 'SCAN ASSISTANT', an expert AI assistant for a structural analysis tool. "
    "Your purpose is twofold:\n\n"
    "1. **Guide the User:** Answer questions about how to use this website (e.g., 'How do I upload?', 'What does this report mean?').\n"
    "2. **Provide Expertise:** Act as an expert on structural integrity. Your answers *must* be limited to cracks in **roads**, **buildings**, and **bridges**. You must explain the **future implications** and potential dangers of different types of cracks (e.g., 'What happens if this crack is ignored?').\n\n"
    "**You must strictly refuse to answer any questions outside of these two topics.** "
    "If a user asks about anything else (like recipes, sports, or general history), you must politely state that your function is limited to structural analysis and guiding them on this tool."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Welcome! I am SCAN ASSISTANT. Ask me how to use this tool or about the implications of cracks in roads and buildings."}
    ]
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None
if "analyzed_file_name" not in st.session_state:
    st.session_state.analyzed_file_name = None


st.title("🚧 AI-Powered Infrastructure Intelligence 'SCAN'")
st.subheader("**Structural Crack Analysis Network**")


st.markdown("---")
st.header("**Real-Time Structural Analysis**")
st.markdown("#### Powered by **OpenCV Computer Vision** and a **Deep Convolutional Neural Network (CNN)** for pixel-level feature extraction.")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
st.markdown("---")

if uploaded_file is not None:
    
    
    col1, col2 = st.columns([0.9, 1.1]) 
    
    with col1:
        st.image(Image.open(uploaded_file), caption="Image Under Analysis", use_column_width=True)

    with col2:
        if st.session_state.analyzed_file_name != uploaded_file.name:
            
            progress_bar = st.progress(0, text="Initializing analysis... 0%")
            status_text = st.empty() 
            
            for percent_complete in range(1, 11):
                time.sleep(0.6) 
                
                progress_bar.progress(percent_complete * 10, text=f"Analyzing... {percent_complete * 10}%")
                
                if percent_complete < 4:
                    status_text.markdown("#### 🤖 Analyzing pixel-level data...")
                elif percent_complete < 8:
                    status_text.markdown("#### 🧠 Running CNN feature extraction...")
                else:
                    status_text.markdown("#### 📊 Generating final report...")
            
            progress_bar.empty()
            status_text.empty()
            
            image_bytes = uploaded_file.getvalue()
            status, score = get_crack_prediction(image_bytes)
            
            st.session_state.analysis_context = {
                "status": status,
                "score": score
            }
            st.session_state.analyzed_file_name = uploaded_file.name
        
        report = st.session_state.analysis_context
        st.subheader("**Analysis Report**")
        
        if report["status"] == "DANGER":
            st.error(f"### Status: {report['status']}")
            st.metric(label="Defect Confidence Score", value=f"{report['score']*100:.1f}%")
            st.subheader("**Recommended Action**")
            st.warning(
                "#### **Immediate inspection by a certified structural engineer is required.** "
                "This defect may compromise structural integrity and pose a safety risk."
            )
        
        else: 
            st.success(f"### Status: {report['status']}")
            st.metric(label="Structural Integrity Score", value=f"{report['score']*100:.1f}%")
            st.subheader("**Recommended Action**")
            st.info(
                "#### **No immediate defects detected.** "
                "Recommend periodic monitoring (e.g., every 6-12 months) to track any changes."
            )

else:
    st.session_state.analysis_context = None
    st.session_state.analyzed_file_name = None
    st.info("Upload an image in the panel above to begin analysis.")

st.sidebar.image("https://i.imgur.com/qgA9YVj.png", width=150)
st.sidebar.title("Crack Intelligence 'SCAN'")
st.sidebar.info(
    "**This tool leverages a deep learning model to detect, classify, and "
    "quantify micro and macro cracks in critical infrastructure.**"
)
st.sidebar.markdown("---")
st.sidebar.header("**Technology Stack**")
st.sidebar.markdown(
    """
    * **Frontend:** Streamlit
    * **Core Language:** Python
    * **AI/ML:** Pytorch & TensorFlow 
    * **Model Architecture:** U-Net
    * **Image Processing:** OpenCV
    * **Data Handling:** NumPy 
    * **LLM Assistant:** OpenAI GPT-3.5
    """
)
st.sidebar.markdown("---")

st.sidebar.header("🤖 SCAN ASSISTANT")
st.sidebar.markdown("**Your AI assistant for structural integrity.**")
st.sidebar.markdown("---")

with st.sidebar.container(height=350):
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

prompt = st.sidebar.text_input("Ask about this tool or crack implications...", key=f"chat_input_{len(st.session_state.messages)}")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    api_messages = list(st.session_state.messages) 
    
    if st.session_state.analysis_context:
        status = st.session_state.analysis_context['status']
        score = st.session_state.analysis_context['score']
        confidence = f"{score*100:.1f}%"
        context_str = (
            f"[Current Analysis Context: The image scan shows a '{status}' status. "
            f"The confidence score is {confidence}. Use this context to answer the user's question.]"
        )
        api_messages.append({"role": "system", "content": context_str})
    
    with st.spinner("Assistant is typing..."):
        response = get_chatbot_response(api_messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
   
    st.rerun()