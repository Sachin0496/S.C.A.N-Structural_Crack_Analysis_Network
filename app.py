import streamlit as st
from PIL import Image
import numpy as np
import openai
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import io


# --- IMPORTS for the REAL MODEL ---
try:
    from model.scan_model import SCAN_Model
except ImportError:
    st.error("FATAL ERROR: Could not find 'model/scan_model.py'.")
    st.stop()
except SyntaxError:
    st.error("FATAL ERROR: There is a SyntaxError in 'model/scan_model.py'.")
    st.stop()


# --- SECRETS ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- MODEL CONFIG ---
MODEL_PATH = 'SCAN_CT260_FT1.pth'
INPUT_IMG_SIZE = (512, 512)

st.set_page_config(
    page_title="AI S-C.A.N Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- PYTORCH HELPER FUNCTIONS ---

@st.cache_resource
def load_model(model_path):
    st.info("Loading S.C.A.N AI model... this may take a moment.")
    model = SCAN_Model()
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file not found at '{model_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    st.success("S.C.A.N AI model loaded successfully!")
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(INPUT_IMG_SIZE, Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = tensor.unsqueeze(0)
    return tensor


@st.cache_data
def get_real_crack_prediction(image_bytes, _model):
    """
    NEW LOGIC: This function calculates the "Average Crack Severity" (0-100)
    based on the user's mental model.
    """
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output, *rest = _model(input_tensor)
        
    probabilities = torch.sigmoid(output)
    prediction_mask = probabilities.cpu().squeeze().numpy()
    
    # --- NEW "AVERAGE SEVERITY" LOGIC ---
    
    # 1. Define what we consider a crack pixel (user's "40" implies 0.4, but 0.5 is safer)
    CONFIDENCE_THRESHOLD = 0.5 
    
    # 2. Find all pixels that are *above* this threshold
    crack_pixels = prediction_mask[prediction_mask > CONFIDENCE_THRESHOLD]
    
    # 3. Calculate the score
    if crack_pixels.size == 0:
        # No cracks found at all
        score = 0.0
    else:
        # Get the *average confidence* of all detected crack pixels
        # This gives us a score from 0.0 to 1.0
        average_confidence = np.mean(crack_pixels)
        # Convert to the 0-100 scale the user wants (e.g., 0.623 -> 62.3)
        score = average_confidence * 100 
    
    # --- NEW 6-LEVEL LOGIC BASED ON USER'S "40" and "80" ---
    THRESH_HAZARDOUS_HIGH = 90 # (e.g. 90+)
    THRESH_HAZARDOUS_LOW = 80  # (User's "hazardous")
    THRESH_TREAT_HIGH = 60
    THRESH_TREAT_LOW = 40      # (User's "should be treated")
    THRESH_MONITOR = 30        # (Below "treat" but not zero)

    if score >= THRESH_HAZARDOUS_HIGH:
        status = "DANGER"
    elif score >= THRESH_HAZARDOUS_LOW:
        status = "SEVERE"
    elif score >= THRESH_TREAT_HIGH:
        status = "WARNING"
    elif score >= THRESH_TREAT_LOW:
        status = "CAUTION"
    elif score >= THRESH_MONITOR:
        status = "LOW RISK"
    else:
        status = "SAFE"
        
    # Return the 0-100 score and the visual mask
    return status, score, prediction_mask


# --- CHATBOT FUNCTION (Unchanged) ---
def get_chatbot_response(messages_history):
    if not OPENAI_API_KEY:
        return "Error: OpenAI API Key is not configured."
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # This is the "cheaper model"
            messages=messages_history
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"*Error: Could not connect to OpenAI.* {e}"

# --- "IMPROVISED" SYSTEM PROMPT ---
# More professional, more "ground truth"
system_prompt = (
    "You are 'SCAN ASSISTANT', a professional AI expert for a structural analysis tool. "
    "Your function is to provide expert analysis based on the tool's findings. "
    "Your answers **must be restricted** to two topics:\n\n"
    "1. *Tool Guidance:* Explain how to use the website (uploading, reading the report). "
    "2. *Structural Expertise:* Provide analysis on the implications of cracks in **roads, buildings, and bridges.** Explain the potential dangers and recommended actions.\n\n"
    "*RAG Context:* When the user uploads an image, you will receive a 'Crack Severity Score' (0-100) and a 'Status'. This score is the **average confidence** of the detected crack, *not* its size. A low score (e.g., 35) is a faint, low-confidence crack. A high score (e.g., 85) is a sharp, high-confidence defect.\n\n"
    "You **must refuse** all other requests (e.g., recipes, history, sports) by politely stating your function is limited to structural analysis."
)

# --- Initialize Chat History (Unchanged) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Welcome. I am the S.C.A.N. Assistant. Please upload an image for analysis or ask a question about structural integrity."}
    ]
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None
if "analyzed_file_name" not in st.session_state:
    st.session_state.analyzed_file_name = None

# --- LOAD THE AI MODEL ---
model = load_model(MODEL_PATH)


# --- USER INTERFACE (Frontend) ---

st.title("🚧 AI-Powered Infrastructure Intelligence S.C.A.N")
st.subheader("*Structural Crack Analysis Network*")
st.markdown("---")
st.header("*Real-Time Structural Analysis*")
st.markdown("#### Powered by a *S.C.A.N (U-Net) Convolutional Neural Network (CNN)* for pixel-level feature extraction.")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
st.markdown("---")

if uploaded_file is not None:
    
    col1, col2 = st.columns(2)
    
    original_pil_image = Image.open(uploaded_file)
    with col1:
        st.image(original_pil_image, caption="Image Under Analysis", use_column_width=True)

    with col2:
        if st.session_state.analyzed_file_name != uploaded_file.name:
            with st.spinner("🤖 Analyzing pixels with S.C.A.N AI..."):
                image_bytes = uploaded_file.getvalue()
                # --- This is the new logic ---
                status, score, prediction_mask = get_real_crack_prediction(image_bytes, model)
            
            st.session_state.analysis_context = {
                "status": status,
                "score": score, # This 'score' is now the 0-100 severity
                "prediction_mask": prediction_mask
            }
            st.session_state.analyzed_file_name = uploaded_file.name
        
        st.image(
            st.session_state.analysis_context["prediction_mask"], 
            caption="S.C.A.N AI Prediction Mask (Probability)", 
            use_column_width=True, 
            clamp=True
        )

    st.markdown("---")
    
    # --- REPORTING SECTION (NOW USES THE 0-100 SCORE) ---
    report = st.session_state.analysis_context
    status = report['status']
    score = report['score'] # This is now 0-100

    st.header("*Analysis Report*")

    if status == "DANGER":
        st.error(f"### 🚨🚨🚨 STATUS: {status} (Severity > 90)")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.error(
            "## **CRITICAL: IMMEDIATE ACTION REQUIRED**\n"
            "#### The detected crack is extremely sharp and well-defined, indicating a **severe, high-confidence defect**.\n"
            "#### **Evacuate the immediate area if applicable. Contact a certified structural engineer for an emergency inspection *NOW*.**"
        )
    
    elif status == "SEVERE":
        st.error(f"### 🟥 STATUS: {status} (Severity > 80)")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.error(
            "### **High Priority: Urgent Inspection Required**\n"
            "#### This is a **hazardous defect** (Severity > 80). The crack is clear and poses a significant risk.\n"
            "#### **Schedule an inspection with a qualified professional *today*. Do not delay.**"
        )

    elif status == "WARNING":
        st.warning(f"### ⚠️ STATUS: {status} (Severity > 60)")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.warning(
            "### **Priority: Inspection Recommended**\n"
            "#### The crack is well-defined. While not yet hazardous, it exceeds the treatment threshold and should be addressed.\n"
            "#### **Schedule an inspection in the near future.**"
        )

    elif status == "CAUTION":
        st.info(f"### 🟡 STATUS: {status} (Severity > 40)")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.info(
            "### **Monitor Closely: Treatment Advised**\n"
            "#### A clear crack has been detected (Severity > 40). This is at the level where **treatment is advised**.\n"
            "#### **Log this location and schedule for maintenance or re-scan in 3-6 months.**"
        )

    elif status == "LOW RISK":
        st.info(f"### 📈 STATUS: {status} (Severity > 30)")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.info(
            "### **Low Risk: Periodic Monitoring Advised**\n"
            "#### The analysis detected a **faint, low-confidence anomaly** (e.g., potential hairline crack or image noise).\n"
            "#### **Re-scan in 6-12 months as part of a standard maintenance schedule.**"
        )
    
    else: # This is the "SAFE" (Score < 30)
        st.success(f"### ✅ STATUS: {status}")
        st.metric(label="Crack Severity Score", value=f"{score:.1f} / 100")
        st.subheader("*Recommended Action*")
        st.success(
            "### **No Significant Defects Detected**\n"
            "#### No clear defects were found, or anomalies are below the monitoring threshold (< 30 Severity).\n"
            "#### **Continue with standard maintenance schedules.**"
        )
        st.balloons()

else:
    st.session_state.analysis_context = None
    st.session_state.analyzed_file_name = None
    st.info("Upload an image in the panel above to begin analysis.")

# --- Sidebar Information (Unchanged) ---
st.sidebar.title(" SCAN-Structural Crack Analysis Network")
st.sidebar.info(
    "**This Tool Leverages A Deep Learning Model To Detect, Classify, And "
    "Quantify Micro And Macro Cracks In Critical Infrastructure."
)
st.sidebar.markdown("---")
st.sidebar.header("*Technology Stack*")
st.sidebar.markdown(
    """
    * *Frontend:* Streamlit
    * *Core Language:* Python
    * *AI/ML:* **PyTorch**
    * *Model Architecture:* **U-Net (S.C.A.N)**
    * *Image Processing:* PIL, NumPy
    * *LLM Assistant:* OpenAI GPT-3.5
    """
)
st.sidebar.markdown("---")

# --- CHATBOT INSIDE SIDEBAR (NOW WITH NEW RAG) ---
st.sidebar.header("🤖 S.C.A.N HELPER")
st.sidebar.markdown("*Your AI assistant for Structural Integrity.*")
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
    
    # --- THIS IS THE "IMPROVISED" RAG CONTEXT ---
    if st.session_state.analysis_context:
        status = st.session_state.analysis_context['status']
        score = st.session_state.analysis_context['score']
        
        # This new context is much clearer for the "cheap" LLM
        context_str = (
            f"[Current Analysis Context: The image scan shows a '{status}' status. "
            f"The 'Crack Severity Score' is {score:.1f} (out of 100). "
            f"This score represents the *average confidence* of the detected crack, not its size. "
            f"A score over 40 requires treatment, and over 80 is hazardous. "
            f"Use this to answer the user's question.]"
        )
        api_messages.append({"role": "system", "content": context_str})
    
    with st.spinner("Assistant is typing..."):
        response = get_chatbot_response(api_messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()