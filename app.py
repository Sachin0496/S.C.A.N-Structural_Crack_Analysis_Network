import streamlit as st
from PIL import Image
import numpy as np
import openai
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import io # Added io for the image bytes

# --- IMPORTS for the REAL MODEL ---
try:
    from model.scan_model import SCAN_Model
except ImportError:
    st.error("FATAL ERROR: Could not find 'model/scan_model.py'. "
             "Please make sure your file structure is correct (see instructions).")
    st.stop()
except SyntaxError:
    st.error("FATAL ERROR: There is a SyntaxError in 'model/scan_model.py'. "
             "Please check that file for typos.")
    st.stop()


# --- SECRETS ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- MODEL CONFIG ---
MODEL_PATH = 'SCAN_CT260_FT1.pth' # This still points to your original weights file
INPUT_IMG_SIZE = (512, 512)
# CRACK_THRESHOLD = 0.7  <-- This is no longer used by the new logic

st.set_page_config(
    page_title="AI S-C.A.N Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- CSS HACK TO CHANGE WIDGET COLORS ---
# This CSS makes the file uploader and chat input white
# It overrides the default theme for these specific elements
css_code = """
<style>
    /* --- File Uploader --- */

    /* This targets the "drag and drop" box */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important; /* White background */
        border: 2px dashed #DDDDDD !important; /* Light gray border */
    }

    /* This targets the text inside the box */
    [data-testid="stFileUploader"] section p {
        color: #555555 !important; /* Medium-dark text for readability */
    }

    /* --- Chat Input --- */

    /* This targets the chat input bar at the bottom */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border-top: 1px solid #DDDDDD !important; /* Adds a line to separate it */
    }

    /* This targets the text area inside the chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #212529 !important; /* Dark text for the input */
    }
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)
# --- END OF CSS HACK ---


# --- PYTORCH HELPER FUNCTIONS ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the SCAN_Model and handles the 'module.' prefix
    from DataParallel training.
    """
    model = SCAN_Model()
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file not found at '{model_path}'.")
        st.error("Please make sure the file 'DeepCrack_CT260_FT1.pth' is in the same folder as 'app.py'")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """ Prepares a PIL Image for the SCAN_Model model. """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(INPUT_IMG_SIZE, Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = tensor.unsqueeze(0) # Add batch dimension
    return tensor

# --- [CHANGE 1] - UPDATED SEVERITY PREDICTION LOGIC ---
@st.cache_data # Cache the prediction itself
def get_real_crack_prediction(image_bytes, _model):
    """
    This function now uses the new "Average Crack Severity" logic.
    """
    # --- 1. Run the Model (Unchanged) ---
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output, *rest = _model(input_tensor)
        
    probabilities = torch.sigmoid(output)
    prediction_mask = probabilities.cpu().squeeze().numpy()
    
    # --- 2. NEW "AVERAGE SEVERITY" LOGIC ---
    
    # Define what we consider a crack pixel
    CONFIDENCE_THRESHOLD = 0.5 
    
    # Find all pixels that are above this threshold
    crack_pixels = prediction_mask[prediction_mask > CONFIDENCE_THRESHOLD]
    
    # Calculate the score
    if crack_pixels.size == 0:
        # No cracks found at all
        score = 0.0
    else:
        # Get the average confidence of all detected crack pixels
        # This gives us a score from 0.0 to 1.0
        average_confidence = np.mean(crack_pixels)
        # Convert to the 0-100 scale (e.g., 0.623 -> 62.3)
        score = average_confidence * 100 
    
    # --- 3. NEW 6-LEVEL STATUS LOGIC ---
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
# --- END OF CHANGE 1 ---


# --- CHATBOT FUNCTION (Unchanged) ---
def get_chatbot_response(messages_history):
    """
    Calls OpenAI using the provided message history.
    """
    if not OPENAI_API_KEY:
        return ("Error: OpenAI API Key is not configured. "
                "Please add it to your .streamlit/secrets.toml file.")
        
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_history
        )
        return response.choices[0].message.content
    except Exception as e:
        if "AuthenticationError" in str(e):
            return "Error: The OpenAI API Key in your secrets.toml file is incorrect or invalid."
        if "insufficient_quota" in str(e):
            return "Error: The OpenAI account has exceeded its quota. Please check your billing."
        return f"Error: Could not connect to OpenAI.\n\nDetails: {e}"

# --- SYSTEM PROMPT (Unchanged) ---
system_prompt = (
    "You are 'SCAN ASSISTANT', an expert AI assistant for a structural analysis tool. "
    "Your purpose is twofold:\n\n"
    "1. Guide the User: Answer questions about how to use this website (e.g., 'How do I upload?', 'What does this report mean?').\n"
    "2. Provide Expertise: Act as an expert on structural integrity. Your answers must be limited to cracks in roads, **buildings, and **bridges. You must explain the **future implications and potential dangers of different types of cracks (e.g., 'What happens if this crack is ignored?').\n\n"
    "You must strictly refuse to answer any questions outside of these two topics. "
    "If a user asks about anything else (like recipes, sports, or general history), you must politely state that your function is limited to structural analysis and guiding them on this tool."
)

# --- Initialize Chat History (Unchanged) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Welcome! I am SCAN ASSISTANT. Ask me how to use this tool or about the implications of cracks in roads and buildings."}
    ]
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None
if "analyzed_file_name" not in st.session_state:
    st.session_state.analyzed_file_name = None

# --- LOAD THE AI MODEL ---
model = load_model(MODEL_PATH)


# --- USER INTERFACE (Frontend) ---

# Header (Unchanged)
col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
with col1:
    st.title("🚧 AI-Powered Infrastructure Intelligence S.C.A.N")
    st.subheader("Structural Crack Analysis Network")
with col2:
    st.markdown("<h1 style='text-align: right; font-size: 4.5rem; padding-bottom: 0.5rem;'></h1>", unsafe_allow_html=True)


st.markdown("---") 

# Colored Header (Unchanged)
st.markdown("<h2 style='color: #00A9FF;'><i>Real-Time Structural Analysis</i></h2>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #AAAAAA;'>Powered by a <i>S.C.A.N (U-Net) Convolutional Neural Network (CNN)</i> for pixel-level feature extraction.</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
st.markdown("---")

if uploaded_file is not None:
    
    # Analysis columns (Unchanged)
    col1, col2 = st.columns(2)
    original_pil_image = Image.open(uploaded_file)
    with col1:
        st.image(original_pil_image, caption="Image Under Analysis", use_container_width=True)

    with col2:
        if st.session_state.analyzed_file_name != uploaded_file.name:
            with st.spinner("🤖 Analyzing pixels with S.C.A.N AI..."):
                image_bytes = uploaded_file.getvalue()
                status, score, prediction_mask = get_real_crack_prediction(image_bytes, model)
            
            st.session_state.analysis_context = {
                "status": status,
                "score": score,
                "prediction_mask": prediction_mask
            }
            st.session_state.analyzed_file_name = uploaded_file.name
            st.session_state.auto_message_sent = False
        
        st.image(
            st.session_state.analysis_context["prediction_mask"], 
            caption="S.C.A.N AI Prediction Mask (Probability)", 
            use_container_width=True, 
            clamp=True
        )

    st.markdown("---")
    
    # --- [CHANGE 2] - UPDATED 6-LEVEL REPORT DISPLAY ---
    report = st.session_state.analysis_context
    st.header("Analysis Report")

    if report["status"] == "DANGER":
        st.error(f"### 🚨🚨🚨 STATUS: {report['status']} (Severity > 90)")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.error(
            "## CRITICAL: IMMEDIATE ACTION REQUIRED\n"
            "#### The detected crack is extremely sharp and well-defined, indicating a severe, high-confidence defect.\n"
            "#### Evacuate the immediate area if applicable. Contact a certified structural engineer for an emergency inspection *NOW.*"
        )
    
    elif report["status"] == "SEVERE":
        st.error(f"### 🟥 STATUS: {report['status']} (Severity > 80)")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.error(
            "### High Priority: Urgent Inspection Required\n"
            "#### This is a hazardous defect (Severity > 80). The crack is clear and poses a significant risk.\n"
            "#### Schedule an inspection with a qualified professional *today. Do not delay.*"
        )

    elif report["status"] == "WARNING":
        st.warning(f"### ⚠ STATUS: {report['status']} (Severity > 60)")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.warning(
            "### Priority: Inspection Recommended\n"
            "#### The crack is well-defined. While not yet hazardous, it exceeds the treatment threshold and should be addressed.\n"
            "#### Schedule an inspection in the near future."
        )

    elif report["status"] == "CAUTION":
        st.info(f"### 🟡 STATUS: {report['status']} (Severity > 40)")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.info(
            "### Monitor Closely: Treatment Advised\n"
            "#### A clear crack has been detected (Severity > 40). This is at the level where treatment is advised.\n"
            "#### Log this location and schedule for maintenance or re-scan in 3-6 months."
        )

    elif report["status"] == "LOW RISK":
        st.info(f"### 📈 STATUS: {report['status']} (Severity > 30)")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.info(
            "### Low Risk: Periodic Monitoring Advised\n"
            "#### The analysis detected a faint, low-confidence anomaly (e.g., potential hairline crack or image noise).\n"
            "#### Re-scan in 6-12 months as part of a standard maintenance schedule."
        )
    
    else: # This is the "SAFE" (Score < 30)
        st.success(f"### ✅ STATUS: {report['status']}")
        st.metric(label="Crack Severity Score", value=f"{report['score']:.1f} / 100")
        st.subheader("Recommended Action")
        st.success(
            "### No Significant Defects Detected\n"
            "#### No clear defects were found, or anomalies are below the monitoring threshold (< 30 Severity).\n"
            "#### Continue with standard maintenance schedules."
        )
    # --- END OF CHANGE 2 ---


    # --- [CHANGE 3] - UPDATED AUTO-CHATBOT LOGIC ---
    if st.session_state.analysis_context and not st.session_state.get("auto_message_sent", False):
        status = st.session_state.analysis_context["status"]
        
        # 1. Define the prompt based on the 6-level status
        if status in ["DANGER", "SEVERE", "WARNING"]:
            # High-risk statuses
            auto_prompt = (
                f"The S.C.A.N. tool has just completed an analysis and found a *'{status}'* status. "
                "As the AI assistant, please automatically provide a helpful message for the user. "
                "In your message, please: "
                f"1. State that a '{status}' status was detected. "
                "2. Briefly explain what this means in a serious tone, reflecting the urgency. "
                "3. Provide a clear, bulleted list of recommended safety mechanisms and preventative actions (e.g., 'Immediate inspection by a structural engineer', 'Cordon off the area', 'Look for secondary signs of damage')."
            )
        elif status in ["CAUTION", "LOW RISK"]:
            # Low-risk / monitoring statuses
            auto_prompt = (
                f"The S.C.A.N. tool has just completed an analysis and found a *'{status}'* status. "
                "As the AI assistant, please automatically provide a helpful message for the user. "
                "In your message, please: "
                f"1. State that a '{status}' status was detected. "
                "2. Briefly explain what this means. "
                "3. Provide a clear, bulleted list of preventative monitoring tips for the future (e.g., 'Periodic re-inspection', 'Log this location', 'Document any small changes over time')."
            )
        else: # status == "SAFE"
            auto_prompt = (
                "The S.C.A.N. tool has just completed an analysis and found a *'SAFE'* status. "
                "As the AI assistant, please automatically provide a helpful message for the user. "
                "In your message, please: "
                "1. State that a 'SAFE' status was confirmed. "
                "2. Briefly explain what this means. "
                "3. Provide a clear, bulleted list of standard maintenance tips (e.g., 'Keep the area clean', 'Continue standard inspection schedules')."
            )

        # 2. Create the message list for the API call
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": auto_prompt} 
        ]
        
        # 3. Call the chatbot function
        with st.spinner("Assistant is generating preventative tips..."):
            response = get_chatbot_response(api_messages)
        
        # 4. Add the AI's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 5. Mark the message as sent
        st.session_state.auto_message_sent = True 
        
        # 6. Rerun the script to display the new message
        st.rerun()
    # --- END OF CHANGE 3 ---

else:
    # No file is uploaded
    st.session_state.analysis_context = None
    st.session_state.analyzed_file_name = None
    st.session_state.auto_message_sent = False
    st.info("Upload an image in the panel above to begin analysis.")

# --- Sidebar Information (Unchanged) ---
st.sidebar.title(" SCAN-Structural Crack Analysis Network")
st.sidebar.info(
    "**This Tool Leverages A Deep Learning Model To Detect, Classify, And "
    "Quantify Micro And Macro Cracks In Critical Infrastructure."
)
st.sidebar.markdown("---")
st.sidebar.header("About this Project")
st.sidebar.markdown(
    """
    This tool is a demonstration of using Deep Learning for
    real-world infrastructure monitoring. The *S.C.A.N (U-Net)*
    model was trained on thousands of images to identify 
    pixel-level patterns indicative of structural defects.
    """
)
st.sidebar.markdown("---")
st.sidebar.header("Technology Stack")
st.sidebar.markdown(
    """
    * *Frontend:* Streamlit
    * *Core Language:* Python
    * *AI/ML:* PyTorch
    * *Model Architecture:* U-Net (S.C.A.N)
    * *Image Processing:* PIL, NumPy
    * *LLM Assistant:* OpenAI GPT-3.5
    """
)
st.sidebar.markdown("---")


# --- CHATBOT INTERFACE (Unchanged) ---

st.markdown("---")
st.header("🤖 S-C-A-N HELPER")
st.markdown("Your AI assistant for Structural Integrity.")

# 1. Display all past messages in a container
with st.container(height=350):
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 2. Get new user input using st.chat_input (docked at bottom)
if prompt := st.chat_input("Ask about this tool or crack implications..."):
    
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # RAG logic (Unchanged)
    api_messages = list(st.session_state.messages)  
    if st.session_state.analysis_context:
        status = st.session_state.analysis_context['status']
        score = st.session_state.analysis_context['score']
        # Use the new 0-100 score
        confidence = f"{score:.1f} / 100"
        context_str = (
            f"[Current Analysis Context: The image scan shows a '{status}' status. "
            f"The Crack Severity Score is {confidence}. Use this context to answer the user's question.]"
        )
        api_messages.append({"role": "system", "content": context_str})
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = get_chatbot_response(api_messages)
            st.markdown(response)
    
    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": response})