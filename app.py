import streamlit as st
from PIL import Image
import numpy as np
import openai
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import io # Added io for the image bytes

# --- NEW IMPORTS for the REAL MODEL ---
# This now imports SCAN_Model from your renamed model/scan_model.py file
try:
    from model.scan_model import SCAN_Model
except ImportError:
    st.error("FATAL ERROR: Could not find 'model/scan_model.py'. "
             "Please make sure your file structure is correct (see instructions).")
    st.stop()


# --- SECRETS ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- MODEL CONFIG ---
MODEL_PATH = 'DeepCrack_CT260_FT1.pth' # This still points to your original weights file
INPUT_IMG_SIZE = (512, 512)
CRACK_THRESHOLD = 0.7 

st.set_page_config(
    page_title="AI S-C.A.N Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- NEW PYTORCH HELPER FUNCTIONS ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the SCAN_Model and handles the 'module.' prefix
    from DataParallel training.
    """
    st.info("Loading S.C.A.N AI model... this may take a moment.")
    
    # Instantiate the "body"
    model = SCAN_Model()
    
    # Load the "brain" (.pth file) onto the CPU
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file not found at '{model_path}'.")
        st.stop()

    # Handle 'module.' prefix if it exists
    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Set to evaluation mode (CRITICAL)
    model.eval()
    st.success("S.C.A.N AI model loaded successfully!")
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """ Prepares a PIL Image for the SCAN_Model model. """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(INPUT_IMG_SIZE, Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = tensor.unsqueeze(0) # Add batch dimension
    return tensor

@st.cache_data # Cache the prediction itself
def get_real_crack_prediction(_image_bytes, model):
    """
    This is the REAL prediction function that uses the SCAN_Model.
    It returns the status, the score, and the prediction mask image.
    """
    image = Image.open(io.BytesIO(_image_bytes))
    
    # 1. Preprocess the image for the model
    input_tensor = preprocess_image(image)
    
    # 2. Run inference
    with torch.no_grad():
        # Your model returns 6 outputs, we only need the first (final) one
        output, *rest = model(input_tensor)
        
    # 3. Post-process the output
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(output)
    
    # Remove batch dim, move to CPU, convert to NumPy
    prediction_mask = probabilities.cpu().squeeze().numpy()
    
    # 4. Generate the 'status' and 'score' your RAG bot needs
    # The "score" will be the highest probability pixel in the mask
    score = np.max(prediction_mask)
    
    if score > CRACK_THRESHOLD:
        status = "DANGER"
    else:
        status = "SAFE"
        
    return status, score, prediction_mask


# --- YOUR CHATBOT FUNCTION (Unchanged) ---
def get_chatbot_response(messages_history):
    """
    Calls OpenAI using the provided message history.
    """
    if not OPENAI_API_KEY:
        return ("Error: OpenAI API Key is not configured. "
                "Please add it to your `.streamlit/secrets.toml` file.")
        
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_history
        )
        return response.choices[0].message.content
    except Exception as e:
        if "AuthenticationError" in str(e):
            return "Error: The OpenAI API Key in your `secrets.toml` file is incorrect or invalid."
        if "insufficient_quota" in str(e):
            return "Error: The OpenAI account has exceeded its quota. Please check your billing."
        return f"*Error: Could not connect to OpenAI.\n\nDetails:* {e}"

# --- SYSTEM PROMPT (Unchanged) ---
system_prompt = (
    "You are 'SCAN ASSISTANT', an expert AI assistant for a structural analysis tool. "
    "Your purpose is twofold:\n\n"
    "1. *Guide the User:* Answer questions about how to use this website (e.g., 'How do I upload?', 'What does this report mean?').\n"
    "2. *Provide Expertise:* Act as an expert on structural integrity. Your answers must be limited to cracks in *roads, **buildings, and **bridges. You must explain the **future implications* and potential dangers of different types of cracks (e.g., 'What happens if this crack is ignored?').\n\n"
    "*You must strictly refuse to answer any questions outside of these two topics.* "
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
# This is done once and cached
model = load_model(MODEL_PATH)


# --- USER INTERFACE (Frontend) ---

st.title("🚧 AI-Powered Infrastructure Intelligence S.C.A.N")
st.subheader("*Structural Crack Analysis Network*")
st.markdown("---")
st.header("*Real-Time Structural Analysis*")
# --- UPDATED TEXT ---
st.markdown("#### Powered by a *S.C.A.N (U-Net) Convolutional Neural Network (CNN)* for pixel-level feature extraction.")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
st.markdown("---")

if uploaded_file is not None:
    
    # We now show the original image AND the prediction mask
    col1, col2 = st.columns(2)
    
    original_pil_image = Image.open(uploaded_file)
    with col1:
        st.image(original_pil_image, caption="Image Under Analysis", use_column_width=True)

    with col2:
        if st.session_state.analyzed_file_name != uploaded_file.name:
            # --- THIS IS THE NEW, REAL ANALYSIS ---
            # --- UPDATED TEXT ---
            with st.spinner("🤖 Analyzing pixels with S.C.A.N AI..."):
                image_bytes = uploaded_file.getvalue()
                
                # Call the REAL prediction function
                status, score, prediction_mask = get_real_crack_prediction(image_bytes, model)
            
            # Save results to session state for the RAG bot
            st.session_state.analysis_context = {
                "status": status,
                "score": score,
                "prediction_mask": prediction_mask # Store the mask too
            }
            st.session_state.analyzed_file_name = uploaded_file.name
        
        # --- DISPLAY THE PREDICTION MASK ---
        # --- UPDATED TEXT ---
        st.image(
            st.session_state.analysis_context["prediction_mask"], 
            caption="S.C.A.N AI Prediction Mask (Probability)", 
            use_column_width=True, 
            clamp=True
        )

    st.markdown("---")
    
    # --- DISPLAY THE REPORT (Your logic, unchanged) ---
    report = st.session_state.analysis_context
    st.header("*Analysis Report*")
    
    if report["status"] == "DANGER":
        st.error(f"### Status: {report['status']}")
        # We now use the *real* score
        st.metric(label="Defect Confidence Score", value=f"{report['score']*100:.1f}%")
        st.subheader("*Recommended Action*")
        st.warning(
            "#### *Immediate inspection by a certified structural engineer is required.* "
            "This defect may compromise structural integrity and pose a safety risk."
        )
    
    else:  
        st.success(f"### Status: {report['status']}")
        # The score is now the max probability of a crack.
        # A low "max" probability is good.
        st.metric(label="Confidence Score (No Defect)", value=f"{(1.0 - report['score'])*100:.1f}%")
        st.subheader("*Recommended Action*")
        st.info(
            "#### *No immediate defects detected.* "
            "Recommend periodic monitoring (e.g., every 6-12 months) to track any changes."
        )

else:
    st.session_state.analysis_context = None
    st.session_state.analyzed_file_name = None
    st.info("Upload an image in the panel above to begin analysis.")

# --- Sidebar Information (Your code, unchanged) ---
st.sidebar.title(" SCAN-Structural Crack Analysis Network")
st.sidebar.info(
    "**This Tool Leverages A Deep Learning Model To Detect, Classify, And "
    "Quantify Micro And Macro Cracks In Critical Infrastructure."
)
st.sidebar.markdown("---")
st.sidebar.header("*Technology Stack*")
# --- UPDATED TEXT ---
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

# --- CHATBOT INSIDE SIDEBAR (Your code, unchanged) ---
st.sidebar.header("🤖 S-C.A.N HELPER")
st.sidebar.markdown("*Your AI assistant for Structural Integrity.*")
st.sidebar.markdown("---")

# 1. Display all past messages in a scrolling container
with st.sidebar.container(height=350):
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 2. Get new user input
prompt = st.sidebar.text_input("Ask about this tool or crack implications...", key=f"chat_input_{len(st.session_state.messages)}")

if prompt:
    # User has pressed Enter, so we process the prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    api_messages = list(st.session_state.messages)  
    
    # This is your RAG logic - it still works perfectly!
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