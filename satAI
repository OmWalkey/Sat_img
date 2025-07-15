import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import numpy as np
import gc
import json
import datetime
import io
import base64
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Enhanced Satellite AI Pro", layout="wide", page_icon="🛰️")

# ---------------------- Model & Quantization Config ----------------------
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

# Improved BitsAndBytesConfig
BQB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

@st.cache_resource
def load_model():
    """Load model with improved error handling and configuration"""
    try:
        # Load processor first
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Ensure processor has the correct configuration
        if not hasattr(processor, 'image_processor'):
            st.error("Image processor not found in the model. Please check model compatibility.")
            return None, None
        
        # Load model with proper configuration
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=BQB_CONFIG,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # Add this for some models
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model and processor with error handling
@st.cache_resource
def initialize_model():
    try:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load_model()
        
        if model is None or processor is None:
            return None, None, None
        
        # Move model to device if not already done by device_map
        if torch_device.type == "cpu":
            model = model.to(torch_device)
        
        return model, processor, torch_device
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None, None, None

model, processor, torch_device = initialize_model()

# ---------------------- Image Processing Functions ----------------------
def process_image(uploaded_file):
    """Process uploaded image with error handling"""
    try:
        image = Image.open(uploaded_file)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (LLaVA has input size limits)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def enhance_satellite_image(image, contrast=1.2, brightness=1.0, saturation=1.1, sharpness=1.0, 
                          edge_enhance=False, denoise=False):
    """Enhanced preprocessing specifically for satellite imagery"""
    try:
        img = image.copy()
        
        # Basic enhancements
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
        # Advanced processing
        if edge_enhance:
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        if denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image

# ---------------------- Fixed Analysis Function ----------------------
def run_analysis(image, prompt_text, model, processor, device, max_tokens=300, temperature=0.3, top_p=0.85):
    """Run satellite image analysis with improved error handling and proper input formatting"""
    
    if model is None or processor is None:
        return "Model not properly loaded. Please check the model initialization."
    
    try:
        # Prepare the conversation format that LLaVA expects
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are an expert satellite image analyst. {prompt_text}"
                    },
                    {
                        "type": "image"
                    }
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs with proper formatting
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.1,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        # Generate response with error handling
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode response
        input_len = inputs['input_ids'].shape[1]
        gen_tokens = outputs[0][input_len:]
        result = processor.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Clean up GPU memory
        del inputs, outputs, gen_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result if result else "No response generated. Please try again with a different prompt."
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error guidance
        if "image tokens" in error_msg.lower():
            return "Error: Image token processing issue. Try resizing your image or using a different image format."
        elif "out of memory" in error_msg.lower():
            return "Error: GPU memory insufficient. Try reducing max_tokens or using CPU mode."
        elif "cuda" in error_msg.lower():
            return "Error: CUDA/GPU issue. Try switching to CPU mode in the settings."
        else:
            return f"Analysis failed: {error_msg}"

# ---------------------- Enhanced Prompts ----------------------
def get_enhanced_prompts():
    return {
        "Comprehensive Scene Description": """Provide a detailed, systematic description of this satellite image scene:

SCENE OVERVIEW:
- Overall scene type (urban, rural, industrial, natural, mixed)
- Dominant landscape characteristics
- General layout and organization patterns

BUILT ENVIRONMENT:
- Building types, sizes, and patterns visible
- Road networks and transportation infrastructure
- Infrastructure density and development patterns

NATURAL FEATURES:
- Vegetation types and coverage
- Water bodies and terrain features
- Land use patterns

Organize your response clearly and be specific about what you can observe.""",

        "Infrastructure Analysis": """Analyze the infrastructure visible in this satellite image:

TRANSPORTATION:
- Road networks, intersections, and connectivity
- Parking areas and vehicle infrastructure

BUILDINGS:
- Building types, sizes, and construction patterns
- Density and spatial arrangement

UTILITIES:
- Power lines, communication towers, or utility infrastructure if visible

Focus only on clearly visible infrastructure elements.""",

        "Environmental Assessment": """Assess the environmental features in this satellite image:

VEGETATION:
- Types of vegetation visible
- Coverage density and health indicators
- Seasonal conditions if apparent

LAND USE:
- Agricultural areas, urban development, natural areas
- Land cover patterns and boundaries

WATER FEATURES:
- Rivers, lakes, ponds, or water bodies
- Drainage patterns if visible

Report only what is clearly observable in the image.""",

        "Land Use Classification": """Classify the land use patterns in this satellite image:

PRIMARY LAND USES:
- Residential areas and housing patterns
- Commercial/industrial zones
- Agricultural or undeveloped land
- Transportation infrastructure

DEVELOPMENT PATTERNS:
- Density gradients and spatial organization
- Mixed-use areas if present
- Boundary definitions between different uses

Provide specific observations about land use distribution."""
    }

# ---------------------- Sidebar Configuration ----------------------
st.sidebar.header("📥 Input & Settings")

# File upload
uploaded = st.sidebar.file_uploader(
    "Upload a satellite image", 
    type=["png", "jpg", "jpeg", "tif", "tiff"], 
    accept_multiple_files=False
)

# Enhanced image preprocessing
with st.sidebar.expander("🖼️ Image Processing", expanded=False):
    contrast = st.slider("Contrast", 0.5, 3.0, 1.2, step=0.1)
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, step=0.1)
    saturation = st.slider("Saturation", 0.0, 2.0, 1.1, step=0.1)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.2, step=0.1)
    edge_enhance = st.checkbox("Edge Enhancement", value=False)
    denoise = st.checkbox("Noise Reduction", value=False)

# Analysis configuration
enhanced_prompts = get_enhanced_prompts()
analysis_type = st.sidebar.selectbox(
    "Analysis Type:", 
    list(enhanced_prompts.keys()) + ["Custom"]
)

if analysis_type == "Custom":
    prompt = st.sidebar.text_area(
        "Custom Prompt:",
        "Describe what you see in this satellite image in detail.",
        height=100
    )
else:
    prompt = enhanced_prompts[analysis_type]

# Model parameters
with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    max_tokens = st.slider("Max Response Tokens", 150, 500, 300, step=50)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.85, step=0.05)

# ---------------------- Main Interface ----------------------
st.title("🛰️ Enhanced Satellite AI Analysis")
st.markdown("**Advanced satellite and aerial image analysis with LLaVA**")

# Model status indicator
if model is None or processor is None:
    st.error("⚠️ Model failed to load. Please check your environment and model availability.")
    st.stop()
else:
    st.success("✅ Model loaded successfully")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Image Preview")
    
    if uploaded:
        image = process_image(uploaded)
        if image:
            # Apply enhancements
            enhanced_image = enhance_satellite_image(
                image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
            )
            
            st.image(enhanced_image, caption=f"Enhanced: {uploaded.name}", use_column_width=True)
            
            # Display image info
            st.info(f"Image size: {enhanced_image.size[0]} x {enhanced_image.size[1]} pixels")
            
        else:
            st.error("Failed to process the uploaded image.")
    else:
        st.info("Upload a satellite image to begin analysis")

with col2:
    st.subheader("🔍 Analysis Results")
    
    if st.button("🚀 Run Analysis", type="primary", disabled=(uploaded is None)):
        if uploaded and model is not None:
            image = process_image(uploaded)
            if image:
                enhanced_image = enhance_satellite_image(
                    image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                )
                
                with st.spinner("Analyzing satellite image..."):
                    result = run_analysis(
                        enhanced_image, prompt, model, processor, torch_device,
                        max_tokens, temperature, top_p
                    )
                
                st.success("Analysis Complete!")
                st.write("**Analysis Result:**")
                st.write(result)
                
                # Store result in session state
                if 'analysis_results' not in st.session_state:
                    st.session_state.analysis_results = []
                
                analysis_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'image_name': uploaded.name,
                    'analysis_type': analysis_type,
                    'result': result,
                    'image_size': enhanced_image.size
                }
                
                st.session_state.analysis_results.append(analysis_data)
                
            else:
                st.error("Failed to process the image for analysis.")
        else:
            st.warning("Please upload an image and ensure the model is loaded.")

# ---------------------- Analysis History ----------------------
if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
    st.subheader("📊 Analysis History")
    
    history_data = []
    for result in st.session_state.analysis_results:
        history_data.append({
            'Timestamp': result['timestamp'],
            'Image': result['image_name'],
            'Analysis Type': result['analysis_type'],
            'Word Count': len(result['result'].split()),
            'Image Size': f"{result['image_size'][0]}x{result['image_size'][1]}"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df)
    
    # Export option
    if st.button("📥 Export Analysis History"):
        json_data = json.dumps(st.session_state.analysis_results, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"satellite_analysis_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ---------------------- System Information ----------------------
with st.expander("🔧 System Information", expanded=False):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Device Information:**")
        st.write(f"- PyTorch Version: {torch.__version__}")
        st.write(f"- CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"- CUDA Device: {torch.cuda.get_device_name(0)}")
            st.write(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    with col_b:
        st.write("**Model Information:**")
        st.write(f"- Model: {MODEL_NAME}")
        st.write(f"- Device: {torch_device}")
        st.write(f"- Quantization: 4-bit (NF4)")

# Memory cleanup button
if st.button("🧹 Clear Memory Cache"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.success("Memory cache cleared!")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enhanced Satellite AI Pro - Fixed Version</p>
    <p>Powered by LLaVA Vision Language Model</p>
</div>
""", unsafe_allow_html=True)
