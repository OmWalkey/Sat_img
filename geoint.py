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
st.set_page_config(page_title="Military Image Analysis Pro", layout="wide", page_icon="🛰️")

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
                        "text": f"You are an expert military image analyst. {prompt_text}"
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

# ---------------------- Enhanced Analysis Functions ----------------------
def determine_analysis_approach(image, model, processor, device):
    """
    First run military significance assessment to determine appropriate analysis approach
    """
    screening_prompt = """You are an expert image analyst. Examine this aerial/satellite image and determine if it contains military significance.

MILITARY INDICATORS TO LOOK FOR:
- Military vehicles, aircraft, or naval vessels
- Uniformed personnel in formations
- Military facilities, bases, or installations
- Weapons systems, radar installations, or defense structures
- Military equipment, vehicles, or hardware
- Fortifications, bunkers, or defensive positions

ASSESSMENT RESPONSE FORMAT:
1. MILITARY SIGNIFICANCE: [YES/NO]
2. CONFIDENCE LEVEL: [High/Medium/Low]
3. PRIMARY INDICATORS: [List specific military elements observed]
4. BRIEF DESCRIPTION: [2-3 sentences describing what you see]

If NO military significance is detected, provide a standard aerial image description focusing on civilian infrastructure, natural features, and general landscape characteristics."""
    
    # Run initial screening with conservative parameters
    screening_result = run_analysis(
        image, screening_prompt, model, processor, device,
        max_tokens=250, temperature=0.2, top_p=0.8
    )
    
    # Parse screening result to determine if military analysis is needed
    military_keywords = ["MILITARY SIGNIFICANCE: YES", "MILITARY", "VEHICLE", "WEAPON", 
                        "AIRCRAFT", "UNIFORM", "DEFENSE", "TACTICAL", "PERSONNEL", "EQUIPMENT"]
    
    has_military_significance = any(keyword in screening_result.upper() for keyword in military_keywords)
    
    return "military" if has_military_significance else "civilian", screening_result

def get_enhanced_military_analysis_prompts():
    """Enhanced prompts optimized for LLaVA 1.5 7B model"""
    return {
        "Military Significance Assessment": """You are an expert image analyst. Examine this aerial/satellite image and determine if it contains military significance.

MILITARY INDICATORS TO LOOK FOR:
- Military vehicles, aircraft, or naval vessels
- Uniformed personnel in formations
- Military facilities, bases, or installations
- Weapons systems, radar installations, or defense structures
- Military equipment, vehicles, or hardware
- Fortifications, bunkers, or defensive positions

ASSESSMENT RESPONSE FORMAT:
1. MILITARY SIGNIFICANCE: [YES/NO]
2. CONFIDENCE LEVEL: [High/Medium/Low]
3. PRIMARY INDICATORS: [List specific military elements observed]
4. BRIEF DESCRIPTION: [2-3 sentences describing what you see]

If NO military significance is detected, provide a standard aerial image description focusing on civilian infrastructure, natural features, and general landscape characteristics.""",

        "Basic Military Assessment": """You are a military image analyst. Analyze this aerial image and provide:

1. MILITARY ASSETS VISIBLE:
   - Vehicles (type, quantity, condition)
   - Aircraft or naval vessels
   - Personnel formations
   - Equipment and weapons systems

2. INFRASTRUCTURE:
   - Military facilities and buildings
   - Defensive positions
   - Transportation networks
   - Communication equipment

3. TACTICAL SITUATION:
   - Unit positioning and formation
   - Defensive or offensive posture
   - Activity level and readiness
   - Strategic advantages of location

4. THREAT ASSESSMENT:
   - Weapons capabilities observed
   - Defensive measures in place
   - Potential vulnerabilities
   - Operational implications

Keep responses factual and based only on what is clearly visible in the image.""",

        "Vehicle and Equipment Analysis": """Focus on identifying and analyzing military vehicles and equipment in this aerial image:

VEHICLE IDENTIFICATION:
- Count and classify all military vehicles visible
- Identify vehicle types (tanks, APCs, trucks, artillery)
- Assess vehicle condition and operational status
- Note any distinctive markings or configurations

EQUIPMENT ASSESSMENT:
- Weapons systems and their capabilities
- Communication equipment and antennas
- Support equipment and logistics assets
- Maintenance and repair facilities

DEPLOYMENT ANALYSIS:
- Formation patterns and tactical positioning
- Camouflage and concealment efforts
- Readiness indicators and activity levels
- Logistical support arrangements

Provide specific details about what you observe, including quantities, types, and conditions.""",

        "Facility and Infrastructure Analysis": """Analyze the military facilities and infrastructure visible in this aerial image:

FACILITY IDENTIFICATION:
- Command and control centers
- Barracks and administrative buildings
- Maintenance and repair facilities
- Storage and supply depots
- Training and exercise areas

INFRASTRUCTURE ASSESSMENT:
- Transportation networks (roads, railways, airfields)
- Communication systems and towers
- Power generation and distribution
- Water and fuel supply systems
- Defensive perimeters and barriers

OPERATIONAL CAPABILITY:
- Facility capacity and accommodation
- Operational readiness indicators
- Security measures and protection levels
- Logistical support capabilities

Focus on observable details and their military significance.""",

        "Activity and Movement Analysis": """Analyze military activity and movement patterns in this aerial image:

PERSONNEL ACTIVITY:
- Troop formations and movements
- Training exercises and drills
- Guard duties and security patrols
- Maintenance and support activities

VEHICLE MOVEMENT:
- Vehicle positioning and deployment
- Movement patterns and directions
- Convoy formations and logistics
- Tactical positioning changes

OPERATIONAL TEMPO:
- Activity level indicators
- Readiness state assessment
- Exercise vs. operational activities
- Maintenance and support operations

Report only what is directly observable in the image.""",

        "Defensive Position Analysis": """Examine defensive positions and fortifications in this aerial image:

DEFENSIVE STRUCTURES:
- Bunkers, trenches, and fighting positions
- Barriers, obstacles, and perimeter defenses
- Camouflaged and concealed positions
- Hardened shelters and protective structures

WEAPONS POSITIONS:
- Artillery and mortar positions
- Anti-aircraft defense systems
- Direct fire weapons emplacements
- Observation and fire control positions

TACTICAL LAYOUT:
- Fields of fire and coverage areas
- Interlocking fire patterns
- Dead space and blind spots
- Support and supply routes

Focus on defensive capabilities and tactical advantages.""",

        "Standard Aerial Image Analysis": """You are a professional aerial image analyst. Provide a comprehensive description of this aerial/satellite image:

GENERAL OVERVIEW:
- Image type and quality
- Geographic setting and terrain
- Weather and lighting conditions
- Scale and resolution assessment

LAND USE AND FEATURES:
- Urban, suburban, or rural characteristics
- Residential, commercial, or industrial areas
- Agricultural lands and vegetation
- Natural features (water bodies, forests, mountains)

INFRASTRUCTURE:
- Transportation networks (roads, railways, airports)
- Utilities and communication lines
- Public facilities and services
- Commercial and industrial facilities

NOTABLE FEATURES:
- Distinctive landmarks or structures
- Construction or development activities
- Environmental features or concerns
- Cultural or historical significance

Provide accurate, objective observations based solely on what is visible in the image."""
    }

# ---------------------- Sidebar Configuration ----------------------
st.sidebar.header("📥 Input & Settings")

# File upload
uploaded = st.sidebar.file_uploader(
    "Upload an aerial/satellite image", 
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

# Enhanced Analysis configuration
enhanced_military_prompts = get_enhanced_military_analysis_prompts()

# Add auto-screening option
auto_screen = st.sidebar.checkbox("Auto-detect military significance", value=True)

if auto_screen:
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:", 
        ["Auto-detect"] + list(enhanced_military_prompts.keys()) + ["Custom"]
    )
else:
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:", 
        list(enhanced_military_prompts.keys()) + ["Custom"]
    )

if analysis_type == "Custom":
    prompt = st.sidebar.text_area(
        "Custom Prompt:",
        "Analyze this aerial image using professional standards.",
        height=100
    )
elif analysis_type == "Auto-detect":
    prompt = None  # Will be determined by screening
else:
    prompt = enhanced_military_prompts[analysis_type]

# Model parameters
with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    max_tokens = st.slider("Max Response Tokens", 150, 500, 300, step=50)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.85, step=0.05)

# ---------------------- Main Interface ----------------------
st.title("🛰️ Military Image Analysis Pro")
st.markdown("**Advanced military aerial image analysis with LLaVA Vision Language Model**")

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
        st.info("Upload an aerial/satellite image to begin analysis")

with col2:
    st.subheader("🔍 Analysis Results")
    
    if st.button("🚀 Run Analysis", type="primary", disabled=(uploaded is None)):
        if uploaded and model is not None:
            image = process_image(uploaded)
            if image:
                enhanced_image = enhance_satellite_image(
                    image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                )
                
                if analysis_type == "Auto-detect" or auto_screen:
                    # Run screening first
                    with st.spinner("Screening image for military significance..."):
                        analysis_approach, screening_result = determine_analysis_approach(
                            enhanced_image, model, processor, torch_device
                        )
                    
                    st.info(f"**Screening Result:** {analysis_approach.title()} image detected")
                    
                    # Show screening details in expander
                    with st.expander("📋 Screening Details"):
                        st.write(screening_result)
                    
                    # Determine appropriate analysis
                    if analysis_approach == "military":
                        if analysis_type == "Auto-detect":
                            # Use best military analysis prompt
                            final_prompt = enhanced_military_prompts["Basic Military Assessment"]
                        else:
                            final_prompt = prompt
                    else:
                        # Use civilian analysis
                        final_prompt = enhanced_military_prompts["Standard Aerial Image Analysis"]
                    
                    # Run detailed analysis
                    with st.spinner("Running detailed analysis..."):
                        result = run_analysis(
                            enhanced_image, final_prompt, model, processor, torch_device,
                            max_tokens, temperature, top_p
                        )
                    
                    analysis_used = "Basic Military Assessment" if analysis_approach == "military" and analysis_type == "Auto-detect" else analysis_type
                    
                else:
                    # Direct analysis without screening
                    with st.spinner("Analyzing aerial image..."):
                        result = run_analysis(
                            enhanced_image, prompt, model, processor, torch_device,
                            max_tokens, temperature, top_p
                        )
                    analysis_used = analysis_type
                    analysis_approach = "unknown"
                
                st.success("Analysis Complete!")
                st.write("**Analysis Result:**")
                st.write(result)
                
                # Store result in session state
                if 'analysis_results' not in st.session_state:
                    st.session_state.analysis_results = []
                
                analysis_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'image_name': uploaded.name,
                    'analysis_type': analysis_used,
                    'military_significance': analysis_approach,
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
            'Military Significance': result['military_significance'],
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
            file_name=f"military_analysis_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
    <p>Military Image Analysis Pro</p>
    <p>Powered by LLaVA Vision Language Model</p>
</div>
""", unsafe_allow_html=True)
