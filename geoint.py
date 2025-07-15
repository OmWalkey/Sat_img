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
st.set_page_config(page_title="GeoInt", layout="wide", page_icon="🛰️")

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

# ---------------------- Military Analysis Prompts ----------------------
def get_military_analysis_prompts():
    return {
        "Basic Image Description": """Provide a concise, accurate description of this aerial/satellite image using military observation standards:

IMAGERY METADATA:
- Image type (aerial photograph, satellite imagery, infrared, etc.)
- Approximate resolution and scale
- Time indicators (shadows, lighting conditions, seasonal markers)
- Weather conditions affecting visibility

GEOGRAPHIC CONTEXT:
- Terrain type and topography
- General location characteristics (urban, rural, coastal, mountainous)
- Dominant land use patterns
- Natural boundaries and features

STRUCTURAL OBSERVATIONS:
- Built structures present (count, type, arrangement)
- Transportation infrastructure (roads, railways, airfields)
- Utility infrastructure visible
- Industrial or military facilities

ACTIVITY INDICATORS:
- Vehicle presence and movement
- Personnel activity if observable
- Equipment or material stockpiles
- Signs of recent construction or changes

TACTICAL SIGNIFICANCE:
- Key terrain features
- Mobility corridors and restrictions
- Observation points and fields of fire
- Defensive or strategic value

Report only what is directly observable. Use precise military terminology. Distinguish between confirmed observations and probable assessments.""",

        "Target Identification and Classification": """Analyze this aerial imagery for target identification using military classification standards:

MILITARY TARGETS (BY PRIORITY):
- Command and Control facilities
- Air Defense Systems (radar, missile sites, AAA)
- Military vehicles and equipment (identify type, quantity, condition)
- Personnel concentrations and formations
- Weapons storage facilities and ammunition depots

INFRASTRUCTURE TARGETS:
- Critical nodes (power generation, substations, transformers)
- Transportation hubs (bridges, rail junctions, ports)
- Communication facilities (towers, switching stations)
- Fuel storage and distribution systems
- Manufacturing facilities with military relevance

TARGET CHARACTERISTICS:
- Precise dimensions and coordinates
- Hardness assessment (construction type, protection level)
- Accessibility for targeting systems
- Collateral damage considerations
- Time-sensitive target indicators

DEFENSIVE MEASURES:
- Camouflage, concealment, and deception efforts
- Hardened shelters and protective structures
- Air defense coverage and blind spots
- Electronic warfare equipment signatures

BATTLE DAMAGE ASSESSMENT BASELINES:
- Pre-strike facility conditions
- Key structural components and vulnerabilities
- Functional assessment of target systems
- Repair timelines and capability estimates

Report using standardized target nomenclature and NATO target classification systems.""",

        "Tactical Infrastructure Analysis": """Conduct tactical-level infrastructure analysis for operational planning:

MILITARY INFRASTRUCTURE:
- Classify facilities by function (command, logistics, maintenance, housing)
- Assess capacity and operational capability
- Identify supporting infrastructure requirements
- Note security measures and force protection

TRANSPORTATION MILITARY UTILITY:
- Road classification for military traffic (MLC ratings where assessable)
- Bridge load capacities and bypass routes
- Airfield facilities and aircraft handling capability
- Railroad capacity for military logistics
- Port/harbor facilities for amphibious operations

UTILITIES AND SERVICES:
- Power generation capacity and distribution vulnerabilities
- Water treatment and distribution systems
- Fuel storage and distribution networks
- Communications infrastructure and switching nodes

DEFENSIVE CONSIDERATIONS:
- Natural defensive positions and fields of fire
- Obstacle placement and effectiveness
- Ingress/egress routes and vulnerability
- Dead space and concealment opportunities

ENGINEER REQUIREMENTS:
- Mobility enhancement requirements
- Countermobility opportunities
- Survivability positions and improvements
- Infrastructure hardening assessments

Use military engineer terminology and provide assessments relevant to tactical operations.""",

        "Order of Battle Analysis": """Analyze imagery for Order of Battle (OB) intelligence indicators:

UNIT IDENTIFICATION:
- Vehicle types and quantities (track vs wheeled, combat vs support)
- Equipment signatures and technical characteristics
- Unit marking or identification symbols if visible
- Formation patterns and tactical deployments

FORCE STRUCTURE INDICATORS:
- Estimated unit size based on equipment/facility footprint
- Command structure indicators (headquarters facilities, antennas)
- Support element identification (maintenance, supply, medical)
- Reserve or reinforcement capabilities

ACTIVITY PATTERNS:
- Training activities and exercise indicators
- Operational tempo and readiness levels
- Logistics resupply patterns
- Personnel movement and concentration

CAPABILITY ASSESSMENT:
- Weapons systems capabilities and ranges
- Mobility and maneuver capability
- Sustainment and endurance factors
- Electronic warfare and communications capability

DEFENSIVE POSTURE:
- Defensive positions and fighting positions
- Obstacle construction and placement
- Fire support positions and coordination
- Reserve positioning and commitment options

TEMPORAL ANALYSIS:
- Changes in force posture over time
- Seasonal deployment patterns
- Exercise cycles and training periods
- Maintenance and refit schedules

Report findings using standard OB format with unit designations where determinable.""",

        "Threat Assessment and Force Protection": """Analyze imagery for threat assessment and force protection requirements:

DIRECT FIRE THREATS:
- Anti-tank positions and fields of fire
- Sniper positions and overwatch locations
- Direct fire weapon systems and ranges
- Improvised fighting positions

INDIRECT FIRE THREATS:
- Artillery and mortar positions
- Rocket launch sites and storage
- Forward observer positions
- Fire direction centers

AIR DEFENSE THREATS:
- Surface-to-air missile sites
- Anti-aircraft artillery positions
- Radar installations and coverage
- Electronic warfare systems

ASYMMETRIC THREATS:
- IED emplacement indicators
- Ambush sites and kill zones
- Cache locations and supply routes
- Surveillance and reconnaissance positions

FORCE PROTECTION MEASURES:
- Perimeter security and access control
- Hardened shelters and bunkers
- Early warning systems
- Reaction force positions

VULNERABILITY ASSESSMENT:
- Approach routes and dead space
- Blind spots and surveillance gaps
- Critical facility protection levels
- Evacuation routes and procedures

COUNTERMEASURES:
- Active protection systems
- Camouflage and concealment effectiveness
- Electronic countermeasures
- Deception operations indicators

Use threat assessment matrices and force protection condition classifications.""",

        "Battle Damage Assessment": """Conduct Battle Damage Assessment (BDA) analysis of target engagement:

TARGET STATUS:
- Functional damage assessment (destroyed, damaged, intact)
- Structural integrity evaluation
- Operational capability remaining
- Repair timeline estimates

DAMAGE INDICATORS:
- Blast damage patterns and crater analysis
- Fire damage and secondary explosions
- Structural collapse or deformation
- Equipment destruction or displacement

MISSION EFFECTIVENESS:
- Target function disruption percentage
- Critical components affected
- Redundancy and backup systems status
- Overall mission kill assessment

COLLATERAL DAMAGE:
- Civilian infrastructure impact
- Non-target facility damage
- Environmental considerations
- Population displacement indicators

ENEMY RESPONSE:
- Damage control and repair efforts
- Personnel evacuation patterns
- Security posture changes
- Operational adaptation indicators

FOLLOW-UP REQUIREMENTS:
- Additional strikes needed
- Target restrike recommendations
- Alternate target considerations
- Surveillance requirements

INTELLIGENCE VALUE:
- Lessons learned for future targeting
- Weapon system effectiveness
- Target hardening assessment
- Defensive countermeasure evaluation

Report using standardized BDA terminology and effectiveness scales.""",

        "Special Operations Analysis": """Analyze imagery for special operations planning and execution:

INFILTRATION/EXFILTRATION:
- Helicopter landing zones (primary and alternate)
- Approach routes and concealment
- Obstacle avoidance and bypass routes
- Extraction point suitability

TARGET ACCESS:
- Facility layout and room clearing considerations
- Entry and exit points assessment
- Internal movement routes
- Hostage/personnel location indicators

SECURITY POSTURE:
- Guard positions and patrol patterns
- Sensor placement and coverage
- Reaction force capabilities
- Alert states and posture changes

RECONNAISSANCE REQUIREMENTS:
- Observation post locations
- Surveillance device placement
- Counter-surveillance considerations
- Intelligence collection opportunities

DIRECT ACTION PLANNING:
- Assault positions and fire support
- Breach points and methods
- Casualty evacuation routes
- Rally points and safe houses

SUPPORT REQUIREMENTS:
- Fire support coordination
- Medical evacuation planning
- Logistics and resupply
- Communications relay points

RISK ASSESSMENT:
- Civilian presence and protection
- Collateral damage potential
- Operational security threats
- Mission abort criteria

CONTINGENCY PLANNING:
- Alternate courses of action
- Emergency procedures
- Backup extraction methods
- Deception and misdirection options

Use special operations terminology and planning factors.""",

        "Geospatial Intelligence Analysis": """Conduct comprehensive GEOINT analysis for operational planning:

COORDINATE REFERENCE:
- Precise geographic coordinates (MGRS format)
- Elevation data and contour analysis
- Aspect and slope calculations
- Datum and projection specifications

SPATIAL RELATIONSHIPS:
- Distance and bearing calculations
- Line-of-sight analysis
- Intervisibility assessments
- Spatial pattern analysis

TEMPORAL ANALYSIS:
- Multi-temporal change detection
- Seasonal variation patterns
- Activity cycle identification
- Predictive pattern modeling

MULTI-SPECTRAL ANALYSIS:
- Vegetation health and camouflage effectiveness
- Water body analysis and depth estimation
- Soil composition and trafficability
- Thermal signatures if applicable

MEASUREMENT AND SIGNATURE INTELLIGENCE:
- Facility dimensions and capacities
- Equipment specifications and capabilities
- Communication signature analysis
- Electronic emissions patterns

PREDICTIVE ANALYSIS:
- Future development patterns
- Threat evolution projections
- Environmental impact assessments
- Operational window identification

DISSEMINATION REQUIREMENTS:
- Classification levels and handling
- Distribution lists and need-to-know
- Update frequency requirements
- Format specifications for end users

Format using standardized GEOINT reporting procedures and coordinate systems.""",

        "Counterintelligence Analysis": """Conduct counterintelligence analysis of aerial imagery:

COLLECTION INDICATORS:
- Surveillance equipment and positions
- Communication interception capabilities
- Reconnaissance activity patterns
- Intelligence collection signatures

SECURITY MEASURES:
- Camouflage and concealment effectiveness
- Operational security violations
- Personnel security indicators
- Information security practices

DECEPTION OPERATIONS:
- Decoy installations and equipment
- Misdirection indicators
- False signature creation
- Camouflage pattern analysis

FOREIGN INTELLIGENCE THREATS:
- Known collection platforms
- Surveillance pattern analysis
- Communication monitoring equipment
- Human intelligence indicators

VULNERABILITY ASSESSMENT:
- Signature management failures
- Predictable pattern identification
- Critical information exposure
- Operational security gaps

COUNTERMEASURE EFFECTIVENESS:
- Concealment success rates
- Deception operation results
- Electronic countermeasure performance
- Physical security measure adequacy

THREAT MITIGATION:
- Signature reduction recommendations
- Operational pattern changes
- Security enhancement requirements
- Deception operation planning

INTELLIGENCE PREPARATION:
- Threat capability assessment
- Collection probability analysis
- Countermeasure planning
- Risk mitigation strategies

Report using counterintelligence terminology and threat assessment frameworks."""
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

# Analysis configuration
military_prompts = get_military_analysis_prompts()
analysis_type = st.sidebar.selectbox(
    "Analysis Type:", 
    list(military_prompts.keys()) + ["Custom"]
)

if analysis_type == "Custom":
    prompt = st.sidebar.text_area(
        "Custom Prompt:",
        "Analyze this aerial image using military intelligence standards.",
        height=100
    )
else:
    prompt = military_prompts[analysis_type]

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
                
                with st.spinner("Analyzing aerial image..."):
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
    <p>GeoInt</p>
    <p> by AICoE</p>
</div>
""", unsafe_allow_html=True)
