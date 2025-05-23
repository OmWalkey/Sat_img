import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import numpy as np
import gc

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Enhanced Satellite AI", layout="wide", page_icon="🛰️")

# ---------------------- Model & Quantization Config ----------------------
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
BQB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=BQB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="./offload",
        offload_state_dict=True
    )
    return model, processor

# Initialize model and processor
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, processor = load_model()
model.to(torch_device)

# ---------------------- Enhanced Image Processing ----------------------
def enhance_satellite_image(image, contrast=1.2, brightness=1.0, saturation=1.1, sharpness=1.0, 
                          edge_enhance=False, denoise=False):
    """Enhanced preprocessing specifically for satellite imagery"""
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

def create_mosaic(images, cols=2):
    """Create mosaic with better handling"""
    imgs = [img.copy() for img in images]
    if not imgs:
        return None
    
    # Calculate optimal size
    widths, heights = zip(*(i.size for i in imgs))
    target_w, target_h = min(widths), min(heights)
    
    # Resize all images
    imgs = [i.resize((target_w, target_h), Image.Resampling.LANCZOS) for i in imgs]
    
    # Create mosaic
    rows = (len(imgs) + cols - 1) // cols
    mosaic = Image.new('RGB', (cols * target_w, rows * target_h))
    
    for idx, img in enumerate(imgs):
        x = (idx % cols) * target_w
        y = (idx // cols) * target_h
        mosaic.paste(img, (x, y))
    
    return mosaic

# ---------------------- Improved Prompts ----------------------
def get_enhanced_prompts():
    return {
        "Precise Feature Detection": """This is a satellite or aerial image. Analyze it with extreme precision and accuracy. Focus only on what you can clearly see:

VISIBLE STRUCTURES:
- List only buildings/structures you can definitively identify
- Describe their shape, size, and arrangement patterns
- Note rooftop colors and materials if visible

TRANSPORTATION:
- Identify roads, paths, or tracks you can clearly see
- Describe their width, condition, and connectivity
- Note any vehicles only if clearly visible

TERRAIN & LAND COVER:
- Describe ground surface types (paved, dirt, grass, etc.)
- Identify vegetation only where clearly present
- Note terrain elevation changes if visible

WATER FEATURES:
- ONLY mention water if you can clearly see reflective surfaces, rivers, or water bodies
- Do not assume water exists based on terrain alone

Be specific about what you observe versus what you infer. Use phrases like "appears to be" only when uncertain.""",

        "Infrastructure Analysis": """Analyze this satellite image focusing specifically on human-made infrastructure. Be precise and factual:

BUILDINGS & STRUCTURES:
- Count and describe visible buildings by size/type
- Note construction materials evident from appearance
- Identify industrial vs residential vs commercial if distinguishable

TRANSPORTATION NETWORK:
- Map out road hierarchy (major/minor roads)
- Identify intersections, roundabouts, parking areas
- Note railway lines, bridges, or overpasses if present

UTILITIES & SERVICES:
- Identify power lines, substations, or towers
- Note communication towers or antennas
- Spot storage tanks, industrial facilities

Only describe infrastructure you can clearly observe. Avoid speculation about purpose or function unless obvious.""",

        "Land Use Classification": """Classify the land use in this satellite image with specific percentages and precise boundaries:

LAND USE CATEGORIES (estimate % coverage):
- Residential areas: __%
- Commercial/Industrial: __%
- Transportation (roads/parking): __%
- Undeveloped/Natural: __%
- Agricultural (if visible): __%

SPATIAL PATTERNS:
- Describe how different uses are arranged
- Note density gradients or clustering
- Identify clear boundaries between zones

DEVELOPMENT CHARACTERISTICS:
- Building density and spacing
- Street layout patterns (grid, organic, planned)
- Infrastructure completeness

Base classifications only on clearly visible features, not assumptions about typical development patterns.""",

        "Environmental Assessment": """Assess environmental conditions visible in this satellite image:

VEGETATION:
- Describe actual vegetation visible (trees, grass, crops)
- Assess health/density only where clearly observable
- Note seasonal conditions if apparent

WATER RESOURCES:
- Identify ONLY visible water bodies, streams, or wet areas
- Assess water clarity/color if discernible
- Note drainage patterns or flood evidence

TERRAIN:
- Describe topography and slope if visible
- Identify erosion, disturbed soil, or construction scars
- Note natural vs modified landscapes

ENVIRONMENTAL IMPACTS:
- Visible pollution, waste, or degradation
- Construction or development impacts
- Any signs of environmental stress

Report only what is clearly observable in the image.""",

        "Activity & Movement Analysis": """Analyze human activity and movement patterns visible in this satellite image:

VEHICLES & TRAFFIC:
- Count vehicles only if clearly visible as distinct objects
- Note traffic patterns on major roads if discernible
- Identify parking areas and occupancy levels

HUMAN ACTIVITY INDICATORS:
- Construction sites or active development
- Agricultural activity (equipment, field patterns)
- Commercial/industrial operations if evident

TEMPORAL INDICATORS:
- Evidence of recent activity (fresh construction, new roads)
- Seasonal patterns (agricultural cycles, snow, flooding)
- Time-of-day indicators (shadows, activity levels)

Be extremely conservative - only report activity you can definitively observe, not infer."""
    }

# ---------------------- Sidebar: Enhanced Inputs ----------------------
st.sidebar.header("📥 Input & Settings")

# Mode selection
mode = st.sidebar.radio(
    "Select input mode:", ["Single Image", "Tiled Images"], key="mode_radio"
)

# Enhanced image preprocessing
with st.sidebar.expander("🖼️ Advanced Image Processing", expanded=False):
    contrast = st.slider("Contrast", 0.5, 3.0, 1.2, step=0.1, key="contrast_slider")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, step=0.1, key="brightness_slider")
    saturation = st.slider("Saturation", 0.0, 2.0, 1.1, step=0.1, key="saturation_slider")
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.2, step=0.1, key="sharpness_slider")
    edge_enhance = st.checkbox("Edge Enhancement", value=False, key="edge_checkbox")
    denoise = st.checkbox("Noise Reduction", value=False, key="denoise_checkbox")

# File upload
if mode == "Single Image":
    uploaded = st.sidebar.file_uploader(
        "Upload a satellite image", type=["png","jpg","jpeg","tif","tiff"], 
        accept_multiple_files=False, key="upload_single"
    )
    uploaded_list = None
else:
    uploaded_list = st.sidebar.file_uploader(
        "Upload up to 4 image tiles", type=["png","jpg","jpeg","tif","tiff"], 
        accept_multiple_files=True, key="upload_tiles"
    )
    uploaded = None

# Enhanced analysis prompts
enhanced_prompts = get_enhanced_prompts()
analysis_type = st.sidebar.selectbox(
    "Analysis Type:", list(enhanced_prompts.keys()) + ["Custom"], key="analysis_select"
)

if analysis_type == "Custom":
    prompt = st.sidebar.text_area(
        "Custom Prompt:", 
        value="Analyze this satellite image with precise detail, focusing only on clearly visible features...", 
        height=150, key="custom_prompt"
    )
else:
    prompt = enhanced_prompts[analysis_type]

# Advanced generation settings
with st.sidebar.expander("🎛️ Generation Settings", expanded=True):
    length_mode = st.radio("Response Length:", ["Brief", "Detailed"], index=1, key="length_radio")
    
    if length_mode == "Brief":
        max_tokens = st.slider("Max Tokens", 50, 200, 100, key="tokens_brief")
        temperature = 0.1
        do_sample = True
    else:
        max_tokens = st.slider("Max Tokens", 200, 1024, 600, key="tokens_detailed")
        temperature = st.slider("Temperature", 0.1, 0.8, 0.3, key="temp_slider")
        do_sample = True
    
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.85, key="top_p_slider")
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.5, 1.1, key="rep_penalty")
    
    use_beams = st.checkbox("Use Beam Search", value=False, key="beam_checkbox")
    beams = st.slider("Num Beams", 1, 5, 3, key="beam_slider") if use_beams else 1

run_btn = st.sidebar.button("🚀 Run Enhanced Analysis", key="run_button", type="primary")

# ---------------------- Main Interface ----------------------
st.title("🛰️ Enhanced Satellite Image Analysis")
st.markdown("*Precision-focused satellite imagery analysis with advanced preprocessing*")

col1, col2 = st.columns((2, 3))

with col1:
    st.subheader("📸 Image Preview")
    image = None
    
    if mode == "Single Image":
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img = enhance_satellite_image(
                img, contrast, brightness, saturation, sharpness, edge_enhance, denoise
            )
            image = img
            st.image(image, caption="Enhanced Satellite Image", use_column_width=True)
            
            # Show image info
            st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
        else:
            st.warning("Please upload a satellite image to begin analysis.")
    
    else:  # Tiled mode
        if uploaded_list:
            if len(uploaded_list) > 4:
                st.warning("Using first 4 images only.")
            
            imgs = [Image.open(f).convert("RGB") for f in uploaded_list[:4]]
            enhanced = []
            for im in imgs:
                enhanced_img = enhance_satellite_image(
                    im, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                )
                enhanced.append(enhanced_img)
            
            image = create_mosaic(enhanced, cols=2)
            st.image(image, caption=f"Enhanced Mosaic ({len(enhanced)} tiles)", use_column_width=True)
            st.info(f"**Tiles:** {len(enhanced)} | **Final Size:** {image.size[0]} × {image.size[1]} pixels")
        else:
            st.warning("Please upload 2-4 image tiles to create a mosaic.")

with col2:
    st.subheader("⚙️ Analysis Configuration")
    
    # Show current settings
    with st.expander("Current Settings", expanded=True):
        st.markdown(f"""
        **Mode:** {mode}  
        **Analysis:** {analysis_type}  
        **Max Tokens:** {max_tokens}  
        **Temperature:** {temperature}  
        **Preprocessing:** Contrast={contrast}, Brightness={brightness}, Saturation={saturation}
        """)
    
    # Instructions
    st.markdown("""
    **Instructions:**
    1. Upload high-quality satellite/aerial imagery
    2. Adjust preprocessing to enhance feature visibility
    3. Select analysis type or create custom prompt
    4. Fine-tune generation parameters
    5. Run analysis for precise, accurate results
    
    **Tips for Better Results:**
    - Use high-resolution images when possible
    - Adjust contrast/brightness to highlight features
    - Use specific prompts for targeted analysis
    - Lower temperature for more factual responses
    """)

# Stop if no image
if not image:
    st.stop()

# ---------------------- Enhanced Inference ----------------------
if run_btn and image is not None:
    with st.spinner("🔍 Running precision analysis..."):
        try:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Prepare enhanced prompt
            system_prompt = "You are an expert satellite image analyst. Provide precise, accurate descriptions based only on clearly visible features. Avoid speculation or generic descriptions."
            full_prompt = f"{system_prompt}\n\nImage Analysis Request:\n{prompt.strip()}"
            
            # Process inputs
            inputs = processor(text=full_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(torch_device) for k, v in inputs.items()}
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "length_penalty": 1.0,
            }
            
            if use_beams and beams > 1:
                gen_kwargs.update({
                    "num_beams": beams, 
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3
                })
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode result
            input_len = inputs['input_ids'].shape[1]
            gen_tokens = outputs[0][input_len:]
            result = processor.decode(gen_tokens, skip_special_tokens=True)
            
            # Clean up result
            result = result.strip()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()
    
    # Display results
    st.header("🎯 Analysis Results")
    
    # Check for generic responses
    generic_indicators = [
        "appears to be a typical", "seems to show", "likely contains",
        "probably has", "might be", "could be", "generic"
    ]
    
    is_generic = any(indicator in result.lower() for indicator in generic_indicators)
    
    if is_generic:
        st.warning("⚠️ Response may be generic. Try adjusting preprocessing settings or using a more specific prompt.")
    
    # Show result
    st.markdown(result)
    
    # Analysis metadata
    with st.expander("Analysis Metadata"):
        st.json({
            "tokens_generated": len(gen_tokens),
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "beam_search": use_beams,
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "preprocessing_applied": {
                "contrast": contrast,
                "brightness": brightness,
                "saturation": saturation,
                "sharpness": sharpness,
                "edge_enhance": edge_enhance,
                "denoise": denoise
            }
        })
    
    # Feedback
    st.divider()
    feedback = st.radio("Was this analysis accurate and specific?", 
                       ["👍 Yes, very accurate", "👎 No, too generic", "🤔 Partially accurate"], 
                       key="feedback_radio")
    
    if feedback == "👎 No, too generic":
        st.info("💡 **Tips to improve accuracy:**\n- Try different preprocessing settings\n- Use more specific prompts\n- Lower the temperature\n- Ensure image quality is high")
    elif feedback == "🤔 Partially accurate":
        st.info("💡 **Try:**\n- Adjusting contrast/brightness to highlight specific features\n- Using targeted analysis types\n- Breaking complex scenes into smaller regions")

# ---------------------- Footer ----------------------
st.divider()
st.markdown("*Enhanced Satellite AI v2.0 - Precision-focused analysis with advanced preprocessing*")
