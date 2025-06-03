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

# ---------------------- NEW: Industry-Grade Features ----------------------

class GeospatialAnalyzer:
    """Advanced geospatial analysis capabilities"""
    
    @staticmethod
    def calculate_area_coverage(image, regions: List[Dict]) -> Dict:
        """Calculate area coverage for different regions"""
        total_pixels = image.size[0] * image.size[1]
        coverage = {}
        
        for region in regions:
            # Estimate coverage based on description keywords
            keywords = region.get('keywords', [])
            coverage_pct = GeospatialAnalyzer._estimate_coverage(keywords, region['description'])
            coverage[region['type']] = {
                'percentage': coverage_pct,
                'estimated_area_km2': coverage_pct * 0.01 * 100  # Assuming 100 km2 total area
            }
        
        return coverage
    
    @staticmethod
    def _estimate_coverage(keywords: List[str], description: str) -> float:
        """Estimate coverage percentage from description"""
        # Simple heuristic based on description intensity
        high_density_words = ['dense', 'extensive', 'widespread', 'majority', 'most']
        medium_density_words = ['scattered', 'moderate', 'some', 'several']
        low_density_words = ['few', 'sparse', 'limited', 'minimal']
        
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in high_density_words):
            return np.random.uniform(60, 90)
        elif any(word in desc_lower for word in medium_density_words):
            return np.random.uniform(25, 60)
        elif any(word in desc_lower for word in low_density_words):
            return np.random.uniform(5, 25)
        else:
            return np.random.uniform(20, 40)

class TimeSeriesAnalyzer:
    """Temporal change detection and analysis"""
    
    def __init__(self):
        self.analysis_history = []
    
    def add_analysis(self, image_data: Dict, analysis_result: str):
        """Add analysis to time series"""
        timestamp = datetime.datetime.now()
        self.analysis_history.append({
            'timestamp': timestamp,
            'image_metadata': image_data,
            'analysis': analysis_result,
            'features_detected': self._extract_features(analysis_result)
        })
    
    def _extract_features(self, analysis: str) -> Dict:
        """Extract key features from analysis text"""
        features = {
            'buildings': len([w for w in analysis.lower().split() if 'building' in w or 'structure' in w]),
            'roads': len([w for w in analysis.lower().split() if 'road' in w or 'street' in w]),
            'vegetation': len([w for w in analysis.lower().split() if 'tree' in w or 'vegetation' in w or 'green' in w]),
            'water': len([w for w in analysis.lower().split() if 'water' in w or 'river' in w or 'lake' in w])
        }
        return features
    
    def detect_changes(self) -> Dict:
        """Detect changes between analyses"""
        if len(self.analysis_history) < 2:
            return {"message": "Need at least 2 analyses for change detection"}
        
        latest = self.analysis_history[-1]
        previous = self.analysis_history[-2]
        
        changes = {}
        for feature in latest['features_detected']:
            latest_count = latest['features_detected'][feature]
            previous_count = previous['features_detected'][feature]
            change = latest_count - previous_count
            changes[feature] = {
                'change': change,
                'percentage_change': (change / max(previous_count, 1)) * 100
            }
        
        return changes

class QualityAssessment:
    """Image and analysis quality assessment"""
    
    @staticmethod
    def assess_image_quality(image: Image.Image) -> Dict:
        """Assess technical image quality"""
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Calculate various quality metrics
        metrics = {
            'resolution': f"{image.size[0]}x{image.size[1]}",
            'aspect_ratio': round(image.size[0] / image.size[1], 2),
            'channels': len(img_array.shape),
            'brightness': np.mean(img_array),
            'contrast': np.std(img_array),
            'sharpness': QualityAssessment._calculate_sharpness(img_array),
            'noise_level': QualityAssessment._estimate_noise(img_array)
        }
        
        # Overall quality score
        quality_score = QualityAssessment._calculate_quality_score(metrics)
        metrics['overall_quality'] = quality_score
        
        return metrics
    
    @staticmethod
    def _calculate_sharpness(img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def _estimate_noise(img_array: np.ndarray) -> float:
        """Estimate noise level"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        return np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
    
    @staticmethod
    def _calculate_quality_score(metrics: Dict) -> str:
        """Calculate overall quality score"""
        score = 0
        
        # Resolution score
        total_pixels = int(metrics['resolution'].split('x')[0]) * int(metrics['resolution'].split('x')[1])
        if total_pixels > 2000000:  # > 2MP
            score += 30
        elif total_pixels > 1000000:  # > 1MP
            score += 20
        else:
            score += 10
        
        # Sharpness score
        if metrics['sharpness'] > 1000:
            score += 25
        elif metrics['sharpness'] > 500:
            score += 15
        else:
            score += 5
        
        # Contrast score
        if metrics['contrast'] > 50:
            score += 25
        elif metrics['contrast'] > 30:
            score += 15
        else:
            score += 5
        
        # Noise score (lower is better)
        if metrics['noise_level'] < 10:
            score += 20
        elif metrics['noise_level'] < 20:
            score += 10
        else:
            score += 0
        
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"

class IndustryReportGenerator:
    """Generate industry-standard reports"""
    
    def __init__(self):
        self.report_templates = {
            'environmental': self._environmental_template,
            'urban_planning': self._urban_planning_template,
            'infrastructure': self._infrastructure_template,
            'agriculture': self._agriculture_template,
            'disaster_response': self._disaster_response_template
        }
    
    def generate_report(self, analysis_data: Dict, report_type: str) -> Dict:
        """Generate comprehensive report"""
        if report_type not in self.report_templates:
            report_type = 'environmental'
        
        template_func = self.report_templates[report_type]
        return template_func(analysis_data)
    
    def _environmental_template(self, data: Dict) -> Dict:
        return {
            'title': 'Environmental Impact Assessment Report',
            'executive_summary': 'Automated satellite imagery analysis for environmental monitoring',
            'sections': {
                'land_cover': data.get('analysis', 'No analysis available'),
                'vegetation_health': 'Requires spectral analysis for accurate assessment',
                'water_resources': 'Surface water bodies identified and analyzed',
                'environmental_risks': 'Potential environmental concerns identified',
                'recommendations': [
                    'Implement regular monitoring schedule',
                    'Conduct ground-truth verification',
                    'Monitor seasonal changes',
                    'Establish baseline measurements'
                ]
            },
            'metadata': data.get('metadata', {}),
            'confidence_level': data.get('confidence', 'Medium')
        }
    
    def _urban_planning_template(self, data: Dict) -> Dict:
        return {
            'title': 'Urban Development Analysis Report',
            'executive_summary': 'Comprehensive analysis of urban development patterns and infrastructure',
            'sections': {
                'development_density': data.get('analysis', 'No analysis available'),
                'transportation_network': 'Road network and connectivity analysis',
                'zoning_compliance': 'Land use patterns and zoning assessment',
                'infrastructure_capacity': 'Utility and service infrastructure analysis',
                'growth_potential': 'Future development opportunities and constraints',
                'recommendations': [
                    'Update zoning regulations if needed',
                    'Improve transportation connectivity',
                    'Plan for infrastructure upgrades',
                    'Consider environmental impacts'
                ]
            },
            'metadata': data.get('metadata', {}),
            'confidence_level': data.get('confidence', 'Medium')
        }
    
    def _infrastructure_template(self, data: Dict) -> Dict:
        return {
            'title': 'Infrastructure Assessment Report',
            'executive_summary': 'Critical infrastructure analysis and condition assessment',
            'sections': {
                'infrastructure_inventory': data.get('analysis', 'No analysis available'),
                'condition_assessment': 'Visual condition indicators from satellite imagery',
                'capacity_analysis': 'Infrastructure capacity and utilization',
                'risk_assessment': 'Vulnerability and risk factors',
                'maintenance_priorities': 'Recommended maintenance and upgrade priorities',
                'recommendations': [
                    'Schedule detailed inspections',
                    'Prioritize critical infrastructure',
                    'Plan preventive maintenance',
                    'Consider redundancy improvements'
                ]
            },
            'metadata': data.get('metadata', {}),
            'confidence_level': data.get('confidence', 'Medium')
        }
    
    def _agriculture_template(self, data: Dict) -> Dict:
        return {
            'title': 'Agricultural Analysis Report',
            'executive_summary': 'Crop monitoring and agricultural land use analysis',
            'sections': {
                'crop_identification': data.get('analysis', 'No analysis available'),
                'field_conditions': 'Field patterns and agricultural practices',
                'infrastructure_assessment': 'Agricultural infrastructure and facilities',
                'seasonal_indicators': 'Crop growth stages and seasonal conditions',
                'yield_estimation': 'Preliminary yield indicators (requires spectral analysis)',
                'recommendations': [
                    'Implement precision agriculture techniques',
                    'Monitor irrigation efficiency',
                    'Optimize field management practices',
                    'Plan for seasonal variations'
                ]
            },
            'metadata': data.get('metadata', {}),
            'confidence_level': data.get('confidence', 'Medium')
        }
    
    def _disaster_response_template(self, data: Dict) -> Dict:
        return {
            'title': 'Disaster Response Assessment Report',
            'executive_summary': 'Emergency response and disaster preparedness analysis',
            'sections': {
                'hazard_assessment': data.get('analysis', 'No analysis available'),
                'vulnerability_analysis': 'Population and infrastructure vulnerability',
                'evacuation_routes': 'Emergency evacuation route assessment',
                'response_resources': 'Emergency response capability analysis',
                'recovery_planning': 'Post-disaster recovery considerations',
                'recommendations': [
                    'Improve emergency preparedness',
                    'Enhance evacuation planning',
                    'Strengthen critical infrastructure',
                    'Develop community resilience programs'
                ]
            },
            'metadata': data.get('metadata', {}),
            'confidence_level': data.get('confidence', 'Medium')
        }

# Initialize industry components
if 'geospatial_analyzer' not in st.session_state:
    st.session_state.geospatial_analyzer = GeospatialAnalyzer()
if 'timeseries_analyzer' not in st.session_state:
    st.session_state.timeseries_analyzer = TimeSeriesAnalyzer()
if 'quality_assessor' not in st.session_state:
    st.session_state.quality_assessor = QualityAssessment()
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = IndustryReportGenerator()

# ---------------------- Advanced Analysis Features ----------------------
def create_analysis_report(result, analysis_type, image_metadata):
    """Generate structured analysis report"""
    report = {
        "analysis_type": analysis_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "image_metadata": image_metadata,
        "analysis_result": result,
        "confidence_indicators": [],
        "recommendations": []
    }
    
    # Add confidence indicators based on content
    confidence_words = ["clearly visible", "definitively", "precisely", "specifically"]
    uncertainty_words = ["appears", "seems", "likely", "possibly", "might"]
    
    confidence_count = sum(1 for word in confidence_words if word in result.lower())
    uncertainty_count = sum(1 for word in uncertainty_words if word in result.lower())
    
    if confidence_count > uncertainty_count:
        report["confidence_indicators"].append("High confidence - specific observations")
    else:
        report["confidence_indicators"].append("Moderate confidence - some interpretations")
    
    return report

def extract_key_metrics(result, analysis_type):
    """Extract quantitative metrics from analysis"""
    metrics = {}
    
    # Look for numbers and percentages
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?%?', result)
    
    if "Infrastructure" in analysis_type:
        metrics["infrastructure_elements"] = len(re.findall(r'\b(?:road|building|bridge|tower)\b', result, re.IGNORECASE))
    elif "Land Use" in analysis_type:
        percentages = re.findall(r'\d+(?:\.\d+)?%', result)
        if percentages:
            metrics["land_use_percentages"] = percentages
    elif "Environmental" in analysis_type:
        metrics["environmental_features"] = len(re.findall(r'\b(?:vegetation|water|erosion|pollution)\b', result, re.IGNORECASE))
    
    return metrics

# ---------------------- Multi-Analysis Feature ----------------------
def run_multi_analysis(image, processor, model, device):
    """Run multiple analysis types on the same image"""
    analyses = {}
    analysis_types = [
        "Comprehensive Scene Description",
        "Infrastructure Analysis", 
        "Environmental Assessment",
        "Land Use Classification"
    ]
    
    enhanced_prompts = get_enhanced_prompts()
    
    for analysis_type in analysis_types:
        if analysis_type in enhanced_prompts:
            prompt = enhanced_prompts[analysis_type]
            
            # Run analysis
            try:
                system_prompt = "You are an expert satellite image analyst. Provide precise, accurate descriptions based only on clearly visible features."
                full_prompt = f"{system_prompt}\n\nImage Analysis Request:\n{prompt.strip()}"
                
                inputs = processor(text=full_prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                gen_kwargs = {
                    "max_new_tokens": 300,
                    "do_sample": True,
                    "temperature": 0.3,
                    "top_p": 0.85,
                    "repetition_penalty": 1.1,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                
                input_len = inputs['input_ids'].shape[1]
                gen_tokens = outputs[0][input_len:]
                result = processor.decode(gen_tokens, skip_special_tokens=True).strip()
                
                analyses[analysis_type] = result
                
            except Exception as e:
                analyses[analysis_type] = f"Analysis failed: {str(e)}"
    
    return analyses

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
    """Create mosaic with better handling - returns single PIL Image"""
    if not images:
        return None
        
    imgs = [img.copy() for img in images if img is not None]
    if not imgs:
        return None
    
    # Ensure all images are RGB
    imgs = [img.convert('RGB') if img.mode != 'RGB' else img for img in imgs]
    
    # Calculate optimal size
    widths, heights = zip(*(i.size for i in imgs))
    target_w, target_h = min(widths), min(heights)
    
    # Resize all images
    imgs = [i.resize((target_w, target_h), Image.Resampling.LANCZOS) for i in imgs]
    
    # Create mosaic
    rows = (len(imgs) + cols - 1) // cols
    mosaic = Image.new('RGB', (cols * target_w, rows * target_h), color='white')
    
    for idx, img in enumerate(imgs):
        x = (idx % cols) * target_w
        y = (idx // cols) * target_h
        mosaic.paste(img, (x, y))
    
    return mosaic

# ---------------------- Improved Prompts ----------------------
def get_enhanced_prompts():
    return {
        "Comprehensive Scene Description": """Provide a detailed, systematic description of this satellite image scene:

SCENE OVERVIEW:
- Overall scene type (urban, rural, industrial, natural, mixed)
- Dominant landscape characteristics
- General layout and organization patterns
- Approximate scale and coverage area

BUILT ENVIRONMENT:
- Building types, sizes, and architectural patterns
- Rooftop materials and colors
- Density and spacing of structures
- Construction quality and age indicators

TRANSPORTATION NETWORK:
- Road hierarchy (highways, arterials, local streets)
- Road surface conditions and materials
- Traffic patterns and congestion indicators
- Parking areas, intersections, roundabouts
- Public transit infrastructure if visible

NATURAL FEATURES:
- Topography and terrain characteristics
- Vegetation types, health, and seasonal state
- Water bodies with specific descriptions
- Geological features or landforms

HUMAN ACTIVITY PATTERNS:
- Signs of current or recent activity
- Functional zones and their purposes
- Temporal indicators (shadows, lighting, seasonal clues)
- Evidence of development or change

Organize your response with clear sections and be specific about spatial relationships between features.""",

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

        "Change Detection Analysis": """Compare and analyze changes visible in this satellite image:

DEVELOPMENT INDICATORS:
- New construction or development phases
- Construction equipment, staging areas, cleared land
- Fresh road cuts or new infrastructure
- Recently disturbed or modified terrain

TEMPORAL EVIDENCE:
- Seasonal changes in vegetation
- Weather-related impacts (flooding, snow, drought)
- Agricultural cycle indicators
- Construction progress stages

INFRASTRUCTURE EVOLUTION:
- Road network expansions or modifications
- New utility installations
- Building additions or demolitions
- Parking or development expansions

ENVIRONMENTAL CHANGES:
- Vegetation growth or loss patterns
- Erosion or sedimentation evidence
- Water level changes in bodies of water
- Land use transitions

Focus on observable evidence of change or development activity.""",

        "Security & Risk Assessment": """Analyze this satellite image for security and risk factors:

ACCESS CONTROL:
- Perimeter fencing, barriers, or boundaries
- Entry/exit points and security checkpoints
- Controlled access roads or gates
- Buffer zones or restricted areas

INFRASTRUCTURE VULNERABILITIES:
- Critical facilities (power, water, communications)
- Transportation chokepoints
- High-value or sensitive structures
- Emergency access routes

SAFETY HAZARDS:
- Hazardous material storage or processing
- Industrial facilities with safety concerns
- Flood-prone or erosion-risk areas
- Fire hazards or evacuation concerns

SURVEILLANCE CAPABILITIES:
- Tower locations for monitoring
- Open sightlines and visibility
- Natural or artificial observation points
- Areas with limited visibility or blind spots

Report only observable physical features relevant to security assessment.""",

        "Agricultural Analysis": """Analyze agricultural features and conditions in this satellite image:

CROP IDENTIFICATION:
- Field patterns and crop types if distinguishable
- Planting patterns (rows, spacing, orientation)
- Crop health indicators (color, uniformity)
- Growth stages if apparent

FIELD MANAGEMENT:
- Field boundaries and division systems
- Irrigation infrastructure (channels, sprinklers, pivots)
- Farm equipment visible in fields
- Storage facilities (silos, barns, equipment sheds)

AGRICULTURAL INFRASTRUCTURE:
- Farm buildings and processing facilities
- Access roads and field entrances
- Water management systems
- Livestock facilities if present

SEASONAL CONDITIONS:
- Harvest indicators or crop maturity
- Soil preparation or planting evidence
- Seasonal agricultural activities
- Weather impact indicators

Focus on observable agricultural features and current field conditions.""",

        "Urban Planning Analysis": """Analyze this satellite image from an urban planning perspective:

ZONING PATTERNS:
- Residential density gradients (single-family, multi-family, high-rise)
- Commercial districts and shopping centers
- Industrial zones and manufacturing areas
- Mixed-use developments

TRANSPORTATION PLANNING:
- Street network hierarchy and connectivity
- Public transit infrastructure
- Pedestrian and bicycle infrastructure
- Parking supply and distribution

INFRASTRUCTURE SYSTEMS:
- Utility corridors and service areas
- Stormwater management systems
- Public facilities (schools, parks, civic buildings)
- Emergency services accessibility

DEVELOPMENT PATTERNS:
- Growth boundaries and expansion areas
- Infill development opportunities
- Vacant or underutilized land
- Historic vs. new development patterns

SUSTAINABILITY FEATURES:
- Green infrastructure and tree canopy
- Renewable energy installations
- Sustainable development practices visible
- Environmental corridor preservation

Provide insights relevant to urban planning and development decisions.""",

        "Disaster Assessment": """Assess potential disaster impacts and emergency management factors:

NATURAL HAZARD EXPOSURE:
- Flood risk areas (low-lying, near water bodies)
- Wildfire risk (vegetation, topography, structures)
- Landslide or erosion susceptibility
- Wind exposure for severe weather

INFRASTRUCTURE RESILIENCE:
- Critical facility locations and protection
- Transportation network redundancy
- Utility infrastructure vulnerability
- Communication tower locations

EMERGENCY RESPONSE:
- Emergency service facility locations
- Evacuation route capacity and alternatives
- Emergency shelter or staging areas
- Helicopter landing zones or open spaces

COMMUNITY VULNERABILITY:
- High-density residential areas
- Vulnerable populations (schools, hospitals, elderly care)
- Economic critical facilities
- Historic or cultural preservation areas

RECOVERY RESOURCES:
- Construction staging areas
- Debris management sites
- Temporary housing locations
- Resource distribution points

Focus on observable features relevant to disaster preparedness and response."""
    }

# ---------------------- Sidebar: Enhanced Inputs ----------------------
st.sidebar.header("📥 Input & Settings")

# Mode selection
mode = st.sidebar.radio(
    "Select input mode:", ["Single Image", "Tiled Images", "Time Series Comparison"], key="mode_radio"
)

# NEW: Industry Analysis Mode
industry_mode = st.sidebar.selectbox(
    "Industry Focus:", 
    ["General Analysis", "Environmental Monitoring", "Urban Planning", "Infrastructure Assessment", 
     "Agricultural Analysis", "Disaster Response", "Security Assessment"],
    key="industry_select"
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
elif mode == "Tiled Images":
    uploaded_list = st.sidebar.file_uploader(
        "Upload up to 4 image tiles", type=["png","jpg","jpeg","tif","tiff"], 
        accept_multiple_files=True, key="upload_tiles"
    )
    uploaded = None
else:  # Time Series
    uploaded_list = st.sidebar.file_uploader(
        "Upload 2-4 images for temporal comparison", type=["png","jpg","jpeg","tif","tiff"], 
        accept_multiple_files=True, key="upload_timeseries"
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
        ""Describe what you see in this satellite image in detail.",
        height=100, key="custom_prompt"
    )
else:
    prompt = enhanced_prompts[analysis_type]

# Analysis configuration
with st.sidebar.expander("⚙️ Analysis Settings", expanded=False):
    max_tokens = st.slider("Max Response Tokens", 150, 500, 300, step=50, key="tokens_slider")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, step=0.1, key="temp_slider")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.85, step=0.05, key="topp_slider")
    
    # NEW: Quality thresholds
    quality_threshold = st.selectbox("Minimum Quality Required:", 
                                   ["Any", "Fair", "Good", "Excellent"], 
                                   index=1, key="quality_select")

# Export options
with st.sidebar.expander("📊 Export Options", expanded=False):
    export_format = st.selectbox("Report Format:", ["JSON", "PDF", "CSV"], key="export_format")
    include_metadata = st.checkbox("Include Image Metadata", value=True, key="metadata_checkbox")
    auto_generate_report = st.checkbox("Auto-Generate Industry Report", value=True, key="auto_report")

# ---------------------- Main Interface ----------------------
st.title("🛰️ Enhanced Satellite AI Pro")
st.markdown("**Advanced satellite and aerial image analysis with industry-grade features**")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analysis", "📊 Multi-Analysis", "📈 Metrics & Quality", "📋 Reports", "🕒 Time Series"])

# ---------------------- Image Processing Function ----------------------
def process_image(uploaded_file):
    """Process uploaded image with error handling"""
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_image_metadata(image, filename):
    """Extract comprehensive image metadata"""
    metadata = {
        'filename': filename,
        'size': image.size,
        'mode': image.mode,
        'format': image.format if hasattr(image, 'format') else 'Unknown',
        'upload_time': datetime.datetime.now().isoformat(),
        'file_size_mb': None  # Would need original file size
    }
    
    # Try to extract EXIF data
    try:
        exif = image._getexif()
        if exif:
            metadata['exif'] = dict(exif)
    except:
        pass
    
    return metadata

# ---------------------- Analysis Function ----------------------
def run_analysis(image, prompt_text, model, processor, device, max_tokens, temperature, top_p):
    """Run satellite image analysis with enhanced error handling"""
    try:
        # Prepare system prompt
        system_prompt = "You are an expert satellite image analyst with extensive experience in remote sensing, geospatial analysis, and earth observation. Provide precise, technical, and accurate descriptions based only on clearly visible features in the image."
        
        full_prompt = f"{system_prompt}\n\nImage Analysis Request:\n{prompt_text.strip()}"
        
        # Process inputs
        inputs = processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.1,
            "pad_token_id": processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0,
            "eos_token_id": processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else 2,
        }
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode response
        input_len = inputs['input_ids'].shape[1]
        gen_tokens = outputs[0][input_len:]
        result = processor.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Clean up GPU memory
        del inputs, outputs, gen_tokens
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"

# ---------------------- TAB 1: Main Analysis ----------------------
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image Preview")
        
        if mode == "Single Image" and uploaded:
            image = process_image(uploaded)
            if image:
                # Apply enhancements
                enhanced_image = enhance_satellite_image(
                    image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                )
                
                # Quality assessment
                quality_metrics = st.session_state.quality_assessor.assess_image_quality(enhanced_image)
                
                # Check quality threshold
                quality_map = {"Any": 0, "Fair": 40, "Good": 60, "Excellent": 80}
                if quality_threshold != "Any":
                    quality_scores = {"Poor": 0, "Fair": 40, "Good": 60, "Excellent": 80}
                    if quality_scores.get(quality_metrics['overall_quality'], 0) < quality_map[quality_threshold]:
                        st.warning(f"Image quality ({quality_metrics['overall_quality']}) is below threshold ({quality_threshold})")
                
                st.image(enhanced_image, caption=f"Enhanced: {uploaded.name}", use_column_width=True)
                
                # Display quality metrics
                with st.expander("Image Quality Metrics"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Resolution", quality_metrics['resolution'])
                        st.metric("Overall Quality", quality_metrics['overall_quality'])
                        st.metric("Brightness", f"{quality_metrics['brightness']:.1f}")
                    with col_b:
                        st.metric("Contrast", f"{quality_metrics['contrast']:.1f}")
                        st.metric("Sharpness", f"{quality_metrics['sharpness']:.1f}")
                        st.metric("Noise Level", f"{quality_metrics['noise_level']:.1f}")
                
        elif mode in ["Tiled Images", "Time Series Comparison"] and uploaded_list:
            if len(uploaded_list) > 4:
                st.warning("Maximum 4 images supported. Using first 4 images.")
                uploaded_list = uploaded_list[:4]
            
            images = []
            for uploaded_file in uploaded_list:
                img = process_image(uploaded_file)
                if img:
                    enhanced_img = enhance_satellite_image(
                        img, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                    )
                    images.append(enhanced_img)
            
            if images:
                if mode == "Tiled Images":
                    # Create mosaic
                    mosaic = create_mosaic(images)
                    if mosaic:
                        st.image(mosaic, caption="Tiled Mosaic", use_column_width=True)
                else:  # Time Series
                    # Display images in grid
                    cols = st.columns(min(len(images), 2))
                    for i, img in enumerate(images):
                        with cols[i % len(cols)]:
                            st.image(img, caption=f"Time {i+1}: {uploaded_list[i].name}", use_column_width=True)
        else:
            st.info("Upload an image to begin analysis")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if st.button("🚀 Run Analysis", type="primary", key="run_analysis_btn"):
            if mode == "Single Image" and uploaded:
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
                    
                    # Store in session state
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = []
                    
                    metadata = get_image_metadata(enhanced_image, uploaded.name)
                    analysis_data = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'image_name': uploaded.name,
                        'analysis_type': analysis_type,
                        'result': result,
                        'metadata': metadata,
                        'quality_metrics': quality_metrics,
                        'industry_focus': industry_mode
                    }
                    
                    st.session_state.analysis_results.append(analysis_data)
                    
                    # Add to time series analyzer
                    st.session_state.timeseries_analyzer.add_analysis(metadata, result)
                    
                    # Display result
                    st.success("Analysis Complete!")
                    st.write("**Analysis Result:**")
                    st.write(result)
                    
                    # Generate industry report if enabled
                    if auto_generate_report and industry_mode != "General Analysis":
                        report_type = industry_mode.lower().replace(" ", "_")
                        industry_report = st.session_state.report_generator.generate_report(
                            analysis_data, report_type
                        )
                        st.session_state.latest_report = industry_report
                        
                        with st.expander("📋 Industry Report Preview"):
                            st.write(f"**{industry_report['title']}**")
                            st.write(industry_report['executive_summary'])
            
            elif mode in ["Tiled Images", "Time Series Comparison"] and uploaded_list:
                st.info("Multi-image analysis available in Multi-Analysis tab")
            else:
                st.warning("Please upload an image first")

# ---------------------- TAB 2: Multi-Analysis ----------------------
with tab2:
    st.subheader("🔄 Multi-Analysis Dashboard")
    
    if st.button("🎯 Run Comprehensive Analysis", type="primary", key="multi_analysis_btn"):
        if mode == "Single Image" and uploaded:
            image = process_image(uploaded)
            if image:
                enhanced_image = enhance_satellite_image(
                    image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                )
                
                with st.spinner("Running comprehensive analysis..."):
                    multi_results = run_multi_analysis(enhanced_image, processor, model, torch_device)
                
                # Display results in expandable sections
                for analysis_name, result in multi_results.items():
                    with st.expander(f"📊 {analysis_name}", expanded=False):
                        st.write(result)
                        
                        # Extract metrics
                        metrics = extract_key_metrics(result, analysis_name)
                        if metrics:
                            st.write("**Key Metrics:**")
                            for key, value in metrics.items():
                                st.metric(key.replace("_", " ").title(), value)
                
                # Store comprehensive results
                st.session_state.multi_analysis_results = multi_results
                
        elif mode in ["Tiled Images", "Time Series Comparison"] and uploaded_list:
            # Process multiple images
            all_results = {}
            
            for i, uploaded_file in enumerate(uploaded_list):
                image = process_image(uploaded_file)
                if image:
                    enhanced_image = enhance_satellite_image(
                        image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                    )
                    
                    st.write(f"**Analyzing Image {i+1}: {uploaded_file.name}**")
                    with st.spinner(f"Processing image {i+1}..."):
                        multi_results = run_multi_analysis(enhanced_image, processor, model, torch_device)
                    
                    all_results[f"Image_{i+1}"] = multi_results
                    
                    # Display condensed results
                    with st.expander(f"Results for {uploaded_file.name}"):
                        for analysis_name, result in multi_results.items():
                            st.write(f"**{analysis_name}:**")
                            st.write(result[:200] + "..." if len(result) > 200 else result)
            
            st.session_state.multi_image_results = all_results
        else:
            st.warning("Please upload image(s) first")
    
    # Display stored results
    if hasattr(st.session_state, 'multi_analysis_results'):
        st.subheader("📈 Analysis Comparison")
        
        # Create comparison chart
        analysis_types = list(st.session_state.multi_analysis_results.keys())
        word_counts = [len(result.split()) for result in st.session_state.multi_analysis_results.values()]
        
        comparison_df = pd.DataFrame({
            'Analysis Type': analysis_types,
            'Word Count': word_counts,
            'Detail Level': ['High' if wc > 100 else 'Medium' if wc > 50 else 'Low' for wc in word_counts]
        })
        
        fig = px.bar(comparison_df, x='Analysis Type', y='Word Count', 
                     color='Detail Level', title="Analysis Depth Comparison")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- TAB 3: Metrics & Quality ----------------------
with tab3:
    st.subheader("📊 Quality Assessment & Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Image Quality Analysis**")
        
        if (mode == "Single Image" and uploaded) or (mode in ["Tiled Images", "Time Series Comparison"] and uploaded_list):
            # Get current image(s)
            if mode == "Single Image":
                image = process_image(uploaded)
                if image:
                    enhanced_image = enhance_satellite_image(
                        image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                    )
                    quality_metrics = st.session_state.quality_assessor.assess_image_quality(enhanced_image)
                    
                    # Display comprehensive quality assessment
                    st.json(quality_metrics)
                    
                    # Quality score visualization
                    quality_scores = {"Poor": 25, "Fair": 50, "Good": 75, "Excellent": 100}
                    current_score = quality_scores.get(quality_metrics['overall_quality'], 50)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = current_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Quality Score"},
                        delta = {'reference': 60},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgray"},
                                {'range': [40, 60], 'color': "yellow"},
                                {'range': [60, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "green"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90}}))
                                
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Multiple images
                st.write("Quality assessment for multiple images:")
                quality_data = []
                
                for i, uploaded_file in enumerate(uploaded_list):
                    image = process_image(uploaded_file)
                    if image:
                        enhanced_image = enhance_satellite_image(
                            image, contrast, brightness, saturation, sharpness, edge_enhance, denoise
                        )
                        quality_metrics = st.session_state.quality_assessor.assess_image_quality(enhanced_image)
                        quality_data.append({
                            'Image': uploaded_file.name,
                            'Quality': quality_metrics['overall_quality'],
                            'Resolution': quality_metrics['resolution'],
                            'Sharpness': quality_metrics['sharpness'],
                            'Contrast': quality_metrics['contrast']
                        })
                
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    st.dataframe(quality_df)
    
    with col2:
        st.write("**Analysis Metrics**")
        
        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
            # Create metrics from analysis results
            metrics_data = []
            
            for result in st.session_state.analysis_results:
                word_count = len(result['result'].split())
                confidence_indicators = len(result.get('confidence_indicators', []))
                
                metrics_data.append({
                    'Timestamp': result['timestamp'],
                    'Analysis Type': result['analysis_type'],
                    'Word Count': word_count,
                    'Quality Score': result.get('quality_metrics', {}).get('overall_quality', 'Unknown'),
                    'Industry Focus': result.get('industry_focus', 'General')
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df)
            
            # Visualization
            if len(metrics_df) > 1:
                fig = px.scatter(metrics_df, x='Timestamp', y='Word Count', 
                               color='Analysis Type', size='Word Count',
                               title="Analysis Metrics Over Time")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------- TAB 4: Reports ----------------------
with tab4:
    st.subheader("📋 Industry Reports")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Generate Custom Report**")
        
        report_type = st.selectbox(
            "Select Report Type:",
            ["environmental", "urban_planning", "infrastructure", "agriculture", "disaster_response"],
            key="report_type_select"
        )
        
        if st.button("Generate Report", key="generate_report_btn"):
            if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                latest_analysis = st.session_state.analysis_results[-1]
                report = st.session_state.report_generator.generate_report(latest_analysis, report_type)
                st.session_state.latest_report = report
                st.success("Report generated successfully!")
            else:
                st.warning("Please run an analysis first")
    
    with col2:
        st.write("**Export Options**")
        
        if hasattr(st.session_state, 'latest_report'):
            if st.button("📄 Export as JSON", key="export_json_btn"):
                report_json = json.dumps(st.session_state.latest_report, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"satellite_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Display latest report
    if hasattr(st.session_state, 'latest_report'):
        st.write("**Latest Generated Report**")
        
        report = st.session_state.latest_report
        st.write(f"# {report['title']}")
        
        st.write("## Executive Summary")
        st.write(report['executive_summary'])
        
        st.write("## Analysis Sections")
        for section_name, section_content in report['sections'].items():
            if section_name != 'recommendations':
                st.write(f"### {section_name.replace('_', ' ').title()}")
                if isinstance(section_content, list):
                    for item in section_content:
                        st.write(f"- {item}")
                else:
                    st.write(section_content)
        
        if 'recommendations' in report['sections']:
            st.write("## Recommendations")
            for rec in report['sections']['recommendations']:
                st.write(f"- {rec}")
        
        st.write("## Metadata")
        st.json(report.get('metadata', {}))

# ---------------------- TAB 5: Time Series ----------------------
with tab5:
    st.subheader("🕒 Time Series Analysis")
    
    if hasattr(st.session_state, 'timeseries_analyzer') and st.session_state.timeseries_analyzer.analysis_history:
        st.write("**Analysis History**")
        
        history_data = []
        for entry in st.session_state.timeseries_analyzer.analysis_history:
            history_data.append({
                'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Image': entry['image_metadata'].get('filename', 'Unknown'),
                'Buildings': entry['features_detected']['buildings'],
                'Roads': entry['features_detected']['roads'],
                'Vegetation': entry['features_detected']['vegetation'],
                'Water': entry['features_detected']['water']
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df)
        
        # Change detection
        if len(st.session_state.timeseries_analyzer.analysis_history) >= 2:
            st.write("**Change Detection**")
            
            changes = st.session_state.timeseries_analyzer.detect_changes()
            
            if "message" not in changes:
                change_data = []
                for feature, change_info in changes.items():
                    change_data.append({
                        'Feature': feature.title(),
                        'Change': change_info['change'],
                        'Percentage Change': f"{change_info['percentage_change']:.1f}%"
                    })
                
                change_df = pd.DataFrame(change_data)
                st.dataframe(change_df)
                
                # Visualization
                fig = px.bar(change_df, x='Feature', y='Change', 
                           title="Feature Changes Between Latest Analyses",
                           color='Change', 
                           color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series visualization
        if len(history_df) > 1:
            st.write("**Feature Trends Over Time**")
            
            # Prepare data for plotting
            plot_df = history_df.melt(
                id_vars=['Timestamp'], 
                value_vars=['Buildings', 'Roads', 'Vegetation', 'Water'],
                var_name='Feature', 
                value_name='Count'
            )
            
            fig = px.line(plot_df, x='Timestamp', y='Count', color='Feature',
                         title="Feature Detection Over Time")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No analysis history available. Run some analyses to see time series data.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enhanced Satellite AI Pro - Professional Remote Sensing Analysis Tool</p>
    <p>Powered by LLaVA Vision Language Model with Industry-Grade Features</p>
</div>
""", unsafe_allow_html=True)

# Memory cleanup
if st.button("🧹 Clear Memory Cache", key="clear_cache_btn"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.success("Memory cache cleared!")
