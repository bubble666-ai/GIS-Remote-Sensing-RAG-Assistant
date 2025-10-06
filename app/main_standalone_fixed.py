"""
Fixed and Enhanced Standalone Streamlit app for GIS & Remote Sensing RAG Assistant.
Better styling, multiple XAI models, and improved UX.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="GIS & Remote Sensing Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with darker colors and better contrast
st.markdown("""
<style>
    /* Main chat message styling */
    .chat-container {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 1rem;
    }

    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
        transition: all 0.2s ease;
    }

    .chat-message:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* User message styling - darker blue */
    .user-message {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        border-left: 5px solid #3b82f6;
        margin-left: 2rem;
        position: relative;
    }

    .user-message::before {
        content: "👤";
        position: absolute;
        top: -8px;
        left: -20px;
        background: #3b82f6;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Assistant message styling - darker green/teal instead of pink */
    .assistant-message {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: white;
        border-left: 5px solid #10b981;
        margin-right: 2rem;
        position: relative;
    }

    .assistant-message::before {
        content: "🤖";
        position: absolute;
        top: -8px;
        right: -20px;
        background: #10b981;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Source info styling */
    .source-info {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
    }

    .source-info strong {
        color: #fbbf24;
    }

    /* Timestamp styling */
    .timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        color: rgba(255, 255, 255, 0.8);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        color: white;
    }

    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
        padding: 0.8rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: #10b981;
    }

    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
        padding: 0.6rem;
    }

    /* Success/error message styling */
    .stSuccess {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
    }

    .stError {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        border-radius: 8px;
    }

    .stWarning {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        border-radius: 8px;
    }

    /* Chat input styling */
    .stChatInput {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 0.8rem;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            margin-left: 0.5rem;
            margin-right: 0.5rem;
        }

        .chat-message {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


class EnhancedXAIProvider:
    """Enhanced XAI provider with multiple model support."""

    def __init__(self, api_key: str, model_name: str = "grok-3"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available XAI models with their specifications."""
        return [
            {
                "name": "grok-3",
                "display_name": "Grok 3 (Latest)",
                "context_length": 131072,
                "description": "Most capable model for complex reasoning",
                "input_cost": 600,
                "output_cost": 600
            },
            {
                "name": "grok-4-fast-reasoning",
                "display_name": "Grok 4 Fast Reasoning",
                "context_length": 2000000,
                "description": "Advanced reasoning with 2M context",
                "input_cost": 480,
                "output_cost": 480
            },
            {
                "name": "grok-4-fast-non-reasoning",
                "display_name": "Grok 4 Fast (Non-Reasoning)",
                "context_length": 2000000,
                "description": "Fast responses with 2M context",
                "input_cost": 480,
                "output_cost": 480
            },
            {
                "name": "grok-4-0709",
                "display_name": "Grok 4 (Legacy)",
                "context_length": 256000,
                "description": "Previous version of Grok 4",
                "input_cost": 480,
                "output_cost": 480
            },
            {
                "name": "grok-3-mini",
                "display_name": "Grok 3 Mini",
                "context_length": 131072,
                "description": "Smaller, faster version",
                "input_cost": 480,
                "output_cost": 480
            },
            {
                "name": "grok-code-fast-1",
                "display_name": "Grok Code Fast",
                "context_length": 256000,
                "description": "Specialized for code generation",
                "input_cost": 480,
                "output_cost": 480
            },
            {
                "name": "grok-2-vision-1212us-east-1",
                "display_name": "Grok 2 Vision (US East)",
                "context_length": 32768,
                "description": "Multimodal model (US region)",
                "input_cost": 600,
                "output_cost": 600
            },
            {
                "name": "grok-2-vision-1212eu-west-1",
                "display_name": "Grok 2 Vision (EU West)",
                "context_length": 32768,
                "description": "Multimodal model (EU region)",
                "input_cost": 50,
                "output_cost": 50
            }
        ]

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        available_models = [m["name"] for m in self.get_available_models()]
        if model_name in available_models:
            self.model_name = model_name
            return True
        return False

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generate response using selected XAI model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"


class EnhancedGISRetriever:
    """Enhanced document retriever with better content organization."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load enhanced document collection."""
        documents = [
            {
                "content": """
# NDVI (Normalized Difference Vegetation Index) Analysis Guide

## Overview
NDVI is a standardized index allowing you to generate an image displaying greenness (relative biomass).
This index takes advantage of the contrast of the characteristics of two bands from a multispectral raster dataset —
the chlorophyll pigment absorptions in the red band and the high reflectivity of plant materials in the near-infrared (NIR) band.

## Mathematical Formula
NDVI = (NIR - Red) / (NIR + Red)

## Interpretation Scale
- **-1.0 to 0.0**: Non-vegetated surfaces (water, snow, built-up areas)
- **0.0 to 0.3**: Sparse vegetation
- **0.3 to 0.6**: Moderate vegetation
- **0.6 to 1.0**: Dense, healthy vegetation

## Satellite-Specific Calculations

### Sentinel-2 NDVI Calculation
- **Red Band**: Band 4 (10m resolution)
- **NIR Band**: Band 8 (10m resolution)

### Landsat 8 NDVI Calculation
- **Red Band**: Band 4 (30m resolution)
- **NIR Band**: Band 5 (30m resolution)

## Applications
1. Crop health monitoring
2. Drought assessment
3. Vegetation mapping
4. Land cover classification
5. Environmental change detection

## Best Practices
- Perform atmospheric correction before calculation
- Use quality assessment bands to filter poor quality data
- Consider multi-temporal analysis for trend detection
                """,
                "source": "ndvi_analysis.txt",
                "page": 1,
                "category": "Vegetation Analysis",
                "tags": ["NDVI", "vegetation", "satellite", "calculation"]
            },
            {
                "content": """
# Sentinel Satellite Missions Overview

## Sentinel-2: Multispectral Optical Imaging

### Mission Specifications
- **Mission**: Multispectral optical imaging
- **Constellation**: Two satellites (Sentinel-2A, Sentinel-2B)
- **Revisit Time**: 5 days (dual constellation)
- **Swath Width**: 290km
- **Altitude**: 786km

### Spectral Capabilities
- **Spatial Resolution**: 10m, 20m, and 60m
- **Spectral Bands**: 13 bands (443-2190nm)

### Key Bands for Vegetation Analysis
- **Band 4 (Red)**: 665nm, 10m resolution
- **Band 8 (NIR)**: 842nm, 10m resolution
- **Band 8A (Red Edge)**: 865nm, 20m resolution

### Applications
- Land cover mapping
- Vegetation monitoring
- Agriculture management
- Forest monitoring
- Water quality assessment
- Urban planning

### Data Access
- Free and open data access
- Multiple processing levels available
- Global coverage with systematic acquisition
                """,
                "source": "sentinel_overview.txt",
                "page": 1,
                "category": "Satellite Platforms",
                "tags": ["Sentinel-2", "multispectral", "ESA", "Copernicus"]
            },
            {
                "content": """
# Landsat 8/9 Specifications Handbook

## Mission Details
- **Launch**: Landsat 8 (2013), Landsat 9 (2021)
- **Orbit**: Sun-synchronous, 705km altitude
- **Revisit Time**: 16 days
- **Equatorial Crossing**: 10:11 AM (descending)
- **Swath Width**: 185km

## OLI (Operational Land Imager) Specifications

### Sensor Bands and Resolutions
1. **Band 1 (Coastal/Aerosol)**: 433-453nm, 30m resolution
2. **Band 2 (Blue)**: 450-515nm, 30m resolution
3. **Band 3 (Green)**: 525-600nm, 30m resolution
4. **Band 4 (Red)**: 630-680nm, 30m resolution
5. **Band 5 (NIR)**: 845-885nm, 30m resolution
6. **Band 6 (SWIR 1)**: 1560-1660nm, 30m resolution
7. **Band 7 (SWIR 2)**: 2100-2300nm, 30m resolution
8. **Band 8 (Panchromatic)**: 500-680nm, 15m resolution
9. **Band 9 (Cirrus)**: 1360-1390nm, 30m resolution

### TIRS (Thermal Infrared Sensor)
- **Band 10 (TIRS 1)**: 10.6-11.2μm, 100m resolution
- **Band 11 (TIRS 2)**: 11.5-12.5μm, 100m resolution

## Key Features
- 12-bit radiometric resolution
- Improved signal-to-noise ratio
- Better geometric accuracy
- Enhanced calibration

## Major Applications
1. Agricultural monitoring and yield prediction
2. Forest management and deforestation tracking
3. Water resource management
4. Urban planning and development monitoring
5. Disaster assessment and response
6. Climate change research
7. Geologic mapping and mineral exploration
                """,
                "source": "landsat_handbook.txt",
                "page": 1,
                "category": "Satellite Platforms",
                "tags": ["Landsat", "USGS", "thermal", "multispectral"]
            },
            {
                "content": """
# Advanced Remote Sensing Techniques

## Classification Methods

### Supervised Classification
- **Definition**: Uses training data to classify pixels
- **Process**:
  1. Select training samples
  2. Train classification algorithm
  3. Apply to entire image
- **Common Algorithms**: Maximum Likelihood, Support Vector Machines, Random Forest
- **Advantages**: High accuracy when training data is good
- **Disadvantages**: Requires extensive training data

### Unsupervised Classification
- **Definition**: Automatically groups pixels into classes
- **Process**:
  1. Algorithm identifies spectral clusters
  2. Assigns pixels to clusters
  3. User assigns meaning to clusters
- **Common Algorithms**: K-means, ISODATA
- **Advantages**: No training data required
- **Disadvantages**: Classes may not be meaningful

## Atmospheric Correction

### Why Atmospheric Correction is Needed
- Atmospheric scattering affects surface reflectance
- Path radiance adds noise to the signal
- Different atmospheric conditions impact measurements

### Common Methods
1. **Dark Object Subtraction (DOS)**: Simple method for basic correction
2. **Radiative Transfer Models**: More accurate but complex
3. **FLAASH**: Atmospheric correction for multispectral imagery
4. **Sen2Cor**: Specific correction for Sentinel-2 data

## Change Detection Techniques

### Methods
1. **Image Differencing**: Subtract one image from another
2. **Change Vector Analysis**: Multi-dimensional change detection
3. **Classification Comparison**: Compare classifications from different times
4. **Temporal Trajectory Analysis**: Track changes over time

### Applications
- Urban growth monitoring
- Deforestation tracking
- Agricultural change detection
- Disaster impact assessment
                """,
                "source": "advanced_techniques.txt",
                "page": 1,
                "category": "Image Processing",
                "tags": ["classification", "atmospheric", "change detection"]
            }
        ]

        return documents

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Enhanced retrieval with better scoring."""
        query_words = [word.lower().strip() for word in query.split() if len(word) > 2]
        scored_docs = []

        for doc in self.documents:
            content_lower = doc["content"].lower()

            # Enhanced scoring: title matches, exact phrase matches, keyword matches
            score = 0

            # Check for exact phrase matches
            exact_phrases = [f" {word} " for word in query_words if len(word) > 3]
            for phrase in exact_phrases:
                score += content_lower.count(phrase) * 3

            # Check for individual word matches
            for word in query_words:
                score += content_lower.count(word)

            # Bonus for tag matches
            for tag in doc.get("tags", []):
                if tag.lower() in query_words:
                    score += 5

            # Bonus for category matches
            if doc.get("category", "").lower() in query.lower():
                score += 3

            if score > 0:
                scored_docs.append({
                    "content": doc["content"],
                    "metadata": {
                        "source": doc["source"],
                        "page": doc["page"],
                        "category": doc["category"],
                        "tags": doc.get("tags", [])
                    },
                    "score": score
                })

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


def initialize_session_state():
    """Initialize enhanced session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'xai_provider' not in st.session_state:
        api_key = os.getenv("XAI_API_KEY")
        if api_key:
            st.session_state.xai_provider = EnhancedXAIProvider(api_key)
        else:
            st.session_state.xai_provider = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = EnhancedGISRetriever()
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "grok-3"
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}


def render_sidebar():
    """Render enhanced sidebar with model switching."""
    st.sidebar.title("🌍 GIS Assistant Configuration")

    # XAI Model Selection with detailed info
    if st.session_state.xai_provider:
        st.sidebar.subheader("🤖 XAI Model Selection")

        # Get available models
        available_models = st.session_state.xai_provider.get_available_models()
        model_options = [(model["display_name"], model["name"]) for model in available_models]

        # Create selection
        selected_display_name = st.sidebar.selectbox(
            "Choose Model",
            options=[name for name, _ in model_options],
            index=0,
            help="Select which XAI model to use for responses"
        )

        # Find the selected model's internal name
        selected_model_name = None
        for display_name, internal_name in model_options:
            if display_name == selected_display_name:
                selected_model_name = internal_name
                break

        # Switch model if different
        if selected_model_name and selected_model_name != st.session_state.selected_model:
            if st.session_state.xai_provider.switch_model(selected_model_name):
                st.session_state.selected_model = selected_model_name
                st.sidebar.success(f"Switched to {selected_display_name}")
                st.rerun()

        # Display model information
        selected_model_info = next((m for m in available_models if m["name"] == st.session_state.selected_model), None)
        if selected_model_info:
            with st.sidebar.expander("Model Information", expanded=False):
                st.markdown(f"**Context Length:** {selected_model_info['context_length']:,} tokens")
                st.markdown(f"**Description:** {selected_model_info['description']}")
                st.markdown(f"**Input Cost:** ${selected_model_info['input_cost']/1_000_000:.2f}/M tokens")
                st.markdown(f"**Output Cost:** ${selected_model_info['output_cost']/1_000_000:.2f}/M tokens")
    else:
        st.sidebar.error("❌ XAI API key not found")
        st.sidebar.info("Please add XAI_API_KEY to .env file")

    # Model Settings
    st.sidebar.subheader("⚙️ Model Settings")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Controls randomness in responses (0 = deterministic, 1 = creative)"
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1500,
        step=100,
        help="Maximum number of tokens in the response"
    )

    # Retrieval Settings
    st.sidebar.subheader("📚 Document Retrieval")
    top_k = st.sidebar.slider(
        "Top K Documents",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of relevant documents to retrieve"
    )

    # Clear chat button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History", help="Clear all conversation history"):
        st.session_state.chat_history = []
        st.rerun()

    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k
    }


def render_chat_interface(config: Dict[str, Any]):
    """Render enhanced chat interface with better styling."""

    # Header with model info
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("## 🌍 GIS & Remote Sensing Assistant")
        st.markdown("*Powered by XAI Grok with intelligent document retrieval*")

    with col2:
        if st.session_state.selected_model:
            st.metric("Current Model", st.session_state.selected_model.split('-')[0].upper())

    with col3:
        doc_count = len(st.session_state.retriever.documents)
        st.metric("Documents", doc_count)

    # Check if provider is available
    if not st.session_state.xai_provider:
        st.error("❌ XAI provider not available. Please check your API key configuration.")
        st.info("Add your XAI API key to the `.env` file as `XAI_API_KEY=your_key_here`")
        return

    # Enhanced chat container
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display chat history with enhanced styling
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-content">{message['content']}</div>
                    <div class="timestamp">{message.get('timestamp', '')[:19]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = render_sources(message.get('sources', []))
                timestamp = message.get('timestamp', '')[:19]

                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-content">{message['content']}</div>
                    {sources_html}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced chat input
    st.markdown("---")

    # Sample questions to help users
    with st.expander("💡 Sample Questions You Can Ask", expanded=False):
        sample_questions = [
            "What is NDVI and how is it calculated from satellite imagery?",
            "Which bands does Sentinel-2 use for NDVI calculation?",
            "What's the spatial resolution difference between Sentinel-2 and Landsat-8?",
            "Explain the difference between supervised and unsupervised classification",
            "How does atmospheric correction work in remote sensing?",
            "What are the applications of LiDAR in forestry?",
            "Compare the spectral capabilities of Landsat 8 and Sentinel-2",
            "What is change detection and how is it performed?"
        ]

        for i, question in enumerate(sample_questions, 1):
            if st.button(f"{i}. {question}", key=f"sample_{i}"):
                st.session_state.sample_question = question
                st.rerun()

    # Handle sample question clicks
    if 'sample_question' in st.session_state:
        user_input = st.session_state.sample_question
        del st.session_state.sample_question
        # Process the sample question
        process_user_input(user_input, config)
        return

    # Main chat input
    user_input = st.chat_input(
        "Ask your GIS/Remote Sensing question here...",
        key="main_chat_input"
    )

    if user_input:
        process_user_input(user_input, config)


def process_user_input(user_input: str, config: Dict[str, Any]):
    """Process user input and generate response."""
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })

    # Generate response
    with st.spinner("🔍 Searching documents and generating response..."):
        try:
            start_time = time.time()

            # Retrieve relevant documents
            retriever = st.session_state.retriever
            retrieved_docs = retriever.retrieve(user_input, top_k=config['top_k'])

            # Format context with better structure
            context = ""
            if retrieved_docs:
                context = "**Relevant Documents:**\n\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    source_info = f"Source: {doc['metadata']['source']} (Category: {doc['metadata'].get('category', 'Unknown')})"
                    context += f"**Document {i}:** {source_info}\n"
                    context += f"{doc['content'][:800]}...\n\n"
            else:
                context = "**No specific documents found. Answering based on general knowledge.**\n\n"

            # Create enhanced messages for Grok
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert GIS and Remote Sensing specialist with deep knowledge of:
                    - Satellite imagery and remote sensing platforms (Landsat, Sentinel, MODIS, etc.)
                    - Spectral indices and vegetation analysis (NDVI, EVI, SAVI, etc.)
                    - Image processing and atmospheric correction techniques
                    - GIS data analysis and spatial statistics
                    - Cartography and map projections
                    - Geographic coordinate systems and transformations
                    - Land use/land cover classification
                    - Change detection and time series analysis
                    - Digital elevation models and terrain analysis
                    - GIS software and tools (ArcGIS, QGIS, ENVI, etc.)

                    **Instructions:**
                    1. Use the provided context documents when they are relevant
                    2. Supplement with your expert knowledge when needed
                    3. Provide comprehensive, technically accurate answers
                    4. Include specific details, formulas, or methodologies when relevant
                    5. Use proper GIS terminology
                    6. If information is from general knowledge, clearly indicate that
                    7. Structure your response clearly with headings and bullet points when appropriate"""
                },
                {
                    "role": "user",
                    "content": f"""{context}

**Question:** {user_input}

Please provide a comprehensive, technically accurate answer with relevant details and examples."""
                }
            ]

            # Generate response
            provider = st.session_state.xai_provider
            response = provider.generate_response(messages, config['max_tokens'])

            generation_time = time.time() - start_time

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': retrieved_docs,
                'timestamp': datetime.now().isoformat(),
                'model_used': st.session_state.selected_model,
                'generation_time': generation_time
            })

            # Rerun to display the response
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            logger.error(f"Response generation error: {str(e)}")


def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Render enhanced source citations."""
    if not sources:
        return ""

    sources_html = '<div class="source-info"><strong>📚 Sources Used:</strong><ul>'
    for source in sources:
        source_name = source['metadata'].get('source', 'Unknown')
        category = source['metadata'].get('category', 'Unknown')
        page = source['metadata'].get('page', 'N/A')
        score = source.get('score', 0)
        tags = source['metadata'].get('tags', [])
        tags_str = f" [{', '.join(tags)}]" if tags else ""

        sources_html += f'<li><strong>{source_name}</strong> - {category}{tags_str} [Relevance: {score}]</li>'
    sources_html += '</ul></div>'

    return sources_html


def render_footer():
    """Render footer with additional information."""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🚀 Features")
        st.markdown("- ✅ Multiple XAI models")
        st.markdown("- ✅ Document retrieval")
        st.markdown("- ✅ Source citations")
        st.markdown("- ✅ Enhanced UI")

    with col2:
        st.markdown("### 📊 Current Session")
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        assistant_messages = total_messages - user_messages

        st.metric("Total Messages", total_messages)
        st.metric("Your Questions", user_messages)
        st.metric("AI Responses", assistant_messages)

    with col3:
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [XAI Models](https://console.x.ai/)")
        st.markdown("- [Sentinel Hub](https://scihub.copernicus.eu/)")
        st.markdown("- [USGS EarthExplorer](https://earthexplorer.usgs.gov/)")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🌍 GIS & Remote Sensing Assistant | Powered by XAI Grok | "
        f"Current Model: <strong>{st.session_state.selected_model.split('-')[0].upper()}</strong>"
        "</div>",
        unsafe_allow_html=True
    )


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Render sidebar
    config = render_sidebar()

    # Main content area
    render_chat_interface(config)

    # Footer
    render_footer()


if __name__ == "__main__":
    main()