"""
Standalone Streamlit app for GIS & Remote Sensing RAG Assistant with XAI Grok.
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

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .source-info {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


class SimpleXAIProvider:
    """Simple XAI provider for demonstration."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generate response using XAI Grok."""
        try:
            response = self.client.chat.completions.create(
                model="grok-3",
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"


class SimpleGISRetriever:
    """Simple document retriever for demonstration."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from data directory."""
        documents = []

        # Load sample documents
        sample_docs = [
            {
                "content": """
                NDVI (Normalized Difference Vegetation Index) Analysis Guide

                Overview:
                NDVI is a standardized index allowing you to generate an image displaying greenness (relative biomass).
                This index takes advantage of the contrast of the characteristics of two bands from a multispectral raster dataset —
                the chlorophyll pigment absorptions in the red band and the high reflectivity of plant materials in the near-infrared (NIR) band.

                Mathematical Formula:
                NDVI = (NIR - Red) / (NIR + Red)

                Sentinel-2 NDVI Calculation:
                - Red Band: Band 4 (10m resolution)
                - NIR Band: Band 8 (10m resolution)

                Landsat 8 NDVI Calculation:
                - Red Band: Band 4 (30m resolution)
                - NIR Band: Band 5 (30m resolution)
                """,
                "source": "ndvi_analysis.txt",
                "page": 1
            },
            {
                "content": """
                Sentinel Satellite Missions Overview

                Sentinel-2: Multispectral Imaging
                - Mission: Multispectral optical imaging
                - Constellation: Two satellites (Sentinel-2A, Sentinel-2B)
                - Revisit Time: 5 days (dual constellation)
                - Spatial Resolution: 10m, 20m, and 60m
                - Spectral Bands: 13 bands (443-2190nm)
                - Key Bands for Vegetation:
                  * Band 4 (Red): 665nm, 10m resolution
                  * Band 8 (NIR): 842nm, 10m resolution
                """,
                "source": "sentinel_overview.txt",
                "page": 1
            },
            {
                "content": """
                Landsat Program Specifications

                Landsat 8 Specifications:
                - Launch: 2013
                - Orbit: Sun-synchronous, 705km altitude
                - Revisit Time: 16 days
                - Swath Width: 185km

                OLI Bands and Resolutions:
                - Band 4 (Red): 630-680nm, 30m resolution
                - Band 5 (NIR): 845-885nm, 30m resolution
                - Band 8 (Panchromatic): 500-680nm, 15m resolution
                """,
                "source": "landsat_handbook.txt",
                "page": 1
            }
        ]

        return sample_docs

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval."""
        # Simple keyword matching for demonstration
        query_words = query.lower().split()
        scored_docs = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_docs.append({
                    "content": doc["content"],
                    "metadata": {
                        "source": doc["source"],
                        "page": doc["page"]
                    },
                    "score": score
                })

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'xai_provider' not in st.session_state:
        api_key = os.getenv("XAI_API_KEY")
        if api_key:
            st.session_state.xai_provider = SimpleXAIProvider(api_key)
        else:
            st.session_state.xai_provider = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = SimpleGISRetriever()


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Configuration")

    # Check XAI API key
    api_key = os.getenv("XAI_API_KEY")
    if api_key:
        st.sidebar.success("XAI API key configured")
        st.sidebar.info("Using Grok-3 model")
    else:
        st.sidebar.error("XAI API key not found")
        st.sidebar.info("Please add XAI_API_KEY to .env file")

    # Model Settings
    st.sidebar.subheader("Model Settings")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100
    )

    # Retrieval Settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top K Documents",
        min_value=1,
        max_value=3,
        value=2,
        step=1
    )

    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k
    }


def render_chat_interface(config: Dict[str, Any]):
    """Render the main chat interface."""
    st.title("GIS & Remote Sensing Assistant")
    st.markdown("Powered by XAI Grok-3 with document retrieval")

    # Check if provider is available
    if not st.session_state.xai_provider:
        st.error("XAI provider not available. Please check your API key.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant:</strong><br>
                {message['content']}
                {render_sources(message.get('sources', []))}
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask your GIS/Remote Sensing question here...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Generate response
        with st.spinner("Searching documents and generating response..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.retriever
                retrieved_docs = retriever.retrieve(user_input, top_k=config['top_k'])

                # Format context
                context = ""
                for i, doc in enumerate(retrieved_docs, 1):
                    context += f"Document {i} (Source: {doc['metadata']['source']}):\n{doc['content']}\n\n"

                # Create messages for Grok
                messages = [
                    {
                        "role": "system",
                        "content": """You are a GIS and Remote Sensing expert. Use the provided context to answer questions accurately.
                        If the context doesn't contain enough information, supplement with your expert knowledge.
                        Provide comprehensive, technical answers with specific details when relevant."""
                    },
                    {
                        "role": "user",
                        "content": f"""Based on the following context, please answer the question:

Context:
{context}

Question: {user_input}

Please provide a comprehensive, technically accurate answer. If you use general knowledge beyond the context, please indicate that."""
                    }
                ]

                # Generate response
                provider = st.session_state.xai_provider
                response = provider.generate_response(messages, config['max_tokens'])

                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'sources': retrieved_docs,
                    'timestamp': datetime.now().isoformat()
                })

                # Rerun to display the response
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Response generation error: {str(e)}")


def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Render source citations."""
    if not sources:
        return ""

    sources_html = '<div class="source-info"><strong>Sources:</strong><ul>'
    for source in sources:
        source_name = source['metadata'].get('source', 'Unknown')
        page = source['metadata'].get('page', 'N/A')
        score = source.get('score', 0)
        sources_html += f'<li>{source_name} (Page {page}) [Relevance: {score}]</li>'
    sources_html += '</ul></div>'

    return sources_html


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Render sidebar
    config = render_sidebar()

    # Render main interface
    render_chat_interface(config)

    # Footer
    st.markdown("---")
    st.markdown("**About:** This is a demonstration of XAI Grok-3 integrated with GIS document retrieval.")
    st.markdown("**Sample Questions to Try:**")
    st.markdown("- What is NDVI and how is it calculated?")
    st.markdown("- Which bands does Sentinel-2 use for NDVI?")
    st.markdown("- What's the spatial resolution difference between Sentinel-2 and Landsat-8?")
    st.markdown("- Explain the NDVI formula in detail.")


if __name__ == "__main__":
    main()