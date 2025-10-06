"""
Clean Streamlit web interface for GIS and Remote Sensing RAG assistant.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Configuration")

    # Model Selection
    st.sidebar.subheader("Model Settings")

    model_options = [
        "grok-3",
        "grok-4-fast-reasoning",
        "gpt-3.5-turbo",
        "gpt-4",
        "local"
    ]

    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_options,
        index=0,
        help="Choose the AI model"
    )

    # Temperature
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )

    # Max Tokens
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
        max_value=10,
        value=3,
        step=1
    )

    # Data Management
    st.sidebar.subheader("Data Management")
    rebuild_index = st.sidebar.button("Rebuild Document Index")
    clear_chat = st.sidebar.button("Clear Chat History")

    # API Configuration
    st.sidebar.subheader("API Configuration")

    # Check XAI API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        st.sidebar.success("XAI API key configured")
    else:
        st.sidebar.warning("XAI API key not found")

    # Document Statistics
    if st.session_state.document_stats:
        st.sidebar.subheader("Document Statistics")
        stats = st.session_state.document_stats
        st.sidebar.metric("Total Chunks", stats.get('total_documents', 0))
        st.sidebar.metric("Unique Sources", stats.get('unique_sources', 0))

    return {
        'model': selected_model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k,
        'rebuild_index': rebuild_index,
        'clear_chat': clear_chat
    }


def initialize_system(config: Dict[str, Any]):
    """Initialize the RAG system with given configuration."""
    try:
        with st.spinner("Initializing GIS Assistant..."):
            # Initialize retriever
            from retriever import GISDocumentRetriever
            retriever = GISDocumentRetriever(
                data_dir="data",
                embeddings_dir="embeddings",
                use_faiss=True
            )

            # Load or rebuild index
            force_rebuild = config['rebuild_index']
            retriever.initialize_or_load(force_rebuild=force_rebuild)

            # Get document statistics
            st.session_state.document_stats = retriever.get_document_stats()

            # Initialize RAG chain
            from rag_chain import GISRAGChain
            rag_chain = GISRAGChain(
                retriever=retriever,
                model_name=config['model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens']
            )

            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.system_initialized = True

        st.success("GIS Assistant initialized successfully!")
        return True

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return False


def render_chat_interface():
    """Render the main chat interface."""
    st.title("GIS & Remote Sensing Assistant")
    st.markdown("""
    Ask questions about Geographic Information Systems, Remote Sensing, Satellite Imagery, and more.
    """)

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
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask your GIS/Remote Sensing question here...")

    if user_input and st.session_state.system_initialized:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Generate response
        with st.spinner("Searching documents and generating response..."):
            try:
                result = st.session_state.rag_chain.generate_response(
                    query=user_input,
                    top_k=st.session_state.get('top_k', 3),
                    include_sources=True
                )

                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'sources': result['sources'],
                    'timestamp': result['timestamp']
                })

                # Rerun to display the response
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Response generation error: {str(e)}")

    elif user_input and not st.session_state.system_initialized:
        st.warning("Please initialize the system first using the sidebar configuration.")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Render sidebar
    config = render_sidebar()

    # Handle configuration changes
    if config['clear_chat']:
        st.session_state.chat_history = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain.clear_conversation_history()
        st.rerun()

    # Initialize system if not already done
    if not st.session_state.system_initialized or config['rebuild_index']:
        if initialize_system(config):
            if config['rebuild_index']:
                st.rerun()

    # Render main interface
    if st.session_state.system_initialized:
        render_chat_interface()
    else:
        st.warning("Please configure and initialize the system using the sidebar.")
        st.info("Getting Started:\n1. Add your GIS/Remote Sensing documents to the `data/` folder\n2. Configure your API key in the sidebar\n3. Click 'Initialize System'\n4. Start asking questions!")


if __name__ == "__main__":
    main()