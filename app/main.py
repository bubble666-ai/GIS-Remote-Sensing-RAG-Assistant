"""
Streamlit web interface for GIS and Remote Sensing RAG assistant.
Provides an interactive chat interface with configuration options.
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
import seaborn as sns
from dotenv import load_dotenv

from .retriever import GISDocumentRetriever
from .rag_chain import GISRAGChain
from .model_providers import ModelProviderFactory
from .model_tester import ModelTester

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

# Custom CSS for better styling
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
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
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
    if 'available_models' not in st.session_state:
        st.session_state.available_models = {}
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")

    # Load available models
    if not st.session_state.available_models:
        st.session_state.available_models = ModelProviderFactory.get_available_models()

    # Model Selection
    st.sidebar.subheader("🤖 Model Settings")

    # Group models by provider
    provider_models = ModelProviderFactory.list_models_by_provider()
    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        list(provider_models.keys()),
        help="Choose the model provider"
    )

    if selected_provider in provider_models:
        model_options = provider_models[selected_provider]
        selected_model = st.sidebar.selectbox(
            "Select Model",
            model_options,
            help="Choose the specific model"
        )
    else:
        selected_model = "gpt-3.5-turbo"

    # Temperature
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Controls randomness in responses"
    )

    # Max Tokens
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum length of generated responses"
    )

    # Retrieval Settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top K Documents",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of documents to retrieve"
    )

    # Vector Store Type
    vector_store_type = st.sidebar.selectbox(
        "Vector Store",
        ["FAISS", "ChromaDB"],
        index=0
    )

    # Data Management
    st.sidebar.subheader("Data Management")
    rebuild_index = st.sidebar.button(
        "🔄 Rebuild Document Index",
        help="Rebuild the document vector index"
    )

    clear_chat = st.sidebar.button(
        "🗑️ Clear Chat History",
        help="Clear all chat history"
    )

    # API Configuration
    st.sidebar.subheader("🔑 API Configuration")

    # Display model availability status
    if selected_model in st.session_state.available_models:
        model_info = st.session_state.available_models[selected_model]
        if model_info.get("available", False):
            st.sidebar.success(f"✅ {selected_model} is available")
        else:
            st.sidebar.warning(f"⚠️ {selected_model} not available")
            if "error" in model_info:
                st.sidebar.caption(f"Error: {model_info['error']}")

    # API Keys input
    with st.sidebar.expander("API Keys", expanded=False):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", "")
        )

        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key",
            value=os.getenv("ANTHROPIC_API_KEY", "")
        )

        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key",
            value=os.getenv("GOOGLE_API_KEY", "")
        )

        xai_api_key = st.text_input(
            "XAI API Key",
            type="password",
            help="Enter your XAI API key",
            value=os.getenv("XAI_API_KEY", "")
        )

        zhipu_api_key = st.text_input(
            "Zhipu API Key",
            type="password",
            help="Enter your Zhipu API key",
            value=os.getenv("ZHIPU_API_KEY", "")
        )

        # Update environment variables
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if xai_api_key:
            os.environ["XAI_API_KEY"] = xai_api_key
        if zhipu_api_key:
            os.environ["ZHIPU_API_KEY"] = zhipu_api_key

    # Document Statistics
    if st.session_state.document_stats:
        st.sidebar.subheader("📊 Document Statistics")
        stats = st.session_state.document_stats
        st.sidebar.metric("Total Chunks", stats.get('total_documents', 0))
        st.sidebar.metric("Unique Sources", stats.get('unique_sources', 0))
        st.sidebar.metric("Avg Chunk Length", f"{stats.get('avg_chunk_length', 0):.0f}")

    return {
        'model': selected_model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k,
        'vector_store_type': vector_store_type,
        'rebuild_index': rebuild_index,
        'clear_chat': clear_chat
    }


def initialize_system(config: Dict[str, Any]):
    """Initialize the RAG system with given configuration."""
    try:
        with st.spinner("Initializing GIS Assistant..."):
            # Initialize retriever
            retriever = GISDocumentRetriever(
                data_dir="data",
                embeddings_dir="embeddings",
                use_faiss=(config['vector_store_type'] == 'FAISS')
            )

            # Load or rebuild index
            force_rebuild = config['rebuild_index']
            retriever.initialize_or_load(force_rebuild=force_rebuild)

            # Get document statistics
            st.session_state.document_stats = retriever.get_document_stats()

            # Initialize RAG chain
            rag_chain = GISRAGChain(
                retriever=retriever,
                model_name=config['model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                use_openai=(config['model'] != 'local')
            )

            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.system_initialized = True

        st.success("✅ GIS Assistant initialized successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return False


def render_chat_interface():
    """Render the main chat interface."""
    st.title("🌍 GIS & Remote Sensing Assistant")
    st.markdown("""
    Ask questions about Geographic Information Systems, Remote Sensing, Satellite Imagery, and more.
    I'll provide answers based on the available documents and my specialized knowledge.
    """)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message['content']}
                    {render_sources(message.get('sources', []))}
                </div>
                """, unsafe_allow_html=True)

                # Display metrics if available
                if 'metrics' in message:
                    render_metrics(message['metrics'])

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
        with st.spinner("🔍 Searching documents and generating response..."):
            try:
                result = st.session_state.rag_chain.generate_response(
                    query=user_input,
                    top_k=st.session_state.get('top_k', 3),
                    include_sources=True
                )

                # Add assistant response to chat history
                assistant_message = {
                    'role': 'assistant',
                    'content': result['response'],
                    'sources': result['sources'],
                    'timestamp': result['timestamp'],
                    'metrics': {
                        'retrieval_time': result['retrieval_time'],
                        'generation_time': result['generation_time'],
                        'total_time': result['total_time'],
                        'token_usage': result['token_usage']
                    }
                }
                st.session_state.chat_history.append(assistant_message)

                # Log the interaction
                log_interaction(user_input, result)

                # Rerun to display the response
                st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Response generation error: {str(e)}")

    elif user_input and not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar configuration.")


def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Render source citations."""
    if not sources:
        return ""

    sources_html = '<div class="source-info"><strong>📚 Sources:</strong><ul>'
    for source in sources[:3]:  # Limit to top 3 sources
        source_name = source['metadata'].get('source', 'Unknown')
        page = source['metadata'].get('page', 'N/A')
        score = source.get('score', 0)
        sources_html += f'<li>{source_name} (Page {page}) [Relevance: {score:.3f}]</li>'
    sources_html += '</ul></div>'

    return sources_html


def render_metrics(metrics: Dict[str, Any]):
    """Render performance metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Retrieval Time",
            f"{metrics['retrieval_time']:.2f}s"
        )

    with col2:
        st.metric(
            "Generation Time",
            f"{metrics['generation_time']:.2f}s"
        )

    with col3:
        st.metric(
            "Total Time",
            f"{metrics['total_time']:.2f}s"
        )

    with col4:
        token_usage = metrics['token_usage']
        if token_usage.get('total_tokens', 0) > 0:
            st.metric(
                "Tokens Used",
                token_usage['total_tokens']
            )
        else:
            st.metric("Tokens Used", "N/A")


def log_interaction(query: str, result: Dict[str, Any]):
    """Log the interaction to a CSV file."""
    try:
        log_file = Path("logs/chat_history.csv")
        log_file.parent.mkdir(exist_ok=True)

        # Prepare log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': result['response'][:200] + "..." if len(result['response']) > 200 else result['response'],
            'retrieval_time': result['retrieval_time'],
            'generation_time': result['generation_time'],
            'total_time': result['total_time'],
            'sources_count': len(result['sources']),
            'tokens_used': result['token_usage'].get('total_tokens', 0)
        }

        # Append to CSV
        df = pd.DataFrame([log_entry])
        if log_file.exists():
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

    except Exception as e:
        logger.error(f"Error logging interaction: {str(e)}")


def render_document_stats():
    """Render document statistics visualization."""
    if not st.session_state.document_stats:
        return

    stats = st.session_state.document_stats

    st.subheader("📊 Document Statistics")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", stats.get('total_documents', 0))

    with col2:
        st.metric("Unique Sources", stats.get('unique_sources', 0))

    with col3:
        st.metric("Total Characters", f"{stats.get('total_chars', 0):,}")

    with col4:
        st.metric("Avg Chunk Length", f"{stats.get('avg_chunk_length', 0):.0f}")

    # Source distribution chart
    if 'source_distribution' in stats and stats['source_distribution']:
        st.subheader("Source Distribution")
        source_df = pd.DataFrame(
            list(stats['source_distribution'].items()),
            columns=['Source', 'Count']
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=source_df, x='Count', y='Source', ax=ax)
        ax.set_title('Document Distribution by Source')
        st.pyplot(fig)

    # Document type distribution
    if 'type_distribution' in stats and stats['type_distribution']:
        st.subheader("Document Types")
        type_df = pd.DataFrame(
            list(stats['type_distribution'].items()),
            columns=['Type', 'Count']
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie(type_df['Count'], labels=type_df['Type'], autopct='%1.1f%%')
        ax.set_title('Document Types Distribution')
        st.pyplot(fig)


def render_model_testing_interface():
    """Render the model testing interface."""
    st.title("🧪 Model Testing & Comparison")

    if not st.session_state.retriever:
        st.warning("⚠️ Please initialize the system first before testing models.")
        return

    # Initialize tester
    if 'model_tester' not in st.session_state:
        st.session_state.model_tester = ModelTester(st.session_state.retriever)

    tester = st.session_state.model_tester

    # Testing options
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Test Configuration")

        # Model selection for testing
        available_models = ModelProviderFactory.get_available_models()
        working_models = [name for name, info in available_models.items() if info.get("available", False)]

        if not working_models:
            st.error("❌ No working models found. Please configure API keys in the sidebar.")
            return

        selected_test_models = st.multiselect(
            "Select Models to Test",
            working_models,
            default=working_models[:2] if len(working_models) >= 2 else working_models,
            help="Choose which models to include in the comparison"
        )

        test_subset = st.selectbox(
            "Test Questions",
            ["All Questions", "Quick Test (3 questions)", "Basic Questions", "Advanced Questions"],
            help="Choose which questions to use for testing"
        )

        if test_subset == "All Questions":
            question_ids = None
        elif test_subset == "Quick Test (3 questions)":
            question_ids = [1, 2, 3]
        elif test_subset == "Basic Questions":
            question_ids = [1, 2, 6, 7]
        else:  # Advanced Questions
            question_ids = [3, 4, 5, 8]

    with col2:
        st.subheader("🚀 Run Tests")

        if st.button("🧪 Start Model Testing", type="primary", disabled=not selected_test_models):
            with st.spinner("🔄 Testing models... This may take a few minutes."):
                try:
                    results = tester.compare_models(
                        model_names=selected_test_models,
                        temperature=0.1,
                        max_tokens=1000,
                        test_subset=question_ids
                    )
                    st.session_state.test_results = results
                    st.success("✅ Model testing completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during testing: {str(e)}")

        if st.button("🗑️ Clear Test Results"):
            st.session_state.test_results = {}
            st.rerun()

    # Display test results
    if st.session_state.test_results:
        st.divider()
        st.subheader("📊 Test Results")

        results = st.session_state.test_results

        if "comparison_summary" in results:
            summary = results["comparison_summary"]

            # Overall ranking
            if "ranking" in summary:
                st.subheader("🏆 Model Ranking")
                ranking_df = pd.DataFrame(summary["ranking"])

                # Format for display
                display_df = ranking_df[["model", "avg_overall_score", "avg_response_time", "success_rate"]].copy()
                display_df.columns = ["Model", "Overall Score", "Avg Response Time (s)", "Success Rate"]
                display_df["Overall Score"] = display_df["Overall Score"].round(3)
                display_df["Avg Response Time (s)"] = display_df["Avg Response Time (s)"].round(2)
                display_df["Success Rate"] = (display_df["Success Rate"] * 100).round(1).astype(str) + "%"

                st.dataframe(display_df, use_container_width=True)

                # Best performers
                if "best_models" in summary:
                    st.subheader("🥇 Best Performers by Metric")
                    best_models = summary["best_models"]

                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Best Overall", best_models.get("best_avg_overall_score", "N/A"))
                    with cols[1]:
                        st.metric("Fastest", best_models.get("best_avg_response_time", "N/A"))
                    with cols[2]:
                        st.metric("Most Reliable", best_models.get("best_success_rate", "N/A"))

            # Detailed comparison chart
            if "ranking" in summary and len(summary["ranking"]) > 1:
                st.subheader("📈 Performance Comparison")

                # Create comparison chart
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                models = [item["model"] for item in summary["ranking"]]

                # Overall score
                axes[0, 0].bar(models, [item["avg_overall_score"] for item in summary["ranking"]])
                axes[0, 0].set_title("Overall Score")
                axes[0, 0].tick_params(axis='x', rotation=45)

                # Response time
                axes[0, 1].bar(models, [item["avg_response_time"] for item in summary["ranking"]])
                axes[0, 1].set_title("Response Time (s)")
                axes[0, 1].tick_params(axis='x', rotation=45)

                # Success rate
                axes[1, 0].bar(models, [item["success_rate"] for item in summary["ranking"]])
                axes[1, 0].set_title("Success Rate")
                axes[1, 0].tick_params(axis='x', rotation=45)

                # Tests per minute
                axes[1, 1].bar(models, [item["tests_per_minute"] for item in summary["ranking"]])
                axes[1, 1].set_title("Tests per Minute")
                axes[1, 1].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

            # Detailed results
            with st.expander("📋 Detailed Results", expanded=False):
                for model_name, result in results["detailed_results"].items():
                    if "error" not in result:
                        st.write(f"**{model_name}**")
                        st.json(result["summary"])

        # Test summary
        st.subheader("📝 Test Summary")
        st.write(f"- **Models Tested**: {len(results['models_tested'])}")
        st.write(f"- **Total Test Time**: {results['total_comparison_time']:.2f} seconds")

        if "detailed_results" in results:
            total_tests = sum(result.get("successful_tests", 0) for result in results["detailed_results"].values())
            st.write(f"- **Total Successful Tests**: {total_tests}")


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
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Statistics", "🧪 Model Testing"])

        with tab1:
            render_chat_interface()

        with tab2:
            render_document_stats()

        with tab3:
            render_model_testing_interface()
    else:
        st.warning("⚠️ Please configure and initialize the system using the sidebar.")
        st.info("📝 **Getting Started:**\n1. Add your GIS/Remote Sensing documents to the `data/` folder\n2. Configure your API keys in the sidebar\n3. Click 'Initialize System' or enable auto-initialization\n4. Start asking questions!")


if __name__ == "__main__":
    main()