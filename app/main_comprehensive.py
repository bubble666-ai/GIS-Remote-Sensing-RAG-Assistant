"""
Comprehensive Streamlit app for GIS & Remote Sensing RAG Assistant.
Supports multiple AI providers with all latest models.
"""

import os
import time
import logging
import re
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Iterator
import json

import streamlit as st
import pandas as pd
import openai
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# Optional imports for enhanced features
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import PyPDF2
    import docx
    DOCUMENT_PROCESSORS_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSORS_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="🌍 Multi-Model GIS Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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

    /* Assistant message styling - darker green/teal */
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

    /* Model provider badges */
    .provider-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .provider-xai { background: #ef4444; }
    .provider-google { background: #3b82f6; }
    .provider-anthropic { background: #f59e0b; }
    .provider-openai { background: #10b981; }
    .provider-zhipu { background: #8b5cf6; }
    .provider-google-robotics { background: #6366f1; }

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


class UniversalAIProvider:
    """Universal AI provider supporting multiple model families."""

    def __init__(self, api_key: str, model_name: str = "grok-3", provider: str = "xai"):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        try:
            if self.provider == "xai":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
            elif self.provider == "google":
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
            elif self.provider == "anthropic":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.provider == "openai":
                self.client = openai.OpenAI(api_key=self.api_key)
            elif self.provider == "zhipu":
                # Zhipu AI uses direct API calls
                import requests
                self.client = requests.Session()
                self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
            elif self.provider == "google_robotics":
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized {self.provider} client for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing {self.provider} client: {str(e)}")
            self.client = None

    def get_all_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method - use the global function instead."""
        return get_all_available_models()

    def switch_model(self, provider: str, model_name: str) -> bool:
        """Switch to a different model and provider."""
        try:
            self.provider = provider
            self.model_name = model_name
            self._initialize_client()
            return self.client is not None
        except Exception as e:
            logger.error(f"Error switching to {provider}/{model_name}: {str(e)}")
            return False

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generate response using the selected model."""
        try:
            if self.provider == "xai" or self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            elif self.provider == "google" or self.provider == "google_robotics":
                response = self.client.generate_content(
                    "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text

            elif self.provider == "anthropic":
                system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
                user_messages = [msg for msg in messages if msg['role'] != 'system']

                if system_msg:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        system=system_msg['content'],
                        messages=user_messages
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        messages=user_messages
                    )
                return response.content[0].text

            elif self.provider == "zhipu":
                # Zhipu AI API call
                import requests
                url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": max_tokens
                }
                response = requests.post(url, headers=headers, json=payload)

                # Check if request was successful
                if response.status_code != 200:
                    error_info = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    raise Exception(f"Zhipu API error {response.status_code}: {error_info}")

                result = response.json()

                # Check for API errors in response
                if 'error' in result:
                    raise Exception(f"Zhipu API error: {result['error']}")

                # Return the response content
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception("Invalid response format from Zhipu API")

            else:
                return f"Unsupported provider: {self.provider}"

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def is_available(self) -> bool:
        """Check if the current model is available."""
        return self.client is not None


class HuggingFaceLocalProvider:
    """Hugging Face Local Model Provider for offline inference."""

    def __init__(self, model_name: str = "ibm-granite/granite-docling-258M"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not installed. Install with: pip install transformers torch")

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.provider = "local"
        self.api_key = "local"

        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device == "cuda" else -1,
                max_new_tokens=1000,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info(f"Initialized local model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing local model: {str(e)}")
            self.pipe = None

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generate response using local Hugging Face model."""
        if not self.pipe:
            return "Error: Local model not initialized"

        try:
            # Format messages for the model
            prompt = self._format_messages(messages)

            # Generate response
            response = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract only the new content
            generated_text = response[0]['generated_text']
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for the model."""
        formatted = ""
        for msg in messages:
            if msg['role'] == 'system':
                formatted += f"System: {msg['content']}\n"
            elif msg['role'] == 'user':
                formatted += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                formatted += f"Assistant: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted

    def is_available(self) -> bool:
        """Check if the local model is available."""
        return self.pipe is not None


class UnifiedDocumentLoader:
    """Unified document loader supporting multiple file formats."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.supported_formats = {'.txt', '.pdf', '.docx', '.csv'}
        self.processors_available = DOCUMENT_PROCESSORS_AVAILABLE

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load all supported documents from data directory."""
        documents = []

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    content = self.extract_text(file_path)
                    if content:
                        documents.append({
                            'content': content,
                            'metadata': {
                                'source': str(file_path),
                                'file_type': file_path.suffix.lower(),
                                'file_name': file_path.name,
                                'file_size': file_path.stat().st_size
                            }
                        })
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        # Add default GIS documents if no documents found
        if not documents:
            documents = self._get_default_gis_documents()

        return documents

    def extract_text(self, file_path: Path) -> str:
        """Extract text based on file type."""
        if file_path.suffix.lower() == '.txt':
            return self._extract_txt(file_path)
        elif file_path.suffix.lower() == '.pdf' and self.processors_available:
            return self._extract_pdf(file_path)
        elif file_path.suffix.lower() == '.docx' and self.processors_available:
            return self._extract_docx(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._extract_csv(file_path)
        else:
            # Fallback to default text extraction
            return f"Document: {file_path.name}\nUnable to extract text from this file format."

    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            return f"Unable to read {file_path.name} with supported encodings."

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        if not self.processors_available:
            return "PDF processing not available. Install PyPDF2: pip install PyPDF2"

        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"Page {page_num + 1}:\n{page_text}\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            return f"Error extracting PDF content: {str(e)}"

    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        if not self.processors_available:
            return "DOCX processing not available. Install python-docx: pip install python-docx"

        try:
            doc = docx.Document(file_path)
            text = ""
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    text += f"Paragraph {i+1}: {paragraph.text}\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
            return f"Error extracting DOCX content: {str(e)}"

    def _extract_csv(self, file_path: Path) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Convert to readable text format
            text = f"CSV File: {file_path.name}\n"
            text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            text += f"Columns: {', '.join(df.columns)}\n\n"

            # Add data types
            text += "Data Types:\n"
            for col in df.columns:
                text += f"  - {col}: {df[col].dtype}\n"

            text += "\nSample Data (first 10 rows):\n"
            text += df.head(10).to_string(index=False)

            if len(df) > 10:
                text += f"\n\n... and {len(df) - 10} more rows"

            return text
        except Exception as e:
            logger.error(f"Error extracting CSV {file_path}: {e}")
            return f"Error extracting CSV content: {str(e)}"

    def _get_default_gis_documents(self) -> List[Dict[str, Any]]:
        """Get default GIS documents when no documents are found."""
        return [
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
                """,
                "metadata": {
                    "source": "ndvi_analysis.txt",
                    "file_type": ".txt",
                    "file_name": "ndvi_analysis.txt",
                    "category": "Vegetation Analysis"
                }
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
                """,
                "metadata": {
                    "source": "sentinel_overview.txt",
                    "file_type": ".txt",
                    "file_name": "sentinel_overview.txt",
                    "category": "Satellite Missions"
                }
            }
        ]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval for demo purposes."""
        # Convert document list to retriever format
        docs = self.load_all_documents()

        # Simple keyword matching
        query_words = query.lower().split()
        scored_docs = []

        for doc in docs:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_docs.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score
                })

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


# Database Integration Classes
class DatabaseExporter:
    """Export database tables to text files for RAG processing."""

    def __init__(self, db_type: str, connection_params: dict, output_dir: str = "data"):
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_all_tables(self):
        """Export all tables to text files."""
        if self.db_type == 'sqlite':
            self._export_sqlite_tables()
        elif self.db_type == 'postgresql':
            self._export_postgresql_tables()
        else:
            logger.error(f"Unsupported database type: {self.db_type}")

    def _export_sqlite_tables(self):
        """Export SQLite tables to text files."""
        try:
            conn = sqlite3.connect(self.connection_params['database'])
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table_name, in tables:
                self._export_table_to_file(conn, table_name)

            conn.close()
            logger.info(f"Exported {len(tables)} SQLite tables")

        except Exception as e:
            logger.error(f"Error exporting SQLite tables: {e}")

    def _export_postgresql_tables(self):
        """Export PostgreSQL tables to text files."""
        if not POSTGRESQL_AVAILABLE:
            logger.error("PostgreSQL support not available. Install psycopg2: pip install psycopg2-binary")
            return

        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';
            """)
            tables = cursor.fetchall()

            for table_name, in tables:
                self._export_table_to_file(conn, table_name)

            conn.close()
            logger.info(f"Exported {len(tables)} PostgreSQL tables")

        except Exception as e:
            logger.error(f"Error exporting PostgreSQL tables: {e}")

    def _export_table_to_file(self, conn, table_name: str):
        """Export a single table to a text file."""
        try:
            # Read table data
            query = f"SELECT * FROM {table_name} LIMIT 100"
            df = pd.read_sql(query, conn)

            # Get column info
            cursor = conn.cursor()
            if self.db_type == 'sqlite':
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
            else:
                cursor.execute(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}';
                """)
                columns_info = cursor.fetchall()

            # Create text content
            content = f"Table: {table_name}\n"
            content += "=" * 50 + "\n\n"

            # Column information
            content += "Columns:\n"
            for col_info in columns_info:
                if self.db_type == 'sqlite':
                    content += f"  - {col_info[1]} ({col_info[2]})\n"
                else:
                    content += f"  - {col_info[0]} ({col_info[1]})\n"

            content += f"\nData Preview (First {len(df)} rows):\n"
            content += df.to_string(index=False)

            # Save to file
            output_file = self.output_dir / f"{table_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Exported {table_name} to {output_file}")

        except Exception as e:
            logger.error(f"Error exporting table {table_name}: {e}")


class SQLQueryGenerator:
    """Generate safe SQL queries from natural language questions."""

    def __init__(self, db_schema: Dict[str, Dict]):
        self.db_schema = db_schema  # {table_name: {columns: [col_names], types: [col_types]}}

    def generate_safe_query(self, user_question: str) -> str:
        """Generate a safe SQL query from user question."""
        try:
            # Extract table and column references
            mentioned_tables = self._extract_tables(user_question)
            mentioned_columns = self._extract_columns(user_question)

            # Parse conditions
            conditions = self._extract_conditions(user_question)

            # Build query
            if not mentioned_tables:
                return None

            # Select the most relevant table
            table_name = self._select_best_table(mentioned_tables, mentioned_columns)

            # Build SELECT clause
            columns = self._select_columns(table_name, mentioned_columns)

            # Build WHERE clause
            where_clause = self._build_where_clause(conditions, table_name)

            # Construct final query
            query = f"SELECT {columns} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            query += " LIMIT 1000"  # Safety limit

            return query

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None

    def _extract_tables(self, question: str) -> List[str]:
        """Extract table names from question."""
        mentioned = []
        question_lower = question.lower()

        # Common GIS table keywords
        gis_keywords = ['ndvi', 'satellite', 'landsat', 'sentinel', 'vegetation', 'imagery', 'remote sensing']

        for table_name in self.db_schema.keys():
            if table_name.lower() in question_lower or any(keyword in table_name.lower() for keyword in gis_keywords):
                mentioned.append(table_name)

        return mentioned

    def _extract_columns(self, question: str) -> List[str]:
        """Extract column names from question."""
        mentioned = []
        question_lower = question.lower()

        for table_name, table_info in self.db_schema.items():
            for col in table_info['columns']:
                if col.lower() in question_lower:
                    mentioned.append(col)

        return mentioned

    def _extract_conditions(self, question: str) -> List[Dict]:
        """Extract conditions from question."""
        conditions = []

        # Pattern for numeric ranges (e.g., "between 0.6 and 0.8")
        range_pattern = r'(\w+)\s+between\s+([\d.]+)\s+and\s+([\d.]+)'
        matches = re.findall(range_pattern, question.lower())

        for col, min_val, max_val in matches:
            conditions.append({
                'column': col,
                'operator': 'BETWEEN',
                'value': (float(min_val), float(max_val))
            })

        # Pattern for exact values
        value_pattern = r'(\w+)\s+(?:equals?|=)\s+([\d.]+)'
        matches = re.findall(value_pattern, question.lower())

        for col, value in matches:
            conditions.append({
                'column': col,
                'operator': '=',
                'value': float(value)
            })

        return conditions

    def _select_best_table(self, mentioned_tables: List[str], mentioned_columns: List[str]) -> str:
        """Select the most relevant table based on mentioned columns."""
        if not mentioned_tables:
            return None

        if len(mentioned_tables) == 1:
            return mentioned_tables[0]

        # Score tables based on column matches
        best_table = mentioned_tables[0]
        best_score = 0

        for table in mentioned_tables:
            if table in self.db_schema:
                score = 0
                table_columns = set(self.db_schema[table]['columns'])
                mentioned_cols_set = set(mentioned_columns)

                # Count matching columns
                score = len(table_columns.intersection(mentioned_cols_set))

                if score > best_score:
                    best_score = score
                    best_table = table

        return best_table

    def _select_columns(self, table_name: str, mentioned_columns: List[str]) -> str:
        """Select appropriate columns for the query."""
        if mentioned_columns and table_name in self.db_schema:
            # Filter columns that exist in the table
            table_columns = self.db_schema[table_name]['columns']
            valid_columns = [col for col in mentioned_columns if col in table_columns]

            if valid_columns:
                return ', '.join(valid_columns)

        return '*'  # Select all columns if no specific columns mentioned

    def _build_where_clause(self, conditions: List[Dict], table_name: str) -> str:
        """Build WHERE clause from conditions."""
        if not conditions:
            return ""

        if table_name not in self.db_schema:
            return ""

        table_columns = self.db_schema[table_name]['columns']
        valid_conditions = []

        for condition in conditions:
            if condition['column'] in table_columns:
                if condition['operator'] == 'BETWEEN':
                    min_val, max_val = condition['value']
                    valid_conditions.append(
                        f"{condition['column']} BETWEEN {min_val} AND {max_val}"
                    )
                elif condition['operator'] == '=':
                    valid_conditions.append(
                        f"{condition['column']} = {condition['value']}"
                    )

        return ' AND '.join(valid_conditions) if valid_conditions else ""


class MetadataEnhancedRetriever:
    """Enhanced retriever with rich metadata support."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.loader = UnifiedDocumentLoader(data_dir)
        self.documents = self.loader.load_all_documents()

        # Add chunking if langchain is available
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        else:
            self.text_splitter = None

    def add_documents_with_metadata(self, documents: List[Dict[str, Any]] = None):
        """Add documents with rich metadata."""
        if documents is None:
            documents = self.documents

        if not self.text_splitter:
            # Simple approach without langchain
            return documents

        enriched_documents = []

        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc['content'])

            for i, chunk in enumerate(chunks):
                # Create metadata for each chunk
                metadata = doc['metadata'].copy()
                metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
                })

                enriched_documents.append({
                    'content': chunk,
                    'metadata': metadata
                })

        return enriched_documents

    def retrieve_with_metadata(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve documents with their metadata."""
        # Use simple keyword matching for now
        enriched_docs = self.add_documents_with_metadata()

        query_words = query.lower().split()
        scored_docs = []

        for doc in enriched_docs:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)

            if score > 0:
                scored_docs.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity_score': score / len(query_words)
                })

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_docs[:k]
def initialize_session_state():
    """Initialize enhanced session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ai_provider' not in st.session_state:
        st.session_state.ai_provider = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = MetadataEnhancedRetriever()
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = "xai"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "grok-3"
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}


def render_sidebar():
    """Render enhanced sidebar with multi-provider model selection."""
    st.sidebar.title("🌍 Multi-Model GIS Assistant")

    # Check API keys for different providers
    api_keys = {
        "xai": os.getenv("XAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "zhipu": os.getenv("ZHIPU_API_KEY")
    }

    # Initialize provider if not done yet
    if not st.session_state.ai_provider:
        # Find first available API key
        for provider, key in api_keys.items():
            if key:
                st.session_state.ai_provider = UniversalAIProvider(
                    key,
                    st.session_state.selected_model,
                    provider
                )
                st.session_state.selected_provider = provider
                break

    # Multi-provider model selection
    st.sidebar.subheader("🤖 Model Selection")

    # Get all available models (cached)
    all_models = get_all_available_models()

    # Provider selection
    provider_names = list(all_models.keys())
    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        options=provider_names,
        index=provider_names.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_names else 0,
        help="Choose the AI provider"
    )

    # Get models for selected provider
    models_for_provider = all_models.get(selected_provider, [])
    model_options = [(model["display_name"], model["name"]) for model in models_for_provider]

    if model_options:
        # Try to find the current model in the provider's model list
        current_model_index = 0
        for i, (display_name, internal_name) in enumerate(model_options):
            if internal_name == st.session_state.selected_model:
                current_model_index = i
                break

        selected_display_name = st.sidebar.selectbox(
            "Select Model",
            options=[name for name, _ in model_options],
            index=current_model_index,
            help="Select which model to use"
        )

        # Find the selected model's internal name
        selected_model_name = None
        for display_name, internal_name in model_options:
            if display_name == selected_display_name:
                selected_model_name = internal_name
                break

        # Check if we need to switch provider/model
        if (selected_provider != st.session_state.selected_provider or
            selected_model_name != st.session_state.selected_model):

            # Check if API key exists for this provider
            if api_keys.get(selected_provider):
                if st.session_state.ai_provider.switch_model(selected_provider, selected_model_name):
                    st.session_state.selected_provider = selected_provider
                    st.session_state.selected_model = selected_model_name
                    st.sidebar.success(f"Switched to {selected_display_name}")
                else:
                    st.sidebar.error(f"Failed to switch to {selected_display_name}")
            else:
                st.sidebar.error(f"API key required for {selected_provider}")
                st.sidebar.info(f"Add {selected_provider.upper()}_API_KEY to .env file")

    # Display current model info
    current_model_info = next(
        (m for m in models_for_provider if m["name"] == st.session_state.selected_model),
        None
    )

    if current_model_info:
        with st.sidebar.expander("Model Information", expanded=False):
            st.markdown(f"**Context Length:** {current_model_info['context_length']:,} tokens")
            st.markdown(f"**Description:** {current_model_info['description']}")
            if 'rpm' in current_model_info:
                st.markdown(f"**Rate Limit:** {current_model_info['rpm']} requests/minute")
            st.markdown(f"**Cost:** ${current_model_info.get('cost_per_m_tokens', 0)}/M tokens")

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

    # Header with current model info
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("## 🌍 Multi-Model GIS & Remote Sensing Assistant")
        st.markdown("*Powered by multiple AI providers with intelligent document retrieval*")

    with col2:
        if st.session_state.selected_provider and st.session_state.selected_model:
            provider_badge = f'<span class="provider-badge provider-{st.session_state.selected_provider}">{st.session_state.selected_provider.upper()}</span>'

            # Get display name for current model
            current_model_display = st.session_state.selected_model
            all_models = get_all_available_models()
            provider_models = all_models.get(st.session_state.selected_provider, [])
            for model in provider_models:
                if model['name'] == st.session_state.selected_model:
                    current_model_display = model['display_name']
                    break

            st.markdown(f"**Current Model:** {current_model_display}", unsafe_allow_html=True)
            st.markdown(provider_badge, unsafe_allow_html=True)

    with col3:
        st.markdown(f"**Provider:** {st.session_state.selected_provider.title()}")

    # Check if provider is available
    if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
        st.error("❌ No AI provider available. Please check your API key configuration.")

        # Show required API keys
        api_keys = {
            "xai": os.getenv("XAI_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "zhipu": os.getenv("ZHIPU_API_KEY")
        }
        missing_keys = [k.upper() for k, v in api_keys.items() if not v]
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.info("Add API keys to the `.env` file:")
            for key in missing_keys:
                st.code(f"{key}_API_KEY=your_key_here")

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

        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"{i+1}. {question}", key=f"sample_{i}"):
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
            retrieved_docs = retriever.retrieve_with_metadata(user_input, k=config['top_k'])

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

            # Create enhanced messages for AI model
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert GIS and Remote Sensing specialist with deep knowledge of:
                    - Satellite imagery and remote sensing platforms (Landsat, Sentinel, MODIS, etc.)
                    - Spectral indices and vegetation analysis (NDVI, EVI, SAVI, etc.)
                    - Image processing and atmospheric correction techniques
                    - GIS data analysis and spatial statistics
                    - Cartography and map projections
                    - Geographic coordinate systems and transformations
                    - Land use/land cover classification
                    - Change detection and time series analysis
                    - Digital elevation models and terrain analysis

                    **Current Model:** {get_model_display_name(st.session_state.selected_provider, st.session_state.selected_model)} ({st.session_state.selected_provider.title()})

                    **Instructions:**
                    1. Use the provided context documents when they are relevant
                    2. Supplement with your expert knowledge when needed
                    3. Provide comprehensive, technically accurate answers
                    4. Include specific details, formulas, or methodologies when relevant
                    5. Use proper GIS terminology
                    6. If information is from general knowledge beyond the context, clearly indicate that
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
            provider = st.session_state.ai_provider
            response = provider.generate_response(messages, config['max_tokens'])

            generation_time = time.time() - start_time

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': retrieved_docs,
                'timestamp': datetime.now().isoformat(),
                'model_used': f"{st.session_state.selected_provider}/{st.session_state.selected_model}",
                'generation_time': generation_time,
                'provider': st.session_state.selected_provider
            })

            # Rerun to display the response
            st.rerun()

        except Exception as e:
            error_msg = str(e)
            provider_name = st.session_state.selected_provider.title() if 'selected_provider' in st.session_state else "Unknown"

            # Enhanced error detection for different providers
            if "googleapis.com" in error_msg or "generativelanguage.googleapis.com" in error_msg:
                provider_name = "Google"
            elif "x.ai" in error_msg or "api.x.ai" in error_msg:
                provider_name = "XAI"
            elif "api.anthropic.com" in error_msg or "anthropic" in error_msg.lower():
                provider_name = "Anthropic"
            elif "openai.com" in error_msg or "openai" in error_msg.lower():
                provider_name = "OpenAI"
            elif "bigmodel.cn" in error_msg or "zhipu" in error_msg.lower():
                provider_name = "Zhipu"

            # Provide more specific error messages
            if "API key not valid" in error_msg or "API_KEY_INVALID" in error_msg:
                st.error(f"❌ Invalid {provider_name} API key. Please check your {provider_name.upper()}_API_KEY in the .env file.")
                st.info(f"To get a new {provider_name} API key, visit the {provider_name.lower()} console/platform.")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                st.error(f"❌ Authentication failed for {provider_name}. Please check your {provider_name.upper()}_API_KEY.")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                st.error(f"❌ Rate limit exceeded for {provider_name}. Please try again in a few moments.")
            elif "timeout" in error_msg.lower():
                st.error(f"❌ Request timeout for {provider_name}. Please try again.")
            else:
                st.error(f"❌ Error generating response with {provider_name}: {error_msg}")

            logger.error(f"Response generation error with {provider_name}: {error_msg}")


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
        st.markdown("- ✅ Multiple AI providers (XAI, Google, Anthropic, OpenAI, Zhipu)")
        st.markdown("- ✅ 20+ models to choose from")
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
        st.markdown("- [XAI Console](https://console.x.ai/)")
        st.markdown("- [Google AI Studio](https://aistudio.google.com/)")
        st.markdown("- [Anthropic Console](https://console.anthropic.com/)")
        st.markdown("- [OpenAI Platform](https://platform.openai.com/)")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🌍 Multi-Model GIS & Remote Sensing Assistant | "
        f"Provider: <strong>{st.session_state.selected_provider.title()}</strong> | "
        f"Model: <strong>{get_model_display_name(st.session_state.selected_provider, st.session_state.selected_model)}</strong>"
        "</div>",
        unsafe_allow_html=True
    )


def get_model_display_name(provider, model_name):
    """Get display name for a model."""
    try:
        provider_instance = UniversalAIProvider("dummy", "dummy", provider)
        all_models = provider_instance.get_all_available_models()
        provider_models = all_models.get(provider, [])
        for model in provider_models:
            if model['name'] == model_name:
                return model['display_name']
    except:
        pass
    return model_name


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get all available models from all providers (cached)."""
    models = {}

    # 1. OpenAI Models (show demo models if no API key)
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            models_list = client.models.list()
            openai_models = []
            for m in models_list.data:
                if not m.id.startswith("text-embedding"):  # Skip embedding models
                    openai_models.append({
                        "name": m.id,
                        "display_name": m.id.replace("-", " ").title(),
                        "description": f"OpenAI {m.id}",
                        "context_length": 128000,  # Default
                        "provider": "openai",
                        "cost_per_m_tokens": 0
                    })
            models["openai"] = openai_models
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {str(e)}")

    # Always show OpenAI demo models
    if "openai" not in models:
        models["openai"] = [
            {"name": "gpt-4o", "display_name": "GPT-4o", "description": "OpenAI GPT-4o", "context_length": 128000, "provider": "openai", "cost_per_m_tokens": 5},
            {"name": "gpt-4o-mini", "display_name": "GPT-4o Mini", "description": "OpenAI GPT-4o Mini", "context_length": 128000, "provider": "openai", "cost_per_m_tokens": 0.15},
            {"name": "gpt-4-turbo", "display_name": "GPT-4 Turbo", "description": "OpenAI GPT-4 Turbo", "context_length": 128000, "provider": "openai", "cost_per_m_tokens": 10}
        ]

    # 2. Google Gemini Models (show demo models if no API key)
    if os.getenv("GOOGLE_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            models_list = genai.list_models()
            google_models = []
            for m in models_list:
                if "generateContent" in m.supported_generation_methods:
                    model_name = m.name.replace("models/", "")
                    google_models.append({
                        "name": model_name,
                        "display_name": model_name.replace("-", " ").title(),
                        "description": f"Google {model_name}",
                        "context_length": getattr(m, 'input_token_limit', 1000000),
                        "provider": "google",
                        "cost_per_m_tokens": 0
                    })
            models["google"] = google_models
        except Exception as e:
            logger.error(f"Failed to fetch Google models: {str(e)}")

    # Always show Google demo models
    if "google" not in models:
        models["google"] = [
            {"name": "gemini-1.5-pro", "display_name": "Gemini 1.5 Pro", "description": "Google Gemini 1.5 Pro", "context_length": 2000000, "provider": "google", "cost_per_m_tokens": 0},
            {"name": "gemini-1.5-flash", "display_name": "Gemini 1.5 Flash", "description": "Google Gemini 1.5 Flash", "context_length": 1000000, "provider": "google", "cost_per_m_tokens": 0},
            {"name": "gemini-2.0-flash", "display_name": "Gemini 2.0 Flash", "description": "Google Gemini 2.0 Flash", "context_length": 1000000, "provider": "google", "cost_per_m_tokens": 0}
        ]

    # 3. Anthropic (Claude) Models (show demo models if no API key)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            models_list = client.models.list()
            anthropic_models = []
            for m in models_list.data:
                anthropic_models.append({
                    "name": m.id,
                    "display_name": m.id.replace("-", " ").title(),
                    "description": f"Anthropic {m.id}",
                    "context_length": 200000,  # Default for Claude
                    "provider": "anthropic",
                    "cost_per_m_tokens": 0
                })
            models["anthropic"] = anthropic_models
        except Exception as e:
            logger.error(f"Failed to fetch Anthropic models: {str(e)}")

    # Always show Anthropic demo models
    if "anthropic" not in models:
        models["anthropic"] = [
            {"name": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "description": "Anthropic Claude 3.5 Sonnet", "context_length": 200000, "provider": "anthropic", "cost_per_m_tokens": 3},
            {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "description": "Anthropic Claude 3 Opus", "context_length": 200000, "provider": "anthropic", "cost_per_m_tokens": 15},
            {"name": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku", "description": "Anthropic Claude 3 Haiku", "context_length": 200000, "provider": "anthropic", "cost_per_m_tokens": 0.25}
        ]

    # 4. XAI (Grok) Models (show demo models if no API key)
    if os.getenv("XAI_API_KEY"):
        try:
            import requests
            response = requests.get(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"},
                timeout=5  # Add timeout
            )
            if response.status_code == 200:
                data = response.json()
                xai_models = []
                for model_info in data.get("data", []):
                    model_name = model_info.get("id", "")
                    xai_models.append({
                        "name": model_name,
                        "display_name": model_name.replace("-", " ").title(),
                        "description": f"xAI {model_name}",
                        "context_length": 1000000,  # Default for Grok
                        "provider": "xai",
                        "cost_per_m_tokens": 0
                    })
                models["xai"] = xai_models
        except Exception as e:
            logger.error(f"Failed to fetch XAI models: {str(e)}")

    # Always show XAI demo models
    if "xai" not in models:
        models["xai"] = [
            {"name": "grok-3", "display_name": "Grok 3", "description": "xAI Grok 3", "context_length": 1000000, "provider": "xai", "cost_per_m_tokens": 0},
            {"name": "grok-4-fast-reasoning", "display_name": "Grok 4 Fast Reasoning", "description": "xAI Grok 4 Fast Reasoning", "context_length": 2000000, "provider": "xai", "cost_per_m_tokens": 0}
        ]

    # 5. Zhipu AI (GLM) Models (show demo models if no API key)
    if os.getenv("ZHIPU_API_KEY"):
        try:
            import requests
            response = requests.get(
                "https://open.bigmodel.cn/api/paas/v4/models",
                headers={"Authorization": f"Bearer {os.getenv('ZHIPU_API_KEY')}"},
                timeout=5  # Add timeout
            )
            if response.status_code == 200:
                data = response.json()
                zhipu_models = []
                for model_info in data.get("data", []):
                    model_name = model_info.get("id", "")
                    zhipu_models.append({
                        "name": model_name,
                        "display_name": model_name.replace("-", " ").title(),
                        "description": f"Zhipu {model_name}",
                        "context_length": 128000,  # Default for GLM
                        "provider": "zhipu",
                        "cost_per_m_tokens": 1
                    })
                models["zhipu"] = zhipu_models
        except Exception as e:
            logger.error(f"Failed to fetch Zhipu models: {str(e)}")

    # Always show Zhipu demo models
    if "zhipu" not in models:
        models["zhipu"] = [
            {"name": "glm-4", "display_name": "GLM-4", "description": "Zhipu GLM-4", "context_length": 128000, "provider": "zhipu", "cost_per_m_tokens": 1},
            {"name": "glm-4-plus", "display_name": "GLM-4 Plus", "description": "Zhipu GLM-4 Plus", "context_length": 128000, "provider": "zhipu", "cost_per_m_tokens": 1}
        ]

    # 6. Local Models (Hugging Face)
    if TRANSFORMERS_AVAILABLE:
        try:
            # Add local models if transformers is available
            local_models = [
                {
                    "name": "ibm-granite/granite-docling-258M",
                    "display_name": "IBM Granite Docling (258M)",
                    "description": "Lightweight document processing model",
                    "context_length": 4096,
                    "provider": "local",
                    "cost_per_m_tokens": 0
                },
                {
                    "name": "microsoft/DialoGPT-medium",
                    "display_name": "Microsoft DialoGPT Medium",
                    "description": "Conversational model for chat",
                    "context_length": 128000,
                    "provider": "local",
                    "cost_per_m_tokens": 0
                },
                {
                    "name": "distilbert-base-uncased-distilled-squad",
                    "display_name": "DistilBERT QA",
                    "description": "Question answering model",
                    "context_length": 384,
                    "provider": "local",
                    "cost_per_m_tokens": 0
                }
            ]
            models["local"] = local_models
            logger.info(f"Added {len(local_models)} local models to available models")
        except Exception as e:
            logger.error(f"Failed to initialize local models: {str(e)}")

    return models

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