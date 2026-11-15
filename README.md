# 🌍 Multi-Model GIS & Remote Sensing RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for Geographic Information Systems (GIS) and Remote Sensing. Multi-provider AI (XAI Grok, Google Gemini, Anthropic Claude, OpenAI GPT, Zhipu GLM + optional local Hugging Face models), multi-format document ingestion, and database-to-text export — in a single Streamlit app.

## ✨ Features

- **Multi-provider chat**: 25+ models across 5 cloud providers, switchable at runtime in the sidebar; Grok (XAI) recommended.
- **Document RAG**: drop PDF / DOCX / TXT / CSV files into `data/` — they're loaded, chunked, and retrieved by keyword-overlap scoring with source citations shown under each answer.
- **Database export**: export SQLite/PostgreSQL tables to text files and ask questions about them.
- **Local models (optional)**: Hugging Face models run offline when `transformers` + `torch` are installed.

## 📁 Project Structure

```
├── app/
│   └── app.py            # The whole application (providers, loaders, retriever, UI)
├── data/                 # Your documents (PDF, DOCX, TXT, CSV)
├── .env.example          # Copy to .env and add your API keys
├── requirements.txt
├── start_app.py          # Launcher: checks keys, starts Streamlit
└── README.md
```

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/bubble666-ai/GIS-Remote-Sensing-RAG-Assistant.git
cd GIS-Remote-Sensing-RAG-Assistant
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add at least one key (XAI recommended):
# XAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ZHIPU_API_KEY=your_key_here
```

Never commit your `.env` — it is gitignored.

### 3. Add documents

Copy your GIS / remote-sensing material into `data/` (PDF, DOCX, TXT, CSV; subfolders welcome). The app indexes everything in `data/` at startup.

### 4. Run

```bash
python start_app.py
```

Opens at `http://localhost:8501`. Pick a provider + model in the sidebar, ask questions, and use the sliders to tune temperature, max tokens, and top-K retrieved sources.

## 🗄️ Database export (optional)

```python
from app.app import DatabaseExporter

# SQLite
exporter = DatabaseExporter('sqlite', {'database': 'your_gis_data.db'})
exporter.export_all_tables()   # writes tables to data/ as RAG-ready text

# PostgreSQL (pip install psycopg2-binary)
exporter = DatabaseExporter('postgresql', {
    'host': 'localhost', 'database': 'gis_db',
    'user': 'your_username', 'password': 'your_password', 'port': 5432,
})
exporter.export_all_tables()
```

## 💻 Local models (optional, offline)

```bash
pip install transformers torch
```

Local Hugging Face models (e.g. `ibm-granite/granite-docling-258M`, `microsoft/DialoGPT-medium`) then appear in the provider list.

## 💡 Sample questions

- "What is NDVI and how is it calculated from Sentinel-2 imagery?"
- "Which bands does Sentinel-2 use for vegetation analysis?"
- "Compare the spectral capabilities of Landsat 8 and Sentinel-2."
- "Explain atmospheric correction in remote sensing."

## 🤖 Model selection guide

| Use case | Model | Why |
|---------|-------|-----|
| General GIS questions | `grok-3` (XAI) | Most capable, real-time data |
| Quick technical answers | `grok-4-fast-reasoning` (XAI) | Fast reasoning |
| Complex analysis | `claude-3-5-sonnet` | Advanced reasoning |
| Cost-effective | `gpt-3.5-turbo` | Cheap and fast |
| Offline | local HF model | No API needed |

## 🔍 Troubleshooting

- **"No documents found"** — add supported files (PDF, DOCX, TXT, CSV) to `data/` and restart.
- **"API key not working"** — check the key is set in `.env`, is active, and has credit.
- **PDF/DOCX ignored** — `pip install PyPDF2 python-docx`.
- **PostgreSQL export fails** — `pip install psycopg2-binary`.

## 📄 License

GPL-3.0 — see [LICENSE](LICENSE).
