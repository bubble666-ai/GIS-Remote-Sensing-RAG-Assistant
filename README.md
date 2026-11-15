# Multi-Model GIS & Remote Sensing RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for **Geographic Information Systems (GIS)** and **Remote Sensing**. Ask technical questions in plain language; the app retrieves relevant passages from your own document library and answers with cited sources, using one of 25+ models from five AI providers.

| | |
|---|---|
| **AI providers** | XAI (Grok) · Google (Gemini) · Anthropic (Claude) · OpenAI (GPT) · Zhipu (GLM) |
| **Local models** | IBM Granite Docling · DialoGPT · DistilBERT QA (offline, via Hugging Face) |
| **Document formats** | PDF · DOCX · TXT · CSV |
| **Databases** | SQLite · PostgreSQL |
| **Interface** | Streamlit web app |
| **License** | GPL-3.0 |

## Features

- **Multi-provider chat** — switch provider and model at runtime from the sidebar; each answer shows the model used and its sources.
- **Document-grounded answers** — files in `data/` are loaded, split into overlapping chunks, and retrieved per query by keyword-overlap scoring. Citations with source and category appear under each response.
- **Database-to-text export** — dump SQLite or PostgreSQL tables to RAG-ready text files and query them conversationally.
- **Optional local inference** — with `transformers` and `torch` installed, local Hugging Face models appear in the provider list for fully offline use.

## Quick Start

**Prerequisites:** Python 3.9+, pip, at least one API key (XAI recommended).

```bash
git clone https://github.com/bubble666-ai/GIS-Remote-Sensing-RAG-Assistant.git
cd GIS-Remote-Sensing-RAG-Assistant
pip install -r requirements.txt

cp .env.example .env        # then add at least one key, e.g. XAI_API_KEY=...
```

Add your material to `data/` (PDF, DOCX, TXT, CSV; subfolders are fine), then:

```bash
python start_app.py         # opens http://localhost:8501
```

In the sidebar, pick a provider and model; adjust temperature, max tokens, and how many sources to retrieve (top-K). The launcher checks which keys are configured and tells you before starting.

> Never commit `.env` — it's gitignored. Keys come from `.env` at startup.

## Sample Questions

- What is NDVI and how is it calculated from Sentinel-2 imagery?
- Which bands does Sentinel-2 use for vegetation analysis?
- Compare the spectral capabilities of Landsat 8 and Sentinel-2.
- Explain atmospheric correction in remote sensing.
- What's the difference between supervised and unsupervised classification?

## Model Selection

| Use case | Model | Why |
|---------|-------|-----|
| General GIS questions | `grok-3` (XAI) | Most capable, real-time data |
| Quick technical answers | `grok-4-fast-reasoning` (XAI) | Fast reasoning |
| Complex analysis | `claude-3-5-sonnet` (Anthropic) | Advanced reasoning |
| Cost-effective | `gpt-4o-mini` / `gpt-3.5-turbo` (OpenAI) | Cheap and fast |
| Long context | `gemini-1.5-pro` (Google) | Up to 2M tokens |
| Offline | Local Hugging Face model | No API key needed |

Model lists refresh from each provider's API when the key is configured; the table above shows sensible defaults otherwise.

## Database Export

```python
from app.app import DatabaseExporter

# SQLite
exporter = DatabaseExporter('sqlite', {'database': 'your_gis_data.db'})
exporter.export_all_tables()          # writes tables to data/ as RAG-ready text

# PostgreSQL (pip install psycopg2-binary)
exporter = DatabaseExporter('postgresql', {
    'host': 'localhost', 'database': 'gis_db',
    'user': 'your_username', 'password': 'your_password', 'port': 5432,
})
exporter.export_all_tables()
```

Exported files are indexed like any other document in `data/`.

## Local Models (offline)

```bash
pip install transformers torch
```

Then select the **local** provider in the sidebar. Available models: `ibm-granite/granite-docling-258M` (document processing), `microsoft/DialoGPT-medium` (conversation), `distilbert-base-uncased-distilled-squad` (QA).

## Project Structure

```
├── app/
│   └── app.py            # The whole application: providers, document loaders,
│   │                     #   chunker/retriever, DB exporter, Streamlit UI
│   └── __init__.py
├── data/                 # Your documents (PDF, DOCX, TXT, CSV) — indexed at startup
├── .env.example          # API key template — copy to .env
├── requirements.txt
├── start_app.py          # Launcher: checks keys, starts Streamlit
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No documents found" | Add PDF/DOCX/TXT/CSV files to `data/` and restart |
| API key not working | Key missing/invalid in `.env`, or out of credit |
| PDF/DOCX files ignored | `pip install PyPDF2 python-docx` |
| PostgreSQL export fails | `pip install psycopg2-binary` |
| Local provider missing | `pip install transformers torch` |
| Cloud model unreachable | Check internet connection; try another provider |

## License

[GPL-3.0](LICENSE)
