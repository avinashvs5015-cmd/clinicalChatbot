# Document RAG Application

A hybrid Retrieval-Augmented Generation (RAG) system that combines exact text matching with vector similarity search for document Q&A using NVIDIA's Llama model.

## Features

- Hybrid search combining exact text matching and semantic similarity
- Persistent vector embeddings using ChromaDB
- Clean, responsive Streamlit interface
- Support for PDF and text documents
- Conversation history with context awareness

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run hybrid_streamlit_app.py
   ```

## Deployment

This application is configured for deployment on Streamlit Cloud. See the deployment guide below.

## Documents

The application processes documents from the `documents/` folder. Currently includes:
- Machine learning guides
- Climate change reports (IPCC)
- Literature excerpts
- Medical/clinical information

## Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **LLM**: NVIDIA Llama 3.3 Nemotron
- **Document Processing**: PyPDF2, python-docx
