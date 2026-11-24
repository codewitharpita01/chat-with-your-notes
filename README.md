# Chat With Your Notes

A local-first study assistant for lecture PDFs. The project is being built in steps:
PDF ingestion and indexing, retrieval, cited answers, the Streamlit interface, quizzes,
and retrieval evaluation.

## Architecture

```text
PDF upload -> PyMuPDF page text -> tokenizer-sized chunks -> SQLite metadata
                                                \-> MiniLM embeddings -> FAISS
```

SQLite stores document and chunk metadata; FAISS stores normalized vectors keyed by
SQLite chunk IDs. Page-local chunks preserve citation page numbers.

## Step 1: ingestion

1. Create a Python 3.11 virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and adjust settings if needed.
4. Ingest one local PDF with `python scripts/ingest_pdf.py path/to/lecture.pdf`.
   This command writes metadata to `data/notes.db` and vectors to `data/chunks.faiss`.
5. Run chunking tests with `python -m pytest tests/test_chunking.py`.

The first ingestion downloads the tokenizer and embedding model from Hugging Face.
They then run locally. Uploads are limited to 20 MB by default.

## Screenshots

Screenshots will be added after the Streamlit interface is implemented.
