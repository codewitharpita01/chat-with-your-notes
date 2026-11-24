"""Application settings sourced from environment variables."""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DATA_DIR / "notes.db"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "chunks.faiss"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

