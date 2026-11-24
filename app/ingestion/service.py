"""Extract PDF text, create chunks, embed them, and persist the results."""

import logging
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pymupdf as fitz

from app.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    MAX_UPLOAD_MB,
)
from app.db import connect_db, initialize_database
from app.ingestion.chunking import chunk_page_text

logger = logging.getLogger(__name__)


class IngestionError(ValueError):
    """Raised when a PDF cannot be safely processed."""


class IngestionService:
    """Own the local model and FAISS index used by the ingestion pipeline."""

    def __init__(self) -> None:
        # Keep the CLI importable even before the large ML dependencies are installed.
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _load_models(self) -> None:
        # Models load only when the first valid PDF is ingested.
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model: %s", EMBEDDING_MODEL)
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                # MiniLM normally truncates at 256 tokens; the BERT backbone supports
                # up to 512, so allow our requested ~500-token chunks to be embedded.
                self._model.max_seq_length = min(CHUNK_SIZE_TOKENS + 2, 512)
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        except ImportError as exc:
            raise IngestionError(
                "Embedding packages are missing. Activate the project environment and "
                "run 'python -m pip install -r requirements.txt'."
            ) from exc

    def _read_index(self, dimension: int) -> faiss.IndexIDMap2:
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_INDEX_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            if index.d != dimension:
                raise IngestionError("Saved vector index does not match the embedding model")
            if not isinstance(index, faiss.IndexIDMap2):
                raise IngestionError("Saved vector index has an unsupported format")
            return index
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def ingest_pdf(self, pdf_path: Path, original_name: str) -> dict[str, object]:
        """Ingest one PDF and return document metadata and chunk count."""
        if Path(original_name).suffix.lower() != ".pdf":
            raise IngestionError("Only PDF files are supported")
        if not pdf_path.is_file():
            raise IngestionError("Uploaded file could not be found")
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise IngestionError("The uploaded PDF is empty")
        if file_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise IngestionError(f"PDF exceeds the {MAX_UPLOAD_MB} MB upload limit")

        try:
            with fitz.open(pdf_path) as pdf:
                if pdf.is_encrypted:
                    raise IngestionError("Password-protected PDFs are not supported")
                pages = [(number + 1, page.get_text("text")) for number, page in enumerate(pdf)]
        except IngestionError:
            raise
        except (fitz.FileDataError, RuntimeError) as exc:
            raise IngestionError("The uploaded file is not a readable PDF") from exc

        self._load_models()
        assert self._tokenizer is not None
        records: list[tuple[int, int, str]] = []
        for page_number, page_text in pages:
            for chunk_index, content in enumerate(
                chunk_page_text(
                    page_text,
                    self._tokenizer,
                    chunk_size=CHUNK_SIZE_TOKENS,
                    overlap=CHUNK_OVERLAP_TOKENS,
                )
            ):
                records.append((page_number, chunk_index, content))

        if not records:
            raise IngestionError("No extractable text was found in this PDF")

        vectors = self._model.encode(
            [record[2] for record in records],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        document_id = str(uuid.uuid4())
        initialize_database()

        # Store metadata first so FAISS IDs are stable SQLite chunk IDs.
        with connect_db() as connection:
            connection.execute(
                "INSERT INTO documents (id, name, file_size_bytes, page_count) VALUES (?, ?, ?, ?)",
                (document_id, Path(original_name).name, file_size, len(pages)),
            )
            chunk_ids: list[int] = []
            for page_number, chunk_index, content in records:
                cursor = connection.execute(
                    "INSERT INTO chunks (document_id, page_number, chunk_index, content) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, page_number, chunk_index, content),
                )
                chunk_ids.append(int(cursor.lastrowid))

        try:
            index = self._read_index(vectors.shape[1])
            index.add_with_ids(vectors, np.asarray(chunk_ids, dtype="int64"))
            faiss.write_index(index, str(FAISS_INDEX_PATH))
        except Exception:
            # Avoid leaving metadata that claims to be searchable if indexing failed.
            with connect_db() as connection:
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            logger.exception("Failed to update FAISS index for %s", original_name)
            raise IngestionError("Could not save the local vector index")

        logger.info("Ingested %s: %d pages, %d chunks", original_name, len(pages), len(records))
        return {
            "id": document_id,
            "name": Path(original_name).name,
            "page_count": len(pages),
            "chunk_count": len(records),
        }


ingestion_service = IngestionService()
