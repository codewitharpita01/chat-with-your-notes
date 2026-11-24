"""Small local browser interface for trying the ingestion step."""

import logging
import tempfile
from pathlib import Path

import streamlit as st

from app.db import connect_db, initialize_database
from app.ingestion.service import IngestionError, ingestion_service


logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Chat With Your Notes", page_icon="📚")
st.title("Chat With Your Notes")
st.caption("Step 1 preview: upload lecture PDFs and build your local notes index.")

initialize_database()

uploaded_files = st.file_uploader(
    "Choose one or more lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process PDFs", type="primary"):
    for uploaded_file in uploaded_files:
        # Save each upload to a temporary path because PyMuPDF reads from disk.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)

        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = ingestion_service.ingest_pdf(temporary_path, uploaded_file.name)
            st.success(
                f"Added {result['name']}: {result['page_count']} pages, "
                f"{result['chunk_count']} chunks."
            )
        except IngestionError as error:
            st.error(f"Could not process {uploaded_file.name}: {error}")
        except Exception:
            logging.exception("Unexpected ingestion error for %s", uploaded_file.name)
            st.error(f"Unexpected error while processing {uploaded_file.name}.")
        finally:
            temporary_path.unlink(missing_ok=True)

st.subheader("Indexed documents")
with connect_db() as connection:
    documents = connection.execute(
        "SELECT id, name, page_count, created_at FROM documents ORDER BY created_at DESC"
    ).fetchall()

if documents:
    st.dataframe(
        [dict(document) for document in documents],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No PDFs indexed yet.")
