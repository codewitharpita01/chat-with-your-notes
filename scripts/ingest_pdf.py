"""Ingest one local PDF from the command line during step 1."""

import argparse
import json
import logging
from pathlib import Path

from app.ingestion.service import IngestionError, ingestion_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a PDF to the local notes index")
    parser.add_argument("pdf", type=Path, help="Path to a lecture PDF")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = ingestion_service.ingest_pdf(args.pdf, args.pdf.name)
    except IngestionError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
