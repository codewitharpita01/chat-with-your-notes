"""Unit tests for page-local token chunking."""

import pytest

from app.ingestion.chunking import chunk_page_text


class FakeTokenizer:
    """A deterministic whitespace tokenizer for tests without model downloads."""

    def __call__(self, text: str, **kwargs: object) -> dict[str, list]:
        offsets = []
        start = 0
        for word in text.split():
            start = text.find(word, start)
            offsets.append((start, start + len(word)))
            start += len(word)
        return {"offset_mapping": offsets}


def test_chunk_size_and_overlap() -> None:
    words = [f"w{i}" for i in range(12)]
    chunks = chunk_page_text(" ".join(words), FakeTokenizer(), chunk_size=5, overlap=2)

    assert [chunk.split() for chunk in chunks] == [
        words[0:5],
        words[3:8],
        words[6:11],
        words[9:12],
    ]


def test_empty_page_returns_no_chunks() -> None:
    assert chunk_page_text("  ", FakeTokenizer(), chunk_size=5, overlap=1) == []


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_page_text("some text", FakeTokenizer(), chunk_size=5, overlap=5)
