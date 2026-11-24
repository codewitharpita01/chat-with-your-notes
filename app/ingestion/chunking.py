"""Split page text into overlapping tokenizer-sized chunks."""

from typing import Protocol


class OffsetTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: object) -> dict[str, list]: ...


def chunk_page_text(
    text: str,
    tokenizer: OffsetTokenizer,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Create chunks using model-token boundaries while retaining readable text.

    Chunking is performed separately for each PDF page, so a chunk always has
    one unambiguous page number for citations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")
    if not text.strip():
        return []

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []

    chunks: list[str] = []
    stride = chunk_size - overlap
    for start in range(0, len(offsets), stride):
        end = min(start + chunk_size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(offsets):
            break
    return chunks
