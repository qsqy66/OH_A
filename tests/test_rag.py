import numpy as np

from core.rag import chunk_text, cosine_similarity


def test_chunk_text_preserves_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_handles_empty_text():
    assert chunk_text("   ") == []


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert np.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)