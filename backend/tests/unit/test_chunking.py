import itertools

from groundedai.ingestion.chunking import (
    PAGE_SEPARATOR,
    ChunkingConfig,
    chunk_id_for,
    chunk_pages,
    join_pages,
    page_at,
)
from groundedai.ingestion.parsers import PageText

LONG_TEXT = " ".join(
    f"Sentence number {i} talks about topic {i % 7} with some extra words." for i in range(200)
)


def test_chunks_match_their_source_offsets() -> None:
    pages = [PageText(1, LONG_TEXT)]
    full_text, _ = join_pages(pages)
    chunks = chunk_pages(pages, ChunkingConfig(300, 50))
    assert len(chunks) > 10
    for chunk in chunks:
        assert full_text[chunk.char_start : chunk.char_end] == chunk.text
        assert len(chunk.text) <= 300


def test_chunking_is_deterministic() -> None:
    pages = [PageText(1, LONG_TEXT), PageText(2, LONG_TEXT[:5000])]
    config = ChunkingConfig(400, 60)
    first = chunk_pages(pages, config)
    second = chunk_pages(pages, config)
    assert [(c.text, c.char_start, c.page_start) for c in first] == [
        (c.text, c.char_start, c.page_start) for c in second
    ]


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_pages([PageText(None, LONG_TEXT)], ChunkingConfig(300, 80))
    overlaps = [b.char_start < a.char_end for a, b in itertools.pairwise(chunks)]
    assert all(overlaps)


def test_page_numbers_are_tracked_across_pages() -> None:
    pages = [PageText(1, "a " * 150), PageText(2, "b " * 150), PageText(3, "c " * 150)]
    chunks = chunk_pages(pages, ChunkingConfig(200, 0))
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    covered = {page for chunk in chunks for page in range(chunk.page_start, chunk.page_end + 1)}
    assert covered == {1, 2, 3}


def test_empty_pages_are_skipped_in_offsets() -> None:
    text, spans = join_pages([PageText(1, "first"), PageText(2, ""), PageText(3, "third")])
    assert text == "first" + PAGE_SEPARATOR + "third"
    assert [span.number for span in spans] == [1, 3]


def test_text_documents_have_no_page_numbers() -> None:
    chunks = chunk_pages([PageText(None, "Short text document.")], ChunkingConfig(100, 10))
    assert len(chunks) == 1
    assert chunks[0].page_start is None
    assert chunks[0].page_end is None


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_pages([PageText(1, "   ")], ChunkingConfig(100, 10)) == []


def test_chunk_ids_are_stable_and_sortable() -> None:
    assert chunk_id_for("doc_abc", 7) == "doc_abc:0007"
    assert chunk_id_for("doc_abc", 10) > chunk_id_for("doc_abc", 9)


def test_chunk_hash_depends_only_on_text() -> None:
    a = chunk_pages([PageText(None, "Same text.")], ChunkingConfig(100, 10))[0]
    b = chunk_pages([PageText(5, "Same text.")], ChunkingConfig(100, 10))[0]
    assert a.sha256 == b.sha256


def test_multi_page_chunks_record_where_each_page_starts() -> None:
    pages = [PageText(1, "Intro."), PageText(2, "Tickets cost 14 euros."), PageText(4, "Pets.")]
    (chunk,) = chunk_pages(pages, ChunkingConfig(800, 0))
    assert (chunk.page_start, chunk.page_end) == (1, 4)
    assert [number for _, number in chunk.page_breaks] == [2, 4]
    for offset, number in chunk.page_breaks:
        assert chunk.text[offset:].startswith(pages[[1, 2, 4].index(number)].text)
    tickets = chunk.text.index("Tickets")
    assert page_at(tickets, chunk.page_start, chunk.page_end, chunk.page_breaks) == 2
    assert page_at(0, chunk.page_start, chunk.page_end, chunk.page_breaks) == 1
    assert page_at(chunk.text.index("Pets"), 1, 4, chunk.page_breaks) == 4


def test_page_at_without_breaks() -> None:
    assert page_at(5, 3, 3, None) == 3  # single-page chunk
    assert page_at(5, None, None, None) is None  # text documents have no pages
    assert page_at(5, 1, 2, None) is None  # multi-page chunk indexed before breaks existed
    single = chunk_pages([PageText(1, "Only page.")], ChunkingConfig(800, 0))[0]
    assert single.page_breaks == ()
