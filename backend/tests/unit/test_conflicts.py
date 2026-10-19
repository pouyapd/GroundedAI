from groundedai.generation.conflicts import find_figure_conflicts
from groundedai.generation.context import assemble_context
from groundedai.retrieval.types import ChunkScores, RetrievedChunk


def chunk(index: int, document_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        f"c{index}", document_id, "Doc", 0, text, 0, len(text), None, None, ChunkScores()
    )


def conflicts(question: str, *chunks: RetrievedChunk) -> list[str]:
    context = assemble_context(list(chunks), max_chars=6000)
    return [item.warning() for item in find_figure_conflicts(question, context.sources)]


def test_different_figures_in_two_documents_are_reported() -> None:
    found = conflicts(
        "Within how many days must a delay claim be made?",
        chunk(1, "doc_a", "Claims must be made within 28 days of the date of travel."),
        chunk(2, "doc_b", "Claims must be submitted within 14 days of the date of travel."),
    )
    assert found == [
        "Sources S1 and S2 state different figures (28 vs 14) in sentences about the same "
        "topic. They may disagree; check both."
    ]


def test_figures_from_one_document_are_not_a_conflict() -> None:
    assert not conflicts(
        "What are the meal allowance and the receipt threshold?",
        chunk(1, "doc_a", "The daily meal allowance is EUR 45."),
        chunk(2, "doc_a", "Receipts are required for any meal expense above EUR 25."),
    )


def test_unrelated_sentences_are_not_paired() -> None:
    assert not conflicts(
        "How many guests can a member bring per day?",
        chunk(1, "doc_a", "Members may bring up to two guests per day at no charge."),
        chunk(2, "doc_b", "Mail is held at reception for 14 days."),
    )


def test_agreeing_sources_are_not_a_conflict() -> None:
    assert not conflicts(
        "What is the hotel limit per night?",
        chunk(1, "doc_a", "The hotel limit is EUR 140 per night."),
        chunk(2, "doc_b", "Hotel nights are reimbursed up to EUR 140 per night."),
    )


def test_conflicts_written_in_words_are_not_detected() -> None:
    """A documented limitation: only figures written as digits are compared."""
    assert not conflicts(
        "How many times can an item be renewed?",
        chunk(1, "doc_a", "Items can be renewed up to three times online."),
        chunk(2, "doc_b", "You can renew an item twice."),
    )


def test_labels_are_ordered_by_source_position() -> None:
    chunks = [chunk(i, f"doc_{i}", f"Filler sentence number {i}.") for i in range(1, 11)]
    chunks[1] = chunk(2, "doc_x", "Delay claims are accepted within 28 days of travel.")
    chunks[9] = chunk(10, "doc_y", "Delay claims are accepted within 14 days of travel.")
    context = assemble_context(chunks, max_chars=6000)
    (found,) = find_figure_conflicts("How long do I have for delay claims?", context.sources)
    assert (found.label_a, found.label_b) == ("S2", "S10")
