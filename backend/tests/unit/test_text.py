from groundedai.core.text import (
    clean_extracted_text,
    content_tokens,
    extract_numbers,
    find_normalized_span,
    light_stem,
    split_sentences,
    tokenize,
)


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World! E07_x") == ["hello", "world", "e07", "x"]


def test_light_stem_is_consistent_for_plurals_and_verbs() -> None:
    assert light_stem("limits") == light_stem("limit") == "limit"
    assert light_stem("policies") == "policy"
    assert light_stem("booking") == "book"
    assert light_stem("class") == "class"  # 'ss' endings are kept
    assert light_stem("2025") == "2025"


def test_content_tokens_remove_stopwords() -> None:
    assert content_tokens("What is the hotel limit for the night?") == ["hotel", "limit", "night"]


def test_split_sentences_keeps_numbers_and_codes() -> None:
    text = "E07 means fan failure. E09 means over-temperature.\n\nNew paragraph here."
    assert split_sentences(text) == [
        "E07 means fan failure.",
        "E09 means over-temperature.",
        "New paragraph here.",
    ]


def test_extract_numbers_normalises_thousands() -> None:
    assert extract_numbers("10,000 requests and €0.30 per km, 2.5 MB") == ["10000", "0.30", "2.5"]


def test_extract_numbers_ignores_digits_inside_words() -> None:
    assert extract_numbers("Error E07 on SX5 with [S1]") == []


def test_clean_extracted_text_removes_control_characters() -> None:
    assert clean_extracted_text("a\x00b\r\nc\t\td\n\n\n\ne") == "ab\nc d\n\ne"


def test_find_normalized_span_ignores_case_and_whitespace() -> None:
    haystack = "Intro.\nThe DC input\n voltage range is 150 to 850 V. Next."
    span = find_normalized_span(haystack, "the dc input voltage range is 150 to 850 v.")
    assert span is not None
    start, end = span
    assert haystack[start:end].startswith("The DC input")
    assert haystack[start:end].endswith("850 V.")


def test_find_normalized_span_returns_none_when_absent() -> None:
    assert find_normalized_span("abc", "xyz") is None
    assert find_normalized_span("abc", "   ") is None
