import pytest

from groundedai.core.errors import EmptyDocument, FileTooLarge, InvalidInput, UnsupportedFileType
from groundedai.ingestion.validation import document_id_for, sanitize_filename, validate_upload

MAX = 1024 * 1024


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\system.ini.txt", "system.ini.txt"),
        ("/absolute/path/report.pdf", "report.pdf"),
        ("  spaced   name.md ", "spaced name.md"),
        ("bad\x00name\x1f.txt", "badname.txt"),
        (".hidden.txt", "hidden.txt"),
    ],
)
def test_sanitize_filename_strips_paths_and_control_chars(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_rejects_empty_names() -> None:
    with pytest.raises(InvalidInput):
        sanitize_filename("../")


def test_sanitize_filename_truncates_but_keeps_extension() -> None:
    name = sanitize_filename("a" * 500 + ".pdf")
    assert len(name) == 200
    assert name.endswith(".pdf")


def test_valid_text_upload() -> None:
    upload = validate_upload("notes.txt", "text/plain; charset=utf-8", b"hello world", MAX)
    assert upload.extension == ".txt"
    assert upload.media_type == "text/plain"
    assert len(upload.sha256) == 64
    assert document_id_for(upload.sha256) == f"doc_{upload.sha256[:20]}"


def test_empty_file_is_rejected() -> None:
    with pytest.raises(EmptyDocument):
        validate_upload("a.txt", "text/plain", b"", MAX)


def test_too_large_file_is_rejected() -> None:
    with pytest.raises(FileTooLarge):
        validate_upload("a.txt", "text/plain", b"x" * (MAX + 1), MAX)


@pytest.mark.parametrize("name", ["script.exe", "page.html", "archive.zip", "noextension"])
def test_unsupported_extensions_are_rejected(name: str) -> None:
    with pytest.raises(UnsupportedFileType):
        validate_upload(name, None, b"data", MAX)


def test_mismatched_declared_type_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType):
        validate_upload("a.txt", "application/pdf", b"hello", MAX)


def test_fake_pdf_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType, match="not a PDF"):
        validate_upload("a.pdf", "application/pdf", b"just text", MAX)


def test_binary_data_with_text_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType, match="binary"):
        validate_upload("a.txt", "text/plain", b"abc\x00def", MAX)


def test_non_utf8_text_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType, match="UTF-8"):
        validate_upload("a.txt", "text/plain", "café".encode("latin-1"), MAX)
