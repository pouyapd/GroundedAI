"""Validation of uploaded files before anything is stored or parsed."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

from groundedai.core.errors import EmptyDocument, FileTooLarge, InvalidInput, UnsupportedFileType

MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}
_ACCEPTED_DECLARED_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf", "application/x-pdf", "application/octet-stream"}),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
    ".md": frozenset(
        {"text/markdown", "text/x-markdown", "text/plain", "application/octet-stream"}
    ),
}
MAX_FILENAME_CHARS = 200


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    filename: str
    extension: str
    media_type: str
    data: bytes
    sha256: str


def sanitize_filename(name: str) -> str:
    """Display-safe base name. Never used to build filesystem paths."""
    name = unicodedata.normalize("NFC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip().lstrip(".").strip()
    if not name:
        raise InvalidInput("The uploaded file has no usable name.")
    if len(name) > MAX_FILENAME_CHARS:
        suffix = PurePosixPath(name).suffix[:10]
        name = name[: MAX_FILENAME_CHARS - len(suffix)] + suffix
    return name


def validate_upload(
    filename: str, declared_content_type: str | None, data: bytes, max_bytes: int
) -> ValidatedUpload:
    if len(data) > max_bytes:
        raise FileTooLarge(
            f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.",
            details={"max_bytes": max_bytes},
        )
    if not data:
        raise EmptyDocument("The uploaded file is empty.")

    safe_name = sanitize_filename(filename)
    extension = PurePosixPath(safe_name).suffix.lower()
    if extension not in MEDIA_TYPES:
        raise UnsupportedFileType(details={"allowed_extensions": sorted(MEDIA_TYPES)})

    declared = (declared_content_type or "").split(";")[0].strip().lower()
    if declared and declared not in _ACCEPTED_DECLARED_TYPES[extension]:
        raise UnsupportedFileType("The declared content type does not match the file extension.")

    if extension == ".pdf":
        if b"%PDF-" not in data[:1024]:
            raise UnsupportedFileType("The file has a .pdf extension but is not a PDF.")
    else:
        if b"\x00" in data[:8192]:
            raise UnsupportedFileType("The file looks like binary data, not text.")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnsupportedFileType("Text files must be UTF-8 encoded.") from exc

    return ValidatedUpload(
        filename=safe_name,
        extension=extension,
        media_type=MEDIA_TYPES[extension],
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def document_id_for(sha256: str) -> str:
    """Stable document id derived from the file content."""
    return f"doc_{sha256[:20]}"
