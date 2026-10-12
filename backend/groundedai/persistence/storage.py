"""Storage for original uploaded files (needed for safe re-ingestion).

File names are derived only from the server-generated document id and an
allow-listed extension, so user-supplied names can never influence the path.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCUMENT_ID_RE = re.compile(r"^doc_[0-9a-f]{20}$")
ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})


class FileStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, document_id: str, extension: str) -> Path:
        if not DOCUMENT_ID_RE.fullmatch(document_id):
            raise ValueError("invalid document id")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("invalid extension")
        path = (self.root / f"{document_id}{extension}").resolve()
        if path.parent != self.root:  # defence in depth; cannot happen with the checks above
            raise ValueError("path escapes storage root")
        return path

    def save(self, document_id: str, extension: str, data: bytes) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(document_id, extension)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
        return path

    def load(self, document_id: str, extension: str) -> bytes:
        return self._path(document_id, extension).read_bytes()

    def exists(self, document_id: str, extension: str) -> bool:
        return self._path(document_id, extension).is_file()

    def delete(self, document_id: str, extension: str) -> None:
        self._path(document_id, extension).unlink(missing_ok=True)
