from pathlib import Path

import pytest

from groundedai.cli import build_parser, expand_paths, main


def test_expand_paths_includes_supported_files_from_directories(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.md").write_text("# n", encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"MZ")
    (tmp_path / "sub").mkdir()
    single = tmp_path / "single.txt"
    assert expand_paths([str(tmp_path), str(single)]) == [
        tmp_path / "a.pdf",
        tmp_path / "b.txt",
        tmp_path / "notes.md",
        single,
    ]


def test_parser_knows_all_commands() -> None:
    parser = build_parser()
    for argv in (
        ["serve"],
        ["migrate"],
        ["status"],
        ["ingest", "x.txt"],
        ["reindex", "--dry-run"],
        ["ask", "question?"],
        ["retrieve", "query"],
        ["eval", "retrieval"],
        ["eval", "generation", "--provider", "extractive-demo"],
    ):
        assert parser.parse_args(argv).func


def test_ingest_and_ask_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from groundedai.core.config import get_settings

    monkeypatch.setenv("GROUNDEDAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GROUNDEDAI_LOG_LEVEL", "WARNING")
    get_settings.cache_clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.txt").write_text(
        "Policy\n\nThe hotel limit is 140 euros per night in capital cities.", encoding="utf-8"
    )
    try:
        assert main(["ingest", str(corpus)]) == 0
        assert "policy.txt: ready" in capsys.readouterr().out
        assert main(["ask", "What is the hotel limit in capital cities?"]) == 0
        output = capsys.readouterr().out
        assert "[answered]" in output
        assert "DEMO (extractive, not an LLM)" in output
    finally:
        get_settings.cache_clear()


def test_reindex_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from groundedai.core.config import get_settings

    monkeypatch.setenv("GROUNDEDAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GROUNDEDAI_LOG_LEVEL", "WARNING")
    (tmp_path / "policy.txt").write_text("Policy\n\nThe hotel limit is 140 euros.", "utf-8")
    try:
        monkeypatch.setenv("GROUNDEDAI_TEXT_CLEANUP", "v1")
        get_settings.cache_clear()
        assert main(["ingest", str(tmp_path / "policy.txt")]) == 0
        monkeypatch.setenv("GROUNDEDAI_TEXT_CLEANUP", "v2")
        get_settings.cache_clear()
        capsys.readouterr()
        assert main(["reindex", "--dry-run"]) == 0
        output = capsys.readouterr().out
        assert "1 of 1 ready document(s) were indexed with older settings." in output
        assert "text clean-up: v1 -> v2 (1)" in output
        assert main(["reindex"]) == 0
        output = capsys.readouterr().out
        assert "[1/1] Policy: re-indexed (1 -> 1 chunks)" in output
        assert main(["reindex"]) == 0
        assert "0 of 1 ready document(s)" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()
