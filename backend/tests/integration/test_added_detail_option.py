"""The added-detail rule is optional (off by default since 2026-11-15).

On the protected test claims the rule did not help (macro-F1 .409 -> .388, three more correct
claims flagged, no change in false "supported"), so it was made opt-in:
GROUNDEDAI_CLAIM_DETAIL_CHECK=on (or "v1", the name recorded in earlier reports) turns it on.
The implementation is unchanged.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from groundedai.core.config import Settings
from groundedai.generation.providers import GenerationProvider
from groundedai.schemas.query import AnswerResponse, QueryRequest
from groundedai.services.container import build_services
from tests.conftest import SettingsFactory

FERRY = """Harbourline Ferries

A single adult ticket from Kessport to Mirrow costs 14 euros and includes a seat. Children under 5
travel free on board.
"""
# the same statement as in test_reliability_v3.py: it matches the passage, but "hot meal" is in
# no passage
REPLY = "A single adult ticket to Mirrow costs 14 euros and includes a free hot meal on board [S1]."


def _answer(settings: Settings) -> AnswerResponse:
    model = FakeListChatModel(responses=[REPLY])
    services = build_services(
        settings, generation_provider=GenerationProvider("test", "fake", True, model)
    )
    try:
        services.ingestion.ingest_bytes("ferry.txt", "text/plain", FERRY.encode())
        return services.qa.answer(QueryRequest(question="How much is an adult ticket to Mirrow?"))
    finally:
        services.close()


def test_a_fresh_installation_has_the_rule_off() -> None:
    settings = Settings(_env_file=None)  # no environment variables (see conftest), no .env
    assert settings.claim_detail_check == "off"
    assert settings.claim_detail_check_enabled is False


@pytest.mark.parametrize(("value", "enabled"), [("on", True), ("v1", True), ("off", False)])
def test_the_environment_variable_switches_it(
    value: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROUNDEDAI_CLAIM_DETAIL_CHECK", value)
    assert Settings(_env_file=None).claim_detail_check_enabled is enabled


def test_other_values_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDEDAI_CLAIM_DETAIL_CHECK", "yes")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_by_default_an_added_detail_does_not_change_the_lexical_result(
    make_settings: SettingsFactory,
) -> None:
    response = _answer(make_settings())
    assert response.status == "answered"
    claim = response.claims[0]
    assert claim.status == "supported"
    assert claim.added_details == []
    assert "added_detail" not in claim.flags
    assert "added_details" not in response.checks_run
    assert {"citation_labels", "lexical_claims", "relevance"} <= set(response.checks_run)


def test_when_switched_on_the_rule_works_as_before(make_settings: SettingsFactory) -> None:
    response = _answer(make_settings(claim_detail_check="on"))
    claim = response.claims[0]
    assert claim.status == "partial"
    assert "added_detail" in claim.flags
    assert claim.added_details == ["hot meal"]
    assert "added_details" in response.checks_run
