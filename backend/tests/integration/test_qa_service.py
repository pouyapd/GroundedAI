import threading
import time
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from sqlalchemy import select

from groundedai.core.config import Settings
from groundedai.core.errors import (
    GenerationFailed,
    InvalidInput,
    NotFound,
    ProviderTimeout,
    ProviderUnavailable,
    ServerBusy,
)
from groundedai.generation.providers import GenerationProvider
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import QueryEventRow
from groundedai.schemas.query import QueryRequest
from groundedai.services.container import Services, build_services
from tests.conftest import INJECTION_TEXT, TRAVEL_TEXT, SettingsFactory


def _services_with(settings: Settings, chat_model: BaseChatModel) -> Services:
    provider = GenerationProvider(name="test", model="fake", is_llm=True, chat_model=chat_model)
    services = build_services(settings, generation_provider=provider)
    services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    return services


def _events(services: Services) -> list[QueryEventRow]:
    with session_scope(services.session_factory) as session:
        return list(session.scalars(select(QueryEventRow)).all())


def test_extractive_demo_answer_is_grounded_and_labelled(services: Services) -> None:
    services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    response = services.qa.answer(
        QueryRequest(
            question="What is the hotel limit in capital cities?", include_diagnostics=True
        )
    )
    assert response.status == "answered"
    assert response.provider.is_llm is False
    assert response.grounding == "grounded"
    assert response.citation_check.all_valid
    cited = [source for source in response.sources if source.cited]
    assert cited
    assert "140" in cited[0].excerpt
    assert response.retrieval is not None
    assert response.usage.estimated_cost_usd == 0.0
    event = _events(services)[0]
    assert event.status == "answered"
    assert event.question_chars > 0
    assert event.citations_valid is True


def test_abstains_without_calling_a_model_when_nothing_is_indexed(settings: Settings) -> None:
    calls: list[int] = []

    class NeverCalled(FakeListChatModel):
        def _call(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - must not run
            calls.append(1)
            return ""

    services = build_services(
        settings,
        generation_provider=GenerationProvider("test", "x", True, NeverCalled(responses=["x"])),
    )
    try:
        response = services.qa.answer(QueryRequest(question="Anything at all?"))
        assert response.status == "insufficient_evidence"
        assert response.abstention_reason == "No indexed passage matched the question."
        assert response.timings.generation_ms is None
        assert calls == []
    finally:
        services.close()


def test_invalid_citation_is_reported(settings: Settings) -> None:
    model = FakeListChatModel(responses=["The hotel limit is EUR 140 per night [S1]. See [S9]."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "answered"
        assert response.citation_check.invalid == ["S9"]
        assert not response.citation_check.all_valid
        assert any("not provided" in warning for warning in response.warnings)
    finally:
        services.close()


def test_hallucinated_number_is_flagged(settings: Settings) -> None:
    model = FakeListChatModel(responses=["The hotel limit is EUR 999 per night [S1]."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.citation_check.all_valid
        assert response.sentences[0].support == "unsupported"
        assert response.sentences[0].unsupported_numbers == ["999"]
        assert response.grounding == "ungrounded"
    finally:
        services.close()


def test_empty_model_output_is_invalid(settings: Settings) -> None:
    services = _services_with(settings, FakeListChatModel(responses=["   "]))
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "invalid_output"
        assert response.answer is None
        assert response.sources  # retrieved evidence is still shown
    finally:
        services.close()


def test_citation_markers_without_text_are_invalid_output(settings: Settings) -> None:
    # Regression: real Qwen2.5-0.5B outputs such as "[S1][S2]" were scored as answers.
    services = _services_with(settings, FakeListChatModel(responses=["[S1][S2]"]))
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "invalid_output"
        assert response.answer is None
        assert not any(source.cited for source in response.sources)
        assert any("without any answer text" in warning for warning in response.warnings)
    finally:
        services.close()


class RecordingChatModel(BaseChatModel):
    """Returns a fixed reply and keeps the messages it was sent."""

    reply: str = "The hotel limit is EUR 140 per night [S1]."
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


FAQ_TEXT = """Finance FAQ

How much can I spend on a hotel? Hotel nights are reimbursed up to EUR 120 per night \
in all cities.
"""


@pytest.mark.parametrize("version", ["answer-v1", "answer-v2"])
def test_documents_reach_the_model_only_as_data(make_settings, version: str) -> None:  # type: ignore[no-untyped-def]
    model = RecordingChatModel(seen=[])
    services = _services_with(make_settings(prompt_version=version), model)
    services.ingestion.ingest_bytes("supplier.txt", None, INJECTION_TEXT.encode())
    try:
        services.qa.answer(QueryRequest(question="What are the payment terms for suppliers?"))
        system, human = model.seen[0]
        assert system.type == "system"
        assert "Ignore" not in str(system.content)  # document text never enters the system message
        assert "untrusted data" in str(human.content)
    finally:
        services.close()


def test_conflicting_figures_are_flagged_and_noted_for_the_model(make_settings) -> None:  # type: ignore[no-untyped-def]
    model = RecordingChatModel(seen=[])
    services = _services_with(make_settings(prompt_version="answer-v2"), model)
    services.ingestion.ingest_bytes("faq.txt", None, FAQ_TEXT.encode())
    try:
        response = services.qa.answer(
            QueryRequest(question="What is the hotel limit per night in capital cities?")
        )
        (conflict,) = response.conflicts
        assert conflict.kind == "conflict"
        assert sorted(value for s in conflict.statements for value in s.values) == [
            "EUR 120",
            "EUR 140",
        ]
        for statement in conflict.statements:
            source = next(item for item in response.sources if item.label == statement.label)
            assert source.text[statement.quote_start : statement.quote_end] == statement.quote
        assert conflict.answer_reports == ["S1"] or conflict.answer_reports == ["S2"]
        assert any("disagree" in warning for warning in response.warnings)
        assert any("gives only the value from" in warning for warning in response.warnings)
        human = str(model.seen[0][1].content)
        assert "Application note: sources [S" in human
        assert "state different figures" in human
        assert human.index("Application note") < human.index("<question>")
    finally:
        services.close()


def test_the_published_figure_check_can_still_be_selected(make_settings) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(prompt_version="answer-v2", conflict_check="figures-v1")
    model = RecordingChatModel(seen=[])
    services = _services_with(settings, model)
    services.ingestion.ingest_bytes("faq.txt", None, FAQ_TEXT.encode())
    try:
        response = services.qa.answer(
            QueryRequest(question="What is the hotel limit per night in capital cities?")
        )
        assert response.conflicts == []
        assert any("state different figures" in warning for warning in response.warnings)
        assert "state different figures" in str(model.seen[0][1].content)
    finally:
        services.close()


def test_an_answer_reporting_both_sides_gets_no_one_sided_warning(make_settings) -> None:  # type: ignore[no-untyped-def]
    reply = "The travel policy says EUR 140 [S1], but the FAQ says EUR 120 [S2]."
    model = RecordingChatModel(seen=[], reply=reply)
    services = _services_with(make_settings(prompt_version="answer-v2"), model)
    services.ingestion.ingest_bytes("faq.txt", None, FAQ_TEXT.encode())
    try:
        response = services.qa.answer(
            QueryRequest(question="What is the hotel limit per night in capital cities?")
        )
        (conflict,) = response.conflicts
        assert sorted(conflict.answer_reports) == ["S1", "S2"]
        assert not any("gives only the value" in warning for warning in response.warnings)
    finally:
        services.close()


CONTRACTOR_POLICY = """Contractor policy

Contractors must complete the security training before their first day.
"""
CONTRACTOR_FAQ = """Contractor FAQ

Contractors do not have to complete the security training.
"""


def test_a_requirement_and_its_denial_are_both_named(make_settings) -> None:  # type: ignore[no-untyped-def]
    # Regression: the side stating the requirement had no value ("S1 says , S2 says ...").
    model = RecordingChatModel(seen=[], reply="Contractors must complete the training [S1].")
    services = _services_with(make_settings(prompt_version="answer-v2"), model)
    services.ingestion.ingest_bytes("policy.txt", None, CONTRACTOR_POLICY.encode())
    services.ingestion.ingest_bytes("faq.txt", None, CONTRACTOR_FAQ.encode())
    try:
        response = services.qa.answer(
            QueryRequest(question="Do contractors have to complete the security training?")
        )
        (conflict,) = response.conflicts
        assert conflict.kind == "conflict"
        first, second = conflict.statements
        requires = next(s for s in conflict.statements if "must" in s.quote)
        denies = next(s for s in conflict.statements if "do not have to" in s.quote)
        assert (requires.values, requires.affirms_requirement) == ([], True)
        assert (denies.values, denies.denies_requirement, denies.affirms_requirement) == (
            [],
            True,
            False,
        )
        said = {requires.label: "it is required", denies.label: "there is no such requirement"}
        assert conflict.explanation == (
            f"Sources {first.label} and {second.label} disagree: {first.label} says "
            f"{said[first.label]}, {second.label} says {said[second.label]}. GroundedAI does "
            "not decide which is right; check both documents."
        )
        assert conflict.explanation in response.warnings
        body = response.model_dump(mode="json")["conflicts"][0]["statements"]
        assert {s["label"]: s["affirms_requirement"] for s in body} == {
            requires.label: True,
            denies.label: False,
        }
        assert "disagree. If both answer the question, report what each says" in str(
            model.seen[0][1].content
        )
    finally:
        services.close()


def test_flagged_sources_are_named_in_a_note_to_the_model(make_settings) -> None:  # type: ignore[no-untyped-def]
    model = RecordingChatModel(seen=[])
    services = _services_with(make_settings(prompt_version="answer-v2"), model)
    services.ingestion.ingest_bytes("supplier.txt", None, INJECTION_TEXT.encode())
    try:
        services.qa.answer(QueryRequest(question="What are the payment terms for suppliers?"))
        human = str(model.seen[0][1].content)
        assert "looks like instructions to an AI system" in human
        assert human.index("Application note") < human.index("<question>")
    finally:
        services.close()


def test_answer_v1_prompt_is_unchanged_by_the_conflict_check(make_settings) -> None:  # type: ignore[no-untyped-def]
    model = RecordingChatModel(seen=[])
    services = _services_with(make_settings(prompt_version="answer-v1"), model)
    services.ingestion.ingest_bytes("faq.txt", None, FAQ_TEXT.encode())
    try:
        response = services.qa.answer(
            QueryRequest(question="What is the hotel limit per night in capital cities?")
        )
        assert not any("state different figures" in warning for warning in response.warnings)
        assert "Application note" not in str(model.seen[0][1].content)
    finally:
        services.close()


def test_model_abstention_is_recognised(settings: Settings) -> None:
    model = FakeListChatModel(responses=["INSUFFICIENT_EVIDENCE: nothing about contractors."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="contractor leave days hotel"))
        assert response.status == "insufficient_evidence"
        assert response.abstention_reason == "nothing about contractors."
    finally:
        services.close()


class FailingModel(BaseChatModel):
    error: Exception

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("refused"), ProviderUnavailable),
        (httpx.ReadTimeout("slow"), ProviderTimeout),
        (RuntimeError("boom"), GenerationFailed),
    ],
)
def test_provider_errors_map_to_api_errors(
    settings: Settings, error: Exception, expected: type[Exception]
) -> None:
    services = _services_with(settings, FailingModel(error=error))
    try:
        with pytest.raises(expected):
            services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        event = _events(services)[-1]
        assert event.status == "error"
        assert event.error_code
    finally:
        services.close()


class SlowModel(BaseChatModel):
    delay: float = 1.0
    started: Any = None

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.started is not None:
            self.started.set()
        time.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="late [S1]."))])


def test_generation_timeout(make_settings) -> None:  # type: ignore[no-untyped-def]
    services = _services_with(make_settings(generation_timeout_s=0.2), SlowModel(delay=1.0))
    try:
        with pytest.raises(ProviderTimeout):
            services.qa.answer(QueryRequest(question="hotel limit capital cities"))
    finally:
        services.close()


def test_concurrency_limit_returns_server_busy(make_settings) -> None:  # type: ignore[no-untyped-def]
    started = threading.Event()
    settings = make_settings(max_concurrent_generations=1, generation_queue_timeout_s=0.1)
    services = _services_with(settings, SlowModel(delay=1.0, started=started))
    errors: list[Exception] = []

    def first() -> None:
        try:
            services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        except Exception as exc:  # pragma: no cover - not expected
            errors.append(exc)

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert started.wait(5)
        with pytest.raises(ServerBusy):
            services.qa.answer(QueryRequest(question="hotel limit capital cities"))
    finally:
        thread.join()
        services.close()
    assert errors == []


def test_injected_instructions_produce_a_warning(services: Services) -> None:
    services.ingestion.ingest_bytes("supplier.txt", None, INJECTION_TEXT.encode())
    response = services.qa.answer(
        QueryRequest(question="What are the payment terms for suppliers?")
    )
    assert response.status == "answered"
    assert "45 days" in (response.answer or "")
    assert "PWNED" not in (response.answer or "")
    flagged = [s for s in response.sources if s.injection_flags]
    assert flagged
    assert any("instruction-like" in warning for warning in response.warnings)


def test_question_validation(services: Services) -> None:
    with pytest.raises(InvalidInput):
        services.qa.answer(QueryRequest(question=" a "))
    with pytest.raises(InvalidInput):
        services.qa.answer(QueryRequest(question="x" * 1001))


def test_unknown_document_filter(services: Services) -> None:
    with pytest.raises(NotFound):
        services.qa.answer(QueryRequest(question="hotel limit", document_ids=["doc_" + "f" * 20]))


def test_question_text_is_not_stored(services: Services) -> None:
    services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    secret = "What is the hotel limit for project NIGHTJAR?"
    services.qa.answer(QueryRequest(question=secret))
    event = _events(services)[0]
    assert "NIGHTJAR" not in repr(vars(event))


# ---------------------------------------------------------------- claim check
def test_uncited_answer_gets_the_passage_that_supports_it(settings: Settings) -> None:
    model = FakeListChatModel(responses=["The hotel limit is EUR 140 per night in capital cities."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.grounding == "ungrounded"  # the strict [S#] check is unchanged
        (claim,) = response.claims
        assert claim.status == "added_source"
        evidence = claim.evidence[0]
        source = next(s for s in response.sources if s.label == evidence.label)
        assert source.text[evidence.quote_start : evidence.quote_end] == evidence.quote
        assert "EUR 140" in evidence.quote
        assert evidence.document_title == source.document_title
        assert source.supports_claims == [0]
        assert response.verification is not None
        assert response.verification.verdict == "verified"
        assert response.verification.added_source == 1
        assert any("gave no citation" in warning for warning in response.warnings)
    finally:
        services.close()


def test_variant_citation_is_resolved_and_counted(settings: Settings) -> None:
    model = FakeListChatModel(responses=["[Source 1]: The hotel limit is EUR 140 per night."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.citation_check.cited == []  # not a valid [S#] citation
        assert [claim.status for claim in response.claims] == ["supported"]
        assert response.claims[0].citations == ["S1"]
        assert response.verification is not None
        assert response.verification.references_variant == 1
    finally:
        services.close()


def test_invented_figure_makes_the_answer_unverified(settings: Settings) -> None:
    model = FakeListChatModel(responses=["The hotel limit is EUR 999 per night [S1]."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "answered"  # default: label, do not withhold
        assert response.claims[0].status == "unsupported"
        assert response.claims[0].missing_figures == ["EUR 999"]
        assert response.verification is not None
        assert response.verification.verdict == "unverified"
        assert any("Treat the answer as unsupported" in w for w in response.warnings)
    finally:
        services.close()


def test_unverified_answer_can_be_withheld(make_settings: SettingsFactory) -> None:
    settings = make_settings(unverified_answers="withhold")
    model = FakeListChatModel(responses=["The hotel limit is EUR 999 per night [S1]."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "insufficient_evidence"
        assert response.answer is None
        assert response.unverified_answer == "The hotel limit is EUR 999 per night [S1]."
        assert "withheld" in (response.abstention_reason or "")
        assert response.claims[0].status == "unsupported"
    finally:
        services.close()


def test_supported_answer_is_not_withheld(make_settings: SettingsFactory) -> None:
    settings = make_settings(unverified_answers="withhold")
    model = FakeListChatModel(responses=["The hotel limit is EUR 140 per night [S1]."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="hotel limit capital cities"))
        assert response.status == "answered"
        assert response.unverified_answer is None
    finally:
        services.close()


def test_abstention_has_no_claims(settings: Settings) -> None:
    model = FakeListChatModel(responses=["INSUFFICIENT_EVIDENCE: nothing about contractors."])
    services = _services_with(settings, model)
    try:
        response = services.qa.answer(QueryRequest(question="contractor leave days hotel"))
        assert response.claims == []
        assert response.verification is None
    finally:
        services.close()
