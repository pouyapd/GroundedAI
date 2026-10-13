"""Evaluation reports (published, measured offline) and API-triggered runs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query

from groundedai.api.deps import ServicesDep, rate_limit, require_api_key
from groundedai.core.errors import NotFound
from groundedai.evaluation.reporting import report_prompt_version
from groundedai.persistence.db import session_scope
from groundedai.persistence.repositories import EvaluationRunRepository
from groundedai.schemas.evaluation import (
    EvaluationRunCreate,
    EvaluationRunDetail,
    EvaluationRunList,
    EvaluationRunOut,
    ReportList,
    ReportListItem,
)

router = APIRouter(
    prefix="/evaluations", tags=["evaluation"], dependencies=[Depends(require_api_key)]
)


@router.get("/reports", response_model=ReportList, summary="Published benchmark reports")
def list_reports(services: ServicesDep) -> ReportList:
    items = []
    for report in services.evaluation.published_reports():
        dataset = report.get("dataset") or {}
        items.append(
            ReportListItem(
                id=str(report["report_id"]),
                kind=str(report["kind"]),
                title=str(report.get("title", report["report_id"])),
                generated_at=report.get("generated_at"),
                dataset=dataset.get("name") if isinstance(dataset, dict) else None,
                n_configurations=len(report.get("configurations", []))
                if report["kind"] == "retrieval"
                else 1,
                prompt_version=report_prompt_version(report),
            )
        )
    return ReportList(items=items)


@router.get(
    "/reports/{report_id}",
    summary="A full published report (JSON)",
    responses={404: {"description": "Unknown report"}},
)
def get_report(
    services: ServicesDep,
    report_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9._-]{1,120}$")],
) -> dict[str, Any]:
    return services.evaluation.published_report(report_id)


@router.post(
    "/runs",
    status_code=202,
    response_model=EvaluationRunOut,
    summary="Evaluate this instance's retrieval configuration on the bundled dataset",
    responses={
        409: {"description": "A run is already in progress"},
        429: {"description": "Rate limited"},
    },
    dependencies=[Depends(rate_limit("evaluation"))],
)
def create_run(
    services: ServicesDep, background: BackgroundTasks, body: EvaluationRunCreate | None = None
) -> EvaluationRunOut:
    row = services.evaluation.create_run()
    background.add_task(services.evaluation.execute_run, row.id)
    return EvaluationRunOut.model_validate(row)


@router.get("/runs", response_model=EvaluationRunList, summary="Evaluation run history")
def list_runs(
    services: ServicesDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EvaluationRunList:
    with session_scope(services.session_factory) as session:
        rows, total = EvaluationRunRepository(session).page(offset=offset, limit=limit)
        items = [EvaluationRunOut.model_validate(row) for row in rows]
    return EvaluationRunList(items=items, total=total, offset=offset, limit=limit)


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRunDetail,
    summary="One evaluation run including its report",
    responses={404: {"description": "Unknown run"}},
)
def get_run(
    services: ServicesDep, run_id: Annotated[str, Path(pattern=r"^run_[0-9a-f]{16}$")]
) -> EvaluationRunDetail:
    with session_scope(services.session_factory) as session:
        row = EvaluationRunRepository(session).get(run_id)
        if row is None:
            raise NotFound("Evaluation run not found.")
        return EvaluationRunDetail.model_validate(row)
