"""
FastAPI wrapper around the CP-SAT dispatch solver.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health    -> liveness + OR-Tools version
    POST /optimize  -> solve one planning horizon
    POST /validate  -> re-check an existing plan against every hard constraint
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__
from .models import OptimizeRequest, OptimizeResponse
from .solver import solve
from .validate import validate_plan


class ValidateRequest(BaseModel):
    """A finished plan plus the request that produced it."""

    request: OptimizeRequest
    plan: OptimizeResponse


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("dispatch-optimizer")

app = FastAPI(
    title="FIMTO Dispatch Optimizer",
    version=__version__,
    description="Ready-Mix Concrete dispatching with Google OR-Tools CP-SAT.",
)

_allowed = os.getenv("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        import ortools

        ortools_version = ortools.__version__
    except Exception:  # pragma: no cover - only if the wheel is missing
        ortools_version = None
    return {"ok": True, "version": __version__, "ortools": ortools_version}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    try:
        result = solve(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("solver crashed")
        raise HTTPException(status_code=500, detail=f"solver error: {exc}") from exc

    log.info(
        "plan date=%s status=%s loads=%d/%d trucks=%d solve=%dms obj=%s",
        req.plan_date,
        result.status,
        result.metrics.loads_scheduled,
        result.metrics.loads_total,
        result.metrics.trucks_used,
        result.solve_ms,
        result.objective,
    )
    return result


@app.post("/validate")
def validate(body: ValidateRequest) -> dict[str, Any]:
    """Independent constraint check — used before a plan is persisted/published."""
    report = validate_plan(body.request, body.plan)
    if not report.ok:
        log.warning("plan rejected: %s", report.violations[0].message)
    return {
        "ok": report.ok,
        "checked": report.checked,
        "violations": [v.__dict__ for v in report.violations],
    }
