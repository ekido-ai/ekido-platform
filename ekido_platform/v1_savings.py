"""Authenticated savings reports from operator-managed, job-owned evidence."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, Response
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import accounts
from .accounts import AccountError, JobOwner, Principal
from .savings import SavingsError, _evidence, build_savings_report


def build_savings_router(engine: Engine, principal: Callable, evidence_root: str) -> APIRouter:
    router = APIRouter(prefix="/v1")
    root = Path(evidence_root).resolve() if evidence_root else None

    def load(model_id: str, who: Principal) -> tuple[dict, dict]:
        if root is None:
            raise AccountError("SAVINGS_UNAVAILABLE", "Savings reports are not configured.", 503)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", model_id):
            raise AccountError("NOT_FOUND", "No such report.", 404)
        try:
            manifest, _, _ = _evidence(root, f"models/{model_id}.json")
            job_id = manifest["job_id"]
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("invalid job")
        except (SavingsError, KeyError, ValueError, OSError):
            raise AccountError("NOT_FOUND", "No such report.", 404) from None
        with Session(engine) as session:
            owner = session.get(JobOwner, job_id)
            if owner is None:
                raise AccountError("NOT_FOUND", "No such report.", 404)
            accounts.require_role(session, who, owner.org_id, *accounts.ROLES)
            if who.via == "api_key" and "models.read" not in who.scopes:
                raise AccountError("FORBIDDEN", "The API key needs models.read.", 403)
        try:
            model_run, _, _ = _evidence(root, manifest["model_run_file"])
            if model_run.get("job_id") != job_id:
                raise SavingsError("manifest and model job differ")
            report = build_savings_report(evidence_root=root, model_id=model_id,
                expected_requests=manifest["expected_requests"], api_run_file=manifest["api_run_file"],
                model_run_file=manifest["model_run_file"], quote_file=manifest["quote_file"],
                price_file=manifest["price_file"], self_hosted=manifest.get("self_hosted", False))
        except (SavingsError, KeyError, TypeError, OSError):
            raise AccountError("SAVINGS_UNAVAILABLE", "Savings evidence is incomplete or invalid.", 503) from None
        return manifest, report

    @router.get("/models/{model_id}/savings-report")
    def savings_report(model_id: str, response: Response, who: Principal = Depends(principal)) -> dict:
        _, report = load(model_id, who)
        response.headers["Cache-Control"] = "private, no-store"
        paths = report.pop("evidence")
        hashes = report["source_sha256"]
        links = {role: f"/v1/models/{model_id}/savings-report/evidence/{role}" for role in paths}
        report["sources"] = list(links.values())
        report["source_sha256"] = {links[role]: hashes[path] for role, path in paths.items()}
        return report

    @router.get("/models/{model_id}/savings-report/evidence/{role}")
    def savings_evidence(model_id: str, role: str, who: Principal = Depends(principal)) -> Response:
        _, report = load(model_id, who)
        path = report["evidence"].get(role)
        if path is None:
            raise AccountError("NOT_FOUND", "No such evidence.", 404)
        try:
            resolved = (root / path).resolve()
            resolved.relative_to(root)
            with resolved.open("rb") as handle:
                data = handle.read(1_000_001)
            if len(data) > 1_000_000:
                raise OSError("evidence too large")
            if hashlib.sha256(data).hexdigest() != report["source_sha256"][path]:
                raise OSError("evidence changed")
        except (OSError, ValueError):
            raise AccountError("SAVINGS_UNAVAILABLE", "Savings evidence is unavailable.", 503) from None
        return Response(content=data, media_type="application/json", headers={"Cache-Control": "private, no-store"})

    return router
