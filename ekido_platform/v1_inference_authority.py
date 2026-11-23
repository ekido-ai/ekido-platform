"""Internal, fail-closed identity and usage authority for the inference router.

Only an operator-managed model manifest maps a model to a job and a loopback engine.
The shared internal token must be secret and the service must stay on a private host.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import itertools
import json
import re
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .accounts import AccountError, ApiKey, InferenceUsage, JobOwner
from .models import utcnow


class Reserve(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr
    model_id: str = Field(min_length=1, max_length=120)


class Finish(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ok", "error"]
    input_tokens: int | None = Field(default=None, ge=0, le=2_147_483_647)
    output_tokens: int | None = Field(default=None, ge=0, le=2_147_483_647)
    latency_ms: int = Field(ge=0, le=2_147_483_647)


class BatchUsageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reservation_id: str = Field(min_length=36, max_length=36)
    input_tokens: int = Field(ge=0, le=2_147_483_647)
    output_tokens: int = Field(ge=0, le=2_147_483_647)
    latency_ms: int = Field(ge=0, le=2_147_483_647)


class FinishBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    items: list[BatchUsageItem] = Field(min_length=1, max_length=32)


def model_manifest(root: Path, model_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", model_id):
        raise ValueError("invalid model id")
    target = (root / f"{model_id}.json").resolve()
    target.relative_to(root)
    with target.open("rb") as handle:
        data = handle.read(16385)
    if len(data) > 16384:
        raise ValueError("manifest too large")
    item = json.loads(data)
    if (not isinstance(item, dict) or
            not all(isinstance(item.get(k), str) and item[k] for k in ("job_id", "engine_url", "engine_model")) or
            len(item["job_id"]) > 36 or len(item["engine_url"]) > 512 or len(item["engine_model"]) > 200):
        raise ValueError("invalid manifest")
    if item.get("mode", "live") not in ("test", "live"):
        raise ValueError("invalid model mode")
    url = urlsplit(item["engine_url"])
    if (url.scheme != "http" or url.username is not None or url.password is not None or
            url.path not in ("", "/") or url.query or url.fragment or url.port is None or
            not ipaddress.ip_address(url.hostname or "").is_loopback):
        raise ValueError("engine must use a numeric loopback origin")
    return item


def build_inference_authority(engine: Engine, model_root: str, internal_token: str, rate: int) -> APIRouter:
    router = APIRouter(prefix="/internal/inference")
    root = Path(model_root).resolve() if model_root else None

    def internal(x_ekido_inference_token: str = Header(default="", max_length=512)) -> None:
        if root is None or len(internal_token) < 32 or rate < 1:
            raise AccountError("INFERENCE_UNAVAILABLE", "Inference is not configured.", 503)
        if not hmac.compare_digest(x_ekido_inference_token.encode('utf-8'), internal_token.encode('utf-8')):
            raise AccountError("UNAUTHORIZED", "Invalid internal credential.", 401)

    @router.get("/ready", dependencies=[Depends(internal)])
    def ready() -> dict:
        try:
            files = list(itertools.islice(root.iterdir(), 257))
        except OSError:
            raise AccountError("INFERENCE_UNAVAILABLE", "Model registry is unavailable.", 503) from None
        if len(files) > 256:
            raise AccountError("INFERENCE_UNAVAILABLE", "Model registry exceeds the configured limit.", 503)
        engines = set()
        with Session(engine) as session:
            for path in files:
                if path.suffix != ".json":
                    continue
                try:
                    item = model_manifest(root, path.stem)
                except (ValueError, OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if session.get(JobOwner, item["job_id"]) is not None:
                    engines.add(item["engine_url"])
        if not engines:
            raise AccountError("INFERENCE_UNAVAILABLE", "No models are configured.", 503)
        return {"engines": sorted(engines)}

    @router.post("/reserve", dependencies=[Depends(internal)])
    def reserve(body: Reserve) -> dict:
        key_secret = body.api_key.get_secret_value()
        if not re.fullmatch(r"ek_(?:test|live)_[A-Za-z0-9_-]{20,128}", key_secret):
            raise AccountError("INVALID_KEY", "Invalid or revoked API key.", 401)
        now = utcnow()
        with Session(engine) as session, session.begin():
            key = session.scalar(select(ApiKey).where(ApiKey.key_hash == hashlib.sha256(key_secret.encode()).hexdigest()).with_for_update())
            if key is None or key.revoked_at is not None:
                raise AccountError("INVALID_KEY", "Invalid or revoked API key.", 401)
            if "models.predict" not in key.scopes:
                raise AccountError("FORBIDDEN", "API key needs models.predict.", 403)
            try:
                manifest = model_manifest(root, body.model_id)
            except (ValueError, OSError, UnicodeError, json.JSONDecodeError):
                raise AccountError("NOT_FOUND", "No such model.", 404) from None
            owner = session.get(JobOwner, manifest["job_id"])
            if owner is None or owner.org_id != key.org_id:
                raise AccountError("NOT_FOUND", "No such model.", 404)
            if key.mode != manifest.get("mode", "live"):
                raise AccountError("NOT_FOUND", "No such model.", 404)
            count = session.scalar(select(func.count()).select_from(InferenceUsage).where(
                InferenceUsage.key_id == key.id, InferenceUsage.created_at >= now - timedelta(minutes=1)))
            if count >= rate:
                raise AccountError("RATE_LIMITED", "Too many requests. Try again shortly.", 429)
            reservation_id = str(uuid.uuid4())
            session.add(InferenceUsage(id=reservation_id, key_id=key.id, model_id=body.model_id,
                                       created_at=now, status="pending"))
            key.last_used_at = now
        return {"reservation_id": reservation_id, "engine_url": manifest["engine_url"],
                "engine_model": manifest["engine_model"]}

    @router.post("/usage-batch", dependencies=[Depends(internal)])
    def finish_batch(body: FinishBatch) -> dict:
        items = {item.reservation_id: item for item in body.items}
        if len(items) != len(body.items):
            raise AccountError("INVALID_USAGE", "Duplicate reservations are not allowed.", 422)
        try:
            if any(str(uuid.UUID(item_id)) != item_id for item_id in items):
                raise ValueError("noncanonical UUID")
        except ValueError:
            raise AccountError("INVALID_USAGE", "Invalid reservation ID.", 422) from None
        with Session(engine) as session, session.begin():
            rows = list(session.scalars(select(InferenceUsage).where(InferenceUsage.id.in_(items))
                                        .order_by(InferenceUsage.id).with_for_update()))
            if len(rows) != len(items):
                raise AccountError("NOT_FOUND", "No such reservation.", 404)
            if len({(row.key_id, row.model_id) for row in rows}) != 1:
                raise AccountError("INVALID_USAGE", "Reservations must share a key and model.", 422)
            if all(row.status == "ok" and row.input_tokens == items[row.id].input_tokens
                   and row.output_tokens == items[row.id].output_tokens
                   and row.latency_ms == items[row.id].latency_ms for row in rows):
                return {"ok": True}  # Exact replay after a lost response.
            if any(row.status != "pending" for row in rows):
                raise AccountError("ALREADY_FINISHED", "Reservation was already finalized.", 409)
            for row in rows:
                item = items[row.id]
                row.status, row.input_tokens = "ok", item.input_tokens
                row.output_tokens, row.latency_ms = item.output_tokens, item.latency_ms
        return {"ok": True}

    @router.post("/usage/{reservation_id}", dependencies=[Depends(internal)])
    def finish(reservation_id: str, body: Finish) -> dict:
        if body.status == "ok" and (body.input_tokens is None or body.output_tokens is None):
            raise AccountError("INVALID_USAGE", "Successful requests require token counts.", 422)
        if (body.input_tokens is None) != (body.output_tokens is None):
            raise AccountError("INVALID_USAGE", "Token counts must be supplied together.", 422)
        with Session(engine) as session, session.begin():
            usage = session.scalar(select(InferenceUsage).where(InferenceUsage.id == reservation_id).with_for_update())
            if usage is None:
                raise AccountError("NOT_FOUND", "No such reservation.", 404)
            if usage.status != "pending":
                raise AccountError("ALREADY_FINISHED", "Reservation was already finalized.", 409)
            usage.status, usage.input_tokens = body.status, body.input_tokens
            usage.output_tokens, usage.latency_ms = body.output_tokens, body.latency_ms
        return {"ok": True}

    return router
