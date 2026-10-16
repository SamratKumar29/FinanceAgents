"""
tracing.py -- Per-run trace (steps, LLM calls, tokens, latency) and JSON logging with secret redaction.

    with start_trace(kind="analysis", query=q) as tr:
        ...                      # llm.complete() and traced steps record into ``tr``
    tr.summary()                 # {"run_id", "steps", "llm_calls", "total_latency_ms", "total_tokens", ...}
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Optional

_current: ContextVar[Optional["Trace"]] = ContextVar("finance_trace", default=None)
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9_\-.]{6,}|api[_-]?key['\"=:\s]+[^\s'\",]{6,})", re.I)


def redact(text: str) -> str:
    return _SECRET.sub("[REDACTED]", text)


@dataclass
class Trace:
    meta: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    steps: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        rec = {"name": name, "ok": True}
        try:
            yield
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {redact(str(e))[:200]}")
            raise
        finally:
            rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            with self._lock:
                self.steps.append(rec)

    def record_llm(self, task: str, attempts: list, served_model: Optional[str] = None,
                   usage: Optional[dict] = None, latency_ms: float = 0.0, error: Optional[str] = None) -> None:
        rec = {"task": task, "served_model": served_model, "attempts": [(m, redact(r)) for m, r in attempts],
               "usage": usage, "latency_ms": round(latency_ms)}
        if error:
            rec["error"] = redact(error)[:200]
        with self._lock:
            self.llm_calls.append(rec)

    def summary(self) -> dict:
        tokens = sum((c.get("usage") or {}).get("total_tokens") or 0 for c in self.llm_calls)
        return {
            "run_id": self.run_id,
            **self.meta,
            "steps": list(self.steps),
            "llm_calls": list(self.llm_calls),
            "total_latency_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "total_tokens": tokens,
            "models_used": sorted({c["served_model"] for c in self.llm_calls if c.get("served_model")}),
        }


def current() -> Optional[Trace]:
    return _current.get()


@contextmanager
def start_trace(**meta) -> Iterator[Trace]:
    tr = Trace(meta={k: redact(str(v))[:200] for k, v in meta.items()})
    token = _current.set(tr)
    try:
        yield tr
    finally:
        _current.reset(token)


@contextmanager
def step(name: str) -> Iterator[None]:
    """Trace a step on the current run, or do nothing when no trace is active."""
    tr = current()
    if tr is None:
        yield
        return
    with tr.step(name):
        yield


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tr = current()
        out = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
               "msg": redact(record.getMessage())}
        if tr is not None:
            out["run_id"] = tr.run_id
        if record.exc_info:
            out["exc"] = redact(self.formatException(record.exc_info))[-1000:]
        return json.dumps(out, default=str)


def configure_logging(level: int = logging.WARNING) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
