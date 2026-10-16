"""
llm.py -- The single place that talks to the LLM provider.

* ``Settings`` is immutable and passed explicitly (one per Streamlit session / CLI run),
  so no user's key, model or base URL leaks into another session.
* ``resolve_settings`` only sends the server key to its own configured base URL.
* ``complete`` walks ``[model, *fallback_models]`` and moves on only for retryable
  failures (429, timeout, connection, 5xx, empty content, invalid output).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional

import openai

from financeagents import tracing
from financeagents.errors import (
    ConfigError,
    EmptyCompletionError,
    InvalidOutputError,
    LLMAuthError,
    LLMBudgetError,
    LLMError,
    LLMRequestError,
    LLMUnavailableError,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "xiaomi/mimo-v2.5"


def parse_model_list(raw: Optional[str]) -> list[str]:
    """Split a comma-separated model list: trim, drop empties, de-duplicate, keep order."""
    seen: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    fallback_models: tuple[str, ...] = ()
    max_attempts: int = 4
    timeout_s: float = 30.0
    uses_server_key: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        key = env.get("LLM_API_KEY", "")
        try:
            max_attempts = max(1, int(env.get("LLM_MAX_ATTEMPTS") or 4))
            timeout_s = float(env.get("LLM_TIMEOUT_S") or 30)
        except ValueError as e:
            raise ConfigError(f"LLM_MAX_ATTEMPTS / LLM_TIMEOUT_S must be numbers: {e}") from e
        return cls(
            api_key=key,
            base_url=_normalize_url(env.get("LLM_BASE_URL") or DEFAULT_BASE_URL),
            model=(env.get("LLM_MODEL") or DEFAULT_MODEL).strip(),
            fallback_models=tuple(parse_model_list(env.get("FALLBACK_MODELS"))),
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            uses_server_key=bool(key),
        )

    @property
    def model_chain_all(self) -> list[str]:
        """Every configured model (primary + fallbacks, de-duplicated), ignoring max_attempts."""
        return list(dict.fromkeys(m for m in (self.model, *self.fallback_models) if m))

    @property
    def model_chain(self) -> list[str]:
        chain: list[str] = []
        for m in (self.model, *self.fallback_models):
            if m and m not in chain:
                chain.append(m)
        return chain[: self.max_attempts]


def resolve_settings(
    server: Settings,
    user_api_key: str = "",
    user_base_url: str = "",
    user_model: str = "",
) -> Settings:
    """Build per-session settings from user input without exposing or redirecting the server key."""
    user_api_key = (user_api_key or "").strip()
    base_url = _normalize_url(user_base_url) or server.base_url
    model = (user_model or "").strip() or server.model

    if user_api_key:
        if not base_url.startswith("https://"):
            raise ConfigError("Base URL must use https://")
        return replace(server, api_key=user_api_key, base_url=base_url, model=model, uses_server_key=False)

    if not server.api_key:
        raise ConfigError("No API key configured. Enter your own API key in the sidebar.")
    allowed = server.model_chain_all
    if model not in allowed:
        raise ConfigError(
            f"With the server key only the operator-configured models are allowed ({', '.join(allowed)}). "
            "Enter your own API key to use other models."
        )
    # The server key is only ever sent to the operator-configured base URL.
    if base_url != server.base_url:
        raise ConfigError(
            "A custom base URL can only be used with your own API key; the server key is restricted "
            "to the configured provider."
        )
    return replace(server, base_url=base_url, model=model, uses_server_key=True)


@dataclass
class Completion:
    text: str
    served_model: str
    requested_model: str
    attempts: list[tuple[str, str]]  # (model, outcome) for every attempt, last one "ok"
    usage: Optional[dict] = None
    latency_ms: float = 0.0


ClientFactory = Callable[[Settings], object]


def make_client(settings: Settings):
    if settings.base_url.startswith("demo://"):
        from financeagents import demo  # offline replay client; never touches the network

        return demo.DemoClient()
    if not settings.api_key:
        raise LLMAuthError("No API key configured.")
    return openai.OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_s,
        max_retries=0,  # the fallback chain owns retries
    )


def _classify(exc: Exception) -> tuple[bool, Exception]:
    """Return (retryable, exception_to_raise_if_not_retryable)."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True, exc
    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        if code in (404, 408, 409, 429) or code >= 500:  # 404: model retired/unknown at this provider
            return True, exc
        if code in (401, 403):
            return False, LLMAuthError(str(exc))
        if code == 402:
            return False, LLMBudgetError(str(exc))
        return False, LLMRequestError(str(exc))
    if isinstance(exc, (EmptyCompletionError, InvalidOutputError)):
        return True, exc
    return False, LLMError(f"{type(exc).__name__}: {exc}")


def complete(
    settings: Settings,
    messages: list[dict],
    *,
    validator: Optional[Callable[[str], object]] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    client_factory: ClientFactory = make_client,
    task: str = "llm",
) -> Completion:
    """Run one chat completion with ordered model fallback.

    ``validator`` receives the text and should raise ``ValueError`` (or return normally);
    a failure triggers the next model.
    """
    chain = settings.model_chain
    if not chain:
        raise ConfigError("No LLM model configured.")
    client = client_factory(settings)
    attempts: list[tuple[str, str]] = []
    for model in chain:
        started = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                raise EmptyCompletionError(f"empty content (finish_reason={choice.finish_reason})")
            if validator is not None:
                try:
                    validator(text)
                except (ValueError, TypeError) as e:
                    raise InvalidOutputError(f"invalid output: {e}") from e
        except Exception as exc:  # noqa: BLE001 - classified below
            retryable, to_raise = _classify(exc)
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            attempts.append((model, reason))
            log.warning("LLM attempt failed model=%s retryable=%s reason=%s", model, retryable, reason)
            if not retryable:
                _trace(task, attempts, error=str(to_raise))
                raise to_raise from exc
            continue
        attempts.append((model, "ok"))
        usage = getattr(resp, "usage", None)
        result = Completion(
            text=text,
            served_model=getattr(resp, "model", None) or model,
            requested_model=model,
            attempts=attempts,
            usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        _trace(task, attempts, result.served_model, result.usage, result.latency_ms)
        return result
    _trace(task, attempts, error="all models unavailable")
    raise LLMUnavailableError(attempts)


def _trace(task, attempts, served_model=None, usage=None, latency_ms=0.0, error=None) -> None:
    tr = tracing.current()
    if tr is not None:
        tr.record_llm(task, attempts, served_model, usage, latency_ms, error)
