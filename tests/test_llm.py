import httpx2
import openai
import pytest
from conftest import FakeLLMClient, response

from financeagents.errors import (
    ConfigError,
    LLMAuthError,
    LLMBudgetError,
    LLMRequestError,
    LLMUnavailableError,
)
from financeagents.llm import DEFAULT_BASE_URL, Settings, complete, parse_model_list, resolve_settings

_REQ = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def status_error(code: int) -> openai.APIStatusError:
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
           400: openai.BadRequestError, 500: openai.InternalServerError}.get(code, openai.APIStatusError)
    return cls(f"status {code}", response=httpx2.Response(code, request=_REQ), body=None)


def settings(**kw) -> Settings:
    base = dict(api_key="sk-test", model="a", fallback_models=("b", "c"), max_attempts=4)
    base.update(kw)
    return Settings(**base)


MSGS = [{"role": "user", "content": "hi"}]


def test_parse_model_list_trims_dedupes_and_keeps_order():
    assert parse_model_list("a, b,,a , c") == ["a", "b", "c"]
    assert parse_model_list("") == [] and parse_model_list(None) == []


def test_from_env_reads_fallbacks_and_caps():
    s = Settings.from_env({"LLM_API_KEY": "SENTINEL-KEY-123", "LLM_MODEL": "a", "FALLBACK_MODELS": "b, a, c", "LLM_MAX_ATTEMPTS": "2"})
    assert s.model_chain == ["a", "b"]
    assert s.uses_server_key and "SENTINEL-KEY-123" not in repr(s)


def test_429_falls_back_to_next_model():
    client = FakeLLMClient({"a": [status_error(429)], "b": ["answer"]})
    r = complete(settings(), MSGS, client_factory=client.factory)
    assert r.text == "answer" and r.requested_model == "b"
    assert len(r.attempts) == 2 and r.attempts[-1] == ("b", "ok")


def test_empty_content_with_length_is_retryable_and_never_returned():
    client = FakeLLMClient({"a": [response("", model="a", finish_reason="length")], "b": ["fine"]})
    r = complete(settings(), MSGS, client_factory=client.factory)
    assert r.text == "fine"
    assert "finish_reason=length" in r.attempts[0][1]


@pytest.mark.parametrize("exc", [openai.APITimeoutError(request=_REQ), openai.APIConnectionError(request=_REQ),
                                 status_error(500)])
def test_transient_errors_fall_back(exc):
    client = FakeLLMClient({"a": [exc], "b": ["ok"]})
    assert complete(settings(), MSGS, client_factory=client.factory).text == "ok"


@pytest.mark.parametrize("code,err", [(401, LLMAuthError), (403, LLMAuthError), (402, LLMBudgetError),
                                      (400, LLMRequestError)])
def test_non_retryable_errors_stop_after_one_attempt(code, err):
    client = FakeLLMClient({"a": [status_error(code)]})
    with pytest.raises(err):
        complete(settings(), MSGS, client_factory=client.factory)
    assert len(client.calls) == 1


def test_validator_failure_moves_to_next_model():
    def validator(text):
        if text != "{}":
            raise ValueError("not json")

    client = FakeLLMClient({"a": ["prose"], "b": ["{}"]})
    r = complete(settings(), MSGS, validator=validator, client_factory=client.factory)
    assert r.text == "{}" and len(client.calls) == 2


def test_exhausted_chain_raises_with_reasons_and_respects_cap():
    client = FakeLLMClient({m: [status_error(429)] for m in "abc"})
    with pytest.raises(LLMUnavailableError) as ei:
        complete(settings(max_attempts=2), MSGS, client_factory=client.factory)
    assert len(client.calls) == 2
    assert [m for m, _ in ei.value.attempts] == ["a", "b"]
    assert "RateLimitError" in str(ei.value)


def test_sdk_retries_disabled_and_timeout_set():
    from financeagents.llm import make_client

    c = make_client(settings(timeout_s=12))
    assert c.max_retries == 0 and c.timeout == 12


# --- SEC-01: server key handling -------------------------------------------------------------

SERVER = Settings(api_key="server-secret", base_url=DEFAULT_BASE_URL, model="srv")


def test_server_key_used_when_user_gives_none():
    s = resolve_settings(SERVER)
    assert s.uses_server_key and s.api_key == "server-secret" and s.base_url == DEFAULT_BASE_URL


def test_server_key_never_sent_to_custom_url():
    with pytest.raises(ConfigError):
        resolve_settings(SERVER, user_base_url="https://evil.example.com/v1")


def test_server_url_with_trailing_slash_ok_but_other_provider_refused():
    assert resolve_settings(SERVER, user_base_url=DEFAULT_BASE_URL + "/").base_url == DEFAULT_BASE_URL
    with pytest.raises(ConfigError):
        resolve_settings(SERVER, user_base_url="https://api.openai.com/v1")


def test_user_key_may_use_custom_https_url_and_replaces_server_key():
    s = resolve_settings(SERVER, user_api_key="mine", user_base_url="https://my.proxy/v1", user_model="x")
    assert s.api_key == "mine" and not s.uses_server_key and s.model == "x"
    with pytest.raises(ConfigError):
        resolve_settings(SERVER, user_api_key="mine", user_base_url="http://plain.example")


def test_no_key_anywhere_is_config_error():
    with pytest.raises(ConfigError):
        resolve_settings(Settings())


def test_sessions_are_independent():
    a = resolve_settings(SERVER, user_api_key="key-a", user_model="m1")
    b = resolve_settings(SERVER)
    assert a.api_key == "key-a" and b.api_key == "server-secret" and SERVER.model == "srv"


def test_bad_numeric_env_is_config_error():
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_MAX_ATTEMPTS": "four"})


def test_404_model_not_found_falls_back():
    client = FakeLLMClient({"a": [status_error(404)], "b": ["ok"]})
    assert complete(settings(), MSGS, client_factory=client.factory).text == "ok"


def test_server_key_sessions_limited_to_configured_models():
    server = Settings(api_key="server-secret", model="cheap", fallback_models=("cheap-2",))
    assert resolve_settings(server, user_model="cheap-2").model == "cheap-2"
    with pytest.raises(ConfigError, match="operator-configured"):
        resolve_settings(server, user_model="openai/gpt-4o")
    assert resolve_settings(server, user_api_key="mine", user_model="openai/gpt-4o").model == "openai/gpt-4o"
