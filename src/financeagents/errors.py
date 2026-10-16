"""errors.py -- Typed failures surfaced to the UI/CLI instead of error strings."""


class FinanceAgentsError(Exception):
    """Base class for errors that have a user-facing message."""

    user_message = "Something went wrong."


class LLMError(FinanceAgentsError):
    user_message = "The language model request failed."


class LLMAuthError(LLMError):
    user_message = "LLM authentication failed. Check your API key."


class LLMBudgetError(LLMError):
    user_message = "The LLM account has insufficient credits."


class LLMRequestError(LLMError):
    user_message = "The LLM provider rejected the request."


class LLMUnavailableError(LLMError):
    """Every model in the fallback chain failed with a retryable error."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts
        reasons = ", ".join(f"{m}: {r}" for m, r in attempts) or "no models configured"
        super().__init__(f"All configured models are unavailable ({reasons})")

    @property
    def user_message(self) -> str:  # type: ignore[override]
        return str(self)


class EmptyCompletionError(LLMError):
    """Model returned no usable content (e.g. reasoning model hit the token cap)."""


class InvalidOutputError(LLMError):
    """Model output failed the call site's validator."""


class ConfigError(FinanceAgentsError):
    @property
    def user_message(self) -> str:  # type: ignore[override]
        return str(self)


class DataUnavailableError(FinanceAgentsError):
    user_message = "Market data is unavailable for this request."


class TickerNotResolvedError(FinanceAgentsError):
    user_message = (
        "Could not identify a stock ticker from your query. "
        "Try the explicit symbol (e.g. TTWO, RELIANCE.NS)."
    )
