
import pytest
from conftest import APP_PATH

from financeagents.ui.presentation import describe_result

RESULT = {
    "final_brief": "AAPL trades at 342.96.",
    "ticker_result": {"ticker": "AAPL", "company": "Apple Inc.", "exchange": "NMS", "confidence": "medium",
                      "source": "yahoo_search", "candidates": ["AAPL", "AAPL.MX"]},
    "data": {"market_data": {"last_close_date": "2026-10-15", "currency": "USD"},
             "risk_metrics": {"benchmark": "^GSPC", "benchmark_name": "S&P 500", "start_date": "2025-10-16",
                              "end_date": "2026-10-14", "risk_free_rate": 0.04088,
                              "risk_free_rate_source": "^IRX close 4.088% on 2026-10-15 (live)"}},
    "data_gaps": ["financials"],
    "sources": ["https://news.example/a"],
    "faithfulness": {"numbers_checked": 4, "unsupported": [{"raw": "47.3%", "context": "x"}]},
    "trace": {"run_id": "abc123def456", "total_latency_ms": 1234.5, "total_tokens": 240, "models_used": ["m"],
              "steps": [{"name": "synthesis", "ms": 900.0, "ok": True}], "llm_calls": []},
}


def test_describe_result_has_every_element():
    f = describe_result(RESULT)
    assert "AAPL" in f["resolution"] and "confidence **medium**" in f["resolution"]
    assert "AAPL.MX" in f["alternatives"]
    assert "2026-10-15" in f["as_of"] and "S&P 500 (^GSPC)" in f["as_of"] and "4.09%" in f["as_of"]
    assert f["sources"][1] == "Web: https://news.example/a" and "Yahoo Finance" in f["sources"][0]
    assert f["gaps"] == "Unavailable data: financials"
    assert f["check"][0] == "warning" and "1 of 4" in f["check"][1] and "47.3%" in f["check"][1]
    assert "abc123def456" in f["trace"] and "240 tokens" in f["trace"]


def test_high_confidence_and_clean_check():
    r = {**RESULT, "ticker_result": {**RESULT["ticker_result"], "confidence": "high"},
         "faithfulness": {"numbers_checked": 3, "unsupported": []}, "data_gaps": []}
    f = describe_result(r)
    assert f["alternatives"] is None and f["gaps"] is None and f["check"][0] == "success" and "all 3 numbers checked" in f["check"][1]


def test_streamlit_analysis_shows_sources_and_as_of(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    from financeagents import pipeline as graph

    monkeypatch.setattr(graph, "run_analysis", lambda g, state: dict(RESULT))
    monkeypatch.setattr(graph, "build_graph", lambda settings: object())
    monkeypatch.setenv("LLM_API_KEY", "SERVER-SENTINEL-KEY")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    next(b for b in at.button if "Analysis Mode" in b.label).click().run()
    at.chat_input[0].set_value("apple?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Resolved **AAPL**" in text and "Prices as of last close 2026-10-15" in text
    assert "Web: https://news.example/a" in text and "abc123def456" in text
    assert any("1 of 4 numbers" in w.value for w in at.warning)
