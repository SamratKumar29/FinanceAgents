"""DEMO-01: both modes complete with no API key and the network blocked."""

import pytest
from conftest import APP_PATH

from financeagents import cli as main
from financeagents import demo


@pytest.fixture
def demo_mode(monkeypatch):
    monkeypatch.setenv("FINANCEAGENTS_DEMO", "1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    yield
    demo.disable()


def test_cli_demo_analysis_uses_recorded_brief(demo_mode, capsys):
    assert main.main(["--demo", "Tell me about Apple"]) == 0
    out = capsys.readouterr().out
    assert "demo mode" in out and "Ticker: AAPL" in out
    assert "(recorded 20" in out  # served model label of a recorded free-model response


def test_cli_demo_template_for_unrecorded_ticker(demo_mode, capsys):
    assert main.main(["--demo", "--ticker", "MSFT"]) == 0
    out = capsys.readouterr().out
    assert "demo template brief" in out and "demo-template" in out


def test_cli_demo_investment_mode(demo_mode, capsys, monkeypatch):
    answers = iter(["India", "5 lakh", "lumpsum", "retirement", "moderate", "10 years", "4", "india", "none"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main.main(["--demo", "--mode", "investment"]) == 0
    out = capsys.readouterr().out
    assert "NIFTYBEES.NS" in out and "Total: INR 500,000.00" in out


def test_demo_search_finds_fixture_companies():
    hits = demo._fixture_search("Infosys")
    assert hits and hits[0]["symbol"] == "INFY.NS"


def test_streamlit_demo_both_modes(demo_mode):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    path = APP_PATH
    at = AppTest.from_file(path, default_timeout=60).run()
    assert any("Demo mode" in i.value for i in at.info)
    assert not [t for t in at.text_input if t.label == "API Key"]  # settings hidden in demo

    next(b for b in at.button if "Analysis Mode" in b.label).click().run()
    at.chat_input[0].set_value("How is NVIDIA doing?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Resolved **NVDA**" in text and "Prices as of last close" in text

    next(b for b in at.button if "Investment Mode" in b.label).click().run()
    for a in ["USA", "10000", "lumpsum", "growth", "aggressive", "15 years", "5", "us", "none"]:
        at.chat_input[0].set_value(a).run()
        assert not at.exception
    assert any("Educational model allocation" in m.value for m in at.markdown)
