PY  ?= venv/bin/python
PIP ?= venv/bin/pip
APP := src/financeagents/ui/app.py

.PHONY: help setup setup-latest test lint format run demo eval-offline eval-live bench fixtures clean

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:          ## venv + exact verified versions (requirements.lock) + editable package
	python3.12 -m venv venv
	$(PIP) install -r requirements.lock
	$(PIP) install -e . --no-deps

setup-latest:   ## venv + newest versions allowed by pyproject.toml
	python3.12 -m venv venv
	$(PIP) install -e ".[dev]"

test:           ## offline test suite (network blocked)
	$(PY) -m pytest

lint:           ## ruff
	$(PY) -m ruff check .

format:         ## ruff auto-fix
	$(PY) -m ruff check --fix .

run:            ## Streamlit app (needs LLM_API_KEY in .env or your own key in the sidebar)
	venv/bin/streamlit run $(APP)

demo:           ## offline demo: recorded data + responses, no key, no network
	FINANCEAGENTS_DEMO=1 venv/bin/streamlit run $(APP)

eval-offline:   ## resolver + portfolio-constraint evaluation with CI gates (offline)
	$(PY) -m evaluation.run_offline

eval-live:      ## brief faithfulness on FREE models only; refreshes demo responses
	$(PY) -m evaluation.run_faithfulness_live

bench:          ## synthetic data-path latency benchmark
	$(PY) -m evaluation.bench_latency

fixtures:       ## re-record Yahoo Finance fixtures (network)
	$(PY) scripts/record_fixtures.py

clean:          ## remove caches and build artefacts
	find . -path ./venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
