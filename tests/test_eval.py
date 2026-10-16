import json

import evaluation.run_offline as ev


def test_offline_eval_runs_and_passes_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    assert ev.main() == 0
    out = json.loads(next(tmp_path.glob("*_offline.json")).read_text())
    assert out["constraints"]["pass_rate_pct"] == 100
    assert out["resolver"]["current"]["no_answer_false_positive_pct"] == 0
    assert out["resolver"]["current"]["accuracy_pct"] > out["resolver"]["baseline"]["accuracy_pct"]


def test_gate_fails_on_accuracy_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"resolver_accuracy_pct": 99.9}))
    monkeypatch.setattr(ev, "BASELINE_FILE", baseline)
    assert ev.main() == 1
