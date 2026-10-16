## Offline evaluation (2026-10-16, fixtures recorded 2026-10-15)

Resolver: 99 labelled queries, deterministic stages only (no LLM, no Yahoo search).

| Metric | Baseline `32abe04` | Current |
|---|---|---|
| Accuracy | 70.7% | 79.8% |
| False positives on no-answer queries | 43.8% | 0.0% |
| Abstain rate | 30.3% | 36.4% |
| Accuracy: company_name | 60.0% | 66.7% |
| Accuracy: contextual | 100.0% | 100.0% |
| Accuracy: delisted | 0.0% | 100.0% |
| Accuracy: explicit_ticker | 96.0% | 88.0% |
| Accuracy: share_class | 0.0% | 33.3% |
| Accuracy: trap | 60.0% | 100.0% |

Portfolio constraints: 144 synthetic profiles, pass rate 100.0%.
