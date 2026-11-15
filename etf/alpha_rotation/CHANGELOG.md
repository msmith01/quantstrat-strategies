# 4.2 Alpha rotation: changelog

## 2026-11-13 (v2, live post 1017 updated)
Owner's reviewer report: `review/external_comments/2026-11-13_boss_alpha_rotation.md`; decisions: `review/reports/alpha_rotation_20261113_boss_reconciliation.md`.
- Main strategy and headline numbers unchanged (7.00% CAGR, 321 months).
- NEW factor availability test (lag 5/21 days x same/next close, 319 months, benchmarks rerun).
- FIXED sensitivity grid: same 309 months in every cell, fresh capital Jan 2001, benchmark rerun on those months; Python == R.
- FIXED script plots start at $100; `pct_change(fill_method=None)`.
- NEW data/MANIFEST.json (retrieval date, factor vintage 202609, SHA-256s), data/raw/ copy of the French zip.
- TEXT: Nobel claim corrected, "every researcher" claim corrected, "noise" softened, alpha defined precisely, window status stated precisely, full-sample regression labelled descriptive, 126/504 choice explained, limitations moved to Summary and Conclusion.

## 2026-11-13 (v1)
First publication.

## Reproduce
```
python python/regen_results.py      # run both scripts with SAVE on
python python/archive_data.py       # data manifest + raw factor zip
python python/verify.py > data/verification_output.txt
python python/make_charts.py
python python/build_post.py
python ../../review/gates.py etf/alpha_rotation   (from strategy_replications: python review/gates.py etf/alpha_rotation)
```
