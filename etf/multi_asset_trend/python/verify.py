"""Compares the Python and R monthly returns of all 18 grid cells."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "data/grid_returns_python.csv", index_col=0, parse_dates=True)
R = pd.read_csv(ROOT / "data/trend_r_returns.csv")
R.index = pd.to_datetime(R.pop("month") + "-01") + pd.offsets.MonthEnd(0)
assert list(P.index) == list(R.index), (P.index[[0, -1]], R.index[[0, -1]], len(P), len(R))
worst = 0
for c in P.columns:
    d = (P[c] - R[c]).abs().max(); worst = max(worst, d)
    assert d < 1e-8, (c, d)
print(f"ALL PASSED: {len(P.columns)} cells x {len(P)} months, max diff {worst:.2e}")
