"""Verification: (1) Python vs R monthly returns + every sensitivity cell,
(2) hand re-computation of one month straight from raw prices."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"

py = pd.read_csv(D / "monthly_returns_python.csv", index_col=0, parse_dates=True)
r = pd.read_csv(D / "monthly_returns_r.csv")
r.index = pd.PeriodIndex(r["month"], freq="M"); py.index = py.index.to_period("M")
assert len(py) == len(r), (len(py), len(r))
diff = (py[["strategy_net", "strategy_gross", "equal_weight", "spy"]]
        - r[["strategy_net", "strategy_gross", "equal_weight", "spy"]]).abs().max()
print("max |Python - R| monthly return diff:\n", diff.to_string())

jp = json.load(open(D / "results_python.json"))
rt = pd.read_csv(D / "results_table_r.csv", index_col=0)
rs = pd.read_csv(D / "sensitivity_r.csv")
gd = max(abs(jp["sensitivity"][f"T{int(r.T_months)}_N{int(r.n)}"][m] - r[m])
         for _, r in rs.iterrows() for m in ("cagr_pct", "sharpe"))
print("max sensitivity-grid diff (16 cells x CAGR,Sharpe):", gd)
cmp = {"strategy_net": "Sector momentum (net)", "equal_weight_sectors": "Equal-weight 9 sectors", "spy": "SPY"}
for k, c in cmp.items():
    for m, row in (("cagr_pct", "CAGR (%)"), ("sharpe", "Sharpe"), ("max_drawdown_pct", "Max drawdown (%)")):
        assert abs(jp[k][m] - rt.loc[row, c]) < 0.011, (k, m, jp[k][m], rt.loc[row, c])

# hand check: signal at 2008-12-31, earns Jan 2009 (plain pandas, no rank trick)
px = pd.read_csv(D / "prices_python.csv", index_col=0, parse_dates=True)
S = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
p_now = px[S].loc[:"2008-12-31"].iloc[-1]; p_then = px[S].loc[:"2007-12-31"].iloc[-1]
mom = (p_now / p_then - 1).sort_values(ascending=False)
top = list(mom.index[:3])
p_next = px[S].loc[:"2009-01-30"].iloc[-1]
hand = (p_next / p_now - 1)[top].mean()
print("hand check -- top 3 at 2008-12-31:", top, "12m mom:", mom[:3].round(3).to_dict())
print("hand gross Jan-2009 return: %.6f   pipeline: %.6f" % (hand, py.loc[pd.Period("2009-01"), "strategy_gross"]))
assert abs(hand - py.loc[pd.Period("2009-01"), "strategy_gross"]) < 1e-9
# the two scripts download their own data: check the prices agree too
pr = pd.read_csv(D / "prices_r.csv", index_col=0, parse_dates=True)
pd_ = px.reindex(pr.index)
rel = ((pd_ - pr).abs() / pr).max().max()
print("max relative price difference, Python (yfinance) vs R (quantmod): %.2e  rows py/r: %d/%d" % (rel, len(px), len(pr)))
assert len(px) == len(pr) and rel < 1e-5
assert diff.max() < 1e-5 and gd < 0.011
print("ALL CHECKS PASSED")
