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

jp, jr = json.load(open(D / "results_python.json")), json.load(open(D / "results_r.json"))
gd = max(abs(jp["sensitivity"][k][m] - jr["sensitivity"][k][m])
         for k in jp["sensitivity"] for m in ("cagr_pct", "sharpe"))
print("max sensitivity-grid diff (16 cells x CAGR,Sharpe):", gd)
for k in ("strategy_net", "equal_weight_sectors", "spy"):
    for m in ("cagr_pct", "sharpe", "max_drawdown_pct"):
        assert abs(jp[k][m] - jr[k][m]) < 0.011, (k, m, jp[k][m], jr[k][m])

# hand check: signal at 2008-12-31, earns Jan 2009 (plain pandas, no rank trick)
px = pd.read_csv(D / "prices.csv", index_col="date", parse_dates=True)
S = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
p_now = px[S].loc[:"2008-12-31"].iloc[-1]; p_then = px[S].loc[:"2007-12-31"].iloc[-1]
mom = (p_now / p_then - 1).sort_values(ascending=False)
top = list(mom.index[:3])
p_next = px[S].loc[:"2009-01-30"].iloc[-1]
hand = (p_next / p_now - 1)[top].mean()
print("hand check -- top 3 at 2008-12-31:", top, "12m mom:", mom[:3].round(3).to_dict())
print("hand gross Jan-2009 return: %.6f   pipeline: %.6f" % (hand, py.loc[pd.Period("2009-01"), "strategy_gross"]))
assert abs(hand - py.loc[pd.Period("2009-01"), "strategy_gross"]) < 1e-9
assert diff.max() < 1e-9 and gd < 0.011
print("ALL CHECKS PASSED")
