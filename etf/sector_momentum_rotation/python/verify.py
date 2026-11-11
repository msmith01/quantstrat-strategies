"""Verification: (1) Python vs R: monthly returns, results table, every sensitivity cell, prices;
(2) hand re-computation of one month's return and one month's turnover from raw prices, with plain
loops (no ranking or drift tricks)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"

py = pd.read_csv(D / "monthly_returns_python.csv", index_col=0, parse_dates=True)
r = pd.read_csv(D / "monthly_returns_r.csv")
r.index = pd.PeriodIndex(r["month"], freq="M"); py.index = py.index.to_period("M")
assert len(py) == len(r), (len(py), len(r))
cols = ["top3_net", "top3_gross", "top1_net", "ew_net", "ew_gross", "spy", "turnover_top3"]
diff = (py[cols] - r[cols]).abs().max()
print("max |Python - R| per monthly series:\n", diff.to_string())

jp = json.load(open(D / "results_python.json"))
rt = pd.read_csv(D / "results_table_r.csv", index_col=0)
rs = pd.read_csv(D / "sensitivity_r.csv")
gd = max(abs(jp["sensitivity"][f"T{int(x.T_months)}_N{int(x.n)}"][m] - x[m])
         for _, x in rs.iterrows() for m in ("cagr_pct", "sharpe"))
print("max sensitivity-grid diff (16 cells x CAGR,Sharpe):", gd)
cmp = {"top3": "Top 3 of 9", "top1": "Top 1 of 9", "equal_weight": "Equal-weight 9 (net)", "spy": "SPY"}
for k, c in cmp.items():
    for m, row in (("cagr_pct", "CAGR (%)"), ("sharpe", "Sharpe"), ("max_drawdown_pct", "Max drawdown (%)")):
        assert abs(jp[k][m] - rt.loc[row, c]) < 0.011, (k, m, jp[k][m], rt.loc[row, c])

px = pd.read_csv(D / "prices_python.csv", index_col=0, parse_dates=True)
S = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
pxm = px.resample("ME").last()


def top_n(date, n=3, T=12):
    """Weights decided at month-end `date`, from raw month-end prices, plain loops."""
    i = list(pxm.index).index(date)
    mom = {s: pxm[s].iloc[i] / pxm[s].iloc[i - T] - 1 for s in S}
    top = sorted(S, key=lambda s: -mom[s])[:n]
    return {s: (1 / n if s in top else 0.0) for s in S}, top, mom


# hand check A: signal at 2008-12-31, earns Jan 2009
d0 = pd.Timestamp("2008-12-31"); w0, top, mom = top_n(d0)
i = list(pxm.index).index(d0)
hand = sum(w0[s] * (pxm[s].iloc[i + 1] / pxm[s].iloc[i] - 1) for s in S)
print("hand check A -- top 3 at 2008-12-31:", top, {s: round(mom[s], 3) for s in top})
print("hand gross Jan-2009 return: %.6f   pipeline: %.6f" % (hand, py.loc[pd.Period("2009-01"), "top3_gross"]))
assert abs(hand - py.loc[pd.Period("2009-01"), "top3_gross"]) < 1e-9

# hand check B: turnover when trading at 2009-02-28 (new weights vs last month's weights after drifting)
d1 = pd.Timestamp("2009-01-31"); d2 = pd.Timestamp("2009-02-28")
w_old, _, _ = top_n(d1); w_new, top_new, _ = top_n(d2)
j = list(pxm.index).index(d1)
grown = {s: w_old[s] * (pxm[s].iloc[j + 1] / pxm[s].iloc[j]) for s in S}      # value of each holding during Feb
tot = sum(grown.values()); drifted = {s: grown[s] / tot for s in S}
tn_hand = sum(abs(w_new[s] - drifted[s]) for s in S)
tn_naive = sum(abs(w_new[s] - w_old[s]) for s in S)
tn_pipe = py.loc[pd.Period("2009-03"), "turnover_top3"]
print("hand check B -- turnover at 2009-02-28: drift-aware by hand %.6f | pipeline %.6f | (ignoring drift: %.6f)"
      % (tn_hand, tn_pipe, tn_naive))
assert abs(tn_hand - tn_pipe) < 1e-9

# the two scripts download their own data: check the prices agree too
pr = pd.read_csv(D / "prices_r.csv", index_col=0, parse_dates=True)
rel = ((px.reindex(pr.index) - pr).abs() / pr).max().max()
print("max relative price difference, Python (yfinance) vs R (quantmod): %.2e  rows py/r: %d/%d" % (rel, len(px), len(pr)))
assert len(px) == len(pr) and rel < 1e-5
assert diff.max() < 1e-5 and gd < 0.011
print("ALL CHECKS PASSED")
