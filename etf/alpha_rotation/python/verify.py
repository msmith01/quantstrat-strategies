"""Verification: (1) Python vs R: monthly returns, every month's alpha for every ETF, both regression
tables, the results table, every sensitivity cell, prices and factors; (2) hand re-computation, from the raw
prices and factors with a third tool (statsmodels OLS) and plain loops: one set of alphas and the top-3
choice, one month's return, one month's turnover."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
S = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

py = pd.read_csv(D / "monthly_returns_python.csv", index_col=0, parse_dates=True)
r = pd.read_csv(D / "monthly_returns_r.csv")
r.index = pd.PeriodIndex(r["month"], freq="M"); py.index = py.index.to_period("M")
assert len(py) == len(r), (len(py), len(r))
cols = ["alpha3_net", "alpha3_gross", "alpha1_net", "mom3_net", "ew_net", "spy", "turnover_alpha3"]
diff = (py[cols] - r[cols]).abs().max()
print("max |Python - R| per monthly series:\n", diff.to_string())

# every month's alpha, every ETF (daily regressions, 12 months x 9 ETFs x ~320 months)
ap = pd.read_csv(D / "alphas_python.csv", index_col=0, parse_dates=True); ap.index = ap.index.to_period("M")
ar = pd.read_csv(D / "alphas_r.csv"); ar.index = pd.PeriodIndex(ar["month"], freq="M"); ar = ar[S]
ap, ar = ap.reindex(ar.index), ar
both = ap.notna() & ar.notna()
assert (ap.notna() == ar.notna()).all().all(), "alpha availability differs"
alpha_diff = float((ap[both] - ar[both]).abs().max().max())
print("max |alpha Python - alpha R| over %d month-ETF alphas (daily alpha): %.2e" % (int(both.sum().sum()), alpha_diff))

for name in ("full", "latest"):
    a = pd.read_csv(D / f"regression_{name}_python.csv", index_col=0)
    b = pd.read_csv(D / f"regression_{name}_r.csv", index_col=0)
    dd = float((a - b[a.columns]).abs().max().max())
    print(f"max diff, regression table ({name}), Python numpy vs R lm(): {dd:.2e}")
    assert dd < 1e-3, (name, dd)   # yfinance vs quantmod prices differ by ~1e-6, which moves alpha by ~1e-4 %

jp = json.load(open(D / "results_python.json"))
rt = pd.read_csv(D / "results_table_r.csv", index_col=0)
rs = pd.read_csv(D / "sensitivity_r.csv")
gd = max(abs(jp["sensitivity"][f"W{int(x.window)}_N{int(x.n)}"][m] - x[m])
         for _, x in rs.iterrows() for m in ("cagr_pct", "sharpe"))
print("max sensitivity-grid diff (12 cells x CAGR,Sharpe):", gd)
# the grid must cover identical months in every cell, in both languages
assert set(rs["months"]) == {jp["sensitivity_sample"]["months"]}, ("R grid months", set(rs["months"]))
assert {v["months"] for v in jp["sensitivity"].values()} == {jp["sensitivity_sample"]["months"]}, "Python grid months differ"
ss = pd.read_csv(D / "sensitivity_sample_r.csv").iloc[0]
assert ss["start"] == jp["sensitivity_sample"]["start"][:7] and ss["end"] == jp["sensitivity_sample"]["end"][:7]
assert abs(ss["ew_cagr_pct"] - jp["sensitivity_sample"]["equal_weight"]["cagr_pct"]) < 0.011
print("sensitivity grid: %d months in every cell, %s to %s, Python == R" % (ss["months"], ss["start"], ss["end"]))
tr = pd.read_csv(D / "timing_test_r.csv"); tp = jp["timing_test"]
assert set(tr["months"]) == {tp["months"]}, (set(tr["months"]), tp["months"])
td = max(abs(tp["variants"][r_.variant][m] - r_[m]) for _, r_ in tr.iterrows() for m in ("cagr_pct", "sharpe", "max_drawdown_pct"))
print("timing test: %d months in every variant, max Python-R diff %.4f" % (tp["months"], td))
assert td < 0.011
tb = pd.read_csv(D / "timing_benchmarks_r.csv").set_index("name")
for k in ("equal_weight", "mom3", "equal_weight_next", "mom3_next"):
    assert abs(tp[k]["cagr_pct"] - tb.loc[k, "cagr_pct"]) < 0.011 and abs(tp[k]["sharpe"] - tb.loc[k, "sharpe"]) < 0.011, k
print("timing-test benchmarks (equal-weight, momentum; month-end and next-day close) on the same months: Python == R")
cmp = {"alpha3": "Alpha top 3", "alpha1": "Alpha top 1", "mom3": "Momentum top 3 (4.1)",
       "equal_weight": "Equal-weight 9 (net)", "spy": "SPY"}
for k, c in cmp.items():
    for m, row in (("cagr_pct", "CAGR (%)"), ("sharpe", "Sharpe"), ("max_drawdown_pct", "Max drawdown (%)")):
        assert abs(jp[k][m] - rt.loc[row, c]) < 0.011, (k, m, jp[k][m], rt.loc[row, c])

px = pd.read_csv(D / "prices_python.csv", index_col=0, parse_dates=True)
ff = pd.read_csv(D / "factors_python.csv", index_col=0, parse_dates=True)
pxm = px.resample("ME").last()
ret_d = px[S].pct_change()
dd_ = pd.concat([ret_d, ff], axis=1, join="inner").dropna()


def alphas_by_statsmodels(date, window=252):
    """alpha of every ETF at `date`: statsmodels OLS on the last `window` trading days, from raw data."""
    last = dd_.index[dd_.index <= date][-1]                  # last trading day of the month
    blk = dd_.loc[:last].iloc[-window:]
    X = sm.add_constant(blk[["MKT", "SMB", "HML"]])
    return {s: sm.OLS(blk[s] - blk["RF"], X).fit().params["const"] for s in S}


def top_n(date, n=3):
    a = alphas_by_statsmodels(date)
    top = sorted(S, key=lambda s: -a[s])[:n]
    return {s: (1 / n if s in top else 0.0) for s in S}, top, a


# hand check A: signal at 2008-12-31, earns Jan 2009
d0 = pd.Timestamp("2008-12-31"); w0, top, a0 = top_n(d0)
i = list(pxm.index).index(d0)
hand = sum(w0[s] * (pxm[s].iloc[i + 1] / pxm[s].iloc[i] - 1) for s in S)
pipe_a = ap.loc[pd.Period("2008-12")]
print("hand check A -- top 3 alphas at 2008-12-31 (statsmodels):", top, {s: round(float(a0[s]) * 252 * 100, 2) for s in top}, "(% per year)")
print("max |statsmodels alpha - pipeline alpha| at 2008-12: %.2e" % max(abs(a0[s] - pipe_a[s]) for s in S))
print("hand gross Jan-2009 return: %.6f   pipeline: %.6f" % (hand, py.loc[pd.Period("2009-01"), "alpha3_gross"]))
assert max(abs(a0[s] - pipe_a[s]) for s in S) < 1e-10
assert abs(hand - py.loc[pd.Period("2009-01"), "alpha3_gross"]) < 1e-9

# hand check B: turnover when trading at 2009-02-28 (new weights vs last month's weights after drifting)
d1 = pd.Timestamp("2009-01-31"); d2 = pd.Timestamp("2009-02-28")
w_old, _, _ = top_n(d1); w_new, top_new, _ = top_n(d2)
j = list(pxm.index).index(d1)
grown = {s: w_old[s] * (pxm[s].iloc[j + 1] / pxm[s].iloc[j]) for s in S}      # value of each holding during Feb
tot = sum(grown.values()); drifted = {s: grown[s] / tot for s in S}
tn_hand = sum(abs(w_new[s] - drifted[s]) for s in S)
tn_naive = sum(abs(w_new[s] - w_old[s]) for s in S)
tn_pipe = py.loc[pd.Period("2009-03"), "turnover_alpha3"]
print("hand check B -- turnover at 2009-02-28: drift-aware by hand %.6f | pipeline %.6f | (ignoring drift: %.6f)"
      % (tn_hand, tn_pipe, tn_naive))
assert abs(tn_hand - tn_pipe) < 1e-9

# hand check C: the factor-availability variant. Alpha from the window ending 5 trading days before month-end
# (statsmodels, from raw data) must equal the pipeline's; the trade is at the next trading day's close.
import sys; sys.path.insert(0, str(ROOT / "python"))
import alpha_rotation as ar
lagged = ar.rolling_alpha(px, ff, 252, 5)[0].loc[pd.Timestamp("2008-12-31")]
last = dd_.index[dd_.index <= d0][-1]; k = dd_.index.get_loc(last) - 5
blk = dd_.iloc[k - 251:k + 1]
Xb = sm.add_constant(blk[["MKT", "SMB", "HML"]])
a5 = {s: sm.OLS(blk[s] - blk["RF"], Xb).fit().params["const"] for s in S}
print("hand check C -- max |statsmodels lag-5 alpha - pipeline| at 2008-12: %.2e" % max(abs(a5[s] - lagged[s]) for s in S))
assert max(abs(a5[s] - lagged[s]) for s in S) < 1e-10
top5 = sorted(S, key=lambda s: -a5[s])[:3]
day0 = px.index[px.index > d0][0]; day1 = px.index[px.index > pd.Timestamp("2009-01-31")][0]   # next trading day after each month-end
hand5 = sum((1 / 3) * (px[s].loc[day1] / px[s].loc[day0] - 1) for s in top5)
print("lag-5 top 3 at 2008-12-31:", top5, "| bought at the %s close, sold at the %s close: gross %.6f" % (day0.date(), day1.date(), hand5))

# the two scripts download their own data: check the prices and factors agree too
pr = pd.read_csv(D / "prices_r.csv", index_col=0, parse_dates=True)
rel = ((px.reindex(pr.index) - pr).abs() / pr).max().max()
print("max relative price difference, Python (yfinance) vs R (quantmod): %.2e  rows py/r: %d/%d" % (rel, len(px), len(pr)))
fr = pd.read_csv(D / "factors_r.csv", index_col=0, parse_dates=True)
fdiff = float((ff.reindex(fr.index) - fr).abs().max().max())
print("max factor difference, Python vs R: %.2e  rows py/r: %d/%d" % (fdiff, len(ff), len(fr)))
assert len(px) == len(pr) and rel < 1e-5 and len(ff) == len(fr) and fdiff < 1e-12
assert diff.max() < 1e-5 and gd < 0.011 and alpha_diff < 1e-6
print("ALL CHECKS PASSED")
