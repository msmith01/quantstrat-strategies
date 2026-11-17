"""4.3 R-squared strategy on Yahoo US equity mutual funds, 2012-2026 (deduplicated universe).
Writes data/results_funds_python.json. Everything the article quotes comes from that file."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QS_RAW_DIR", str(ROOT / "data" / "raw"))
import r_squared as R

FIRST = pd.Timestamp("2012-01-31")           # first month-end signal; returns start the month after
PX_START = "2009-06-01"                      # a 504-day window needs two years before the first signal
D = ROOT / "data/funds"
ded = pd.read_csv(D / "ag_prices.csv", nrows=0).columns[1:].tolist()      # the same active (non-index) funds as the annual study
px = pd.read_parquet(D / "fund_prices.parquet").loc[PX_START:, ded]
px["SPY"] = R.spy_prices(PX_START)
ff = R.download_factors()
last = min(px.index[-1], ff.index[-1], pd.Timestamp("2025-12-31"))      # same end as the annual study; 2026 is partial
px, ff = px.loc[:last], ff.loc[:last]
F = ded
R.UNIVERSE = F


def fit(window=R.WINDOW, lag=0):
    """Rolling 4-factor fit. A fund needs a complete window of daily returns. Returns alpha, t(alpha), R2 by month-end."""
    d = R.pct(px[F]).join(ff, how="inner")
    y = d[F].sub(d["RF"], axis=0)
    X = d[R.FACTORS].to_numpy()
    yv = y.to_numpy()
    ok_all = np.isfinite(yv)
    A, T, Q = {}, {}, {}
    for t in d.groupby(d.index.to_period("M")).tail(1).index:
        i = d.index.get_loc(t) - lag
        if i + 1 < window:
            continue
        s = slice(i + 1 - window, i + 1)
        ok = ok_all[s].all(axis=0)
        if ok.sum() < 10:
            continue
        b, tt, rr = R.ols(yv[s][:, ok], X[s])
        m = t + pd.offsets.MonthEnd(0)
        for dct, v in ((A, b[0]), (T, tt[0]), (Q, rr)):
            r = np.full(len(F), np.nan)
            r[ok] = v
            dct[m] = pd.Series(r, index=F)
    return pd.DataFrame(A).T, pd.DataFrame(T).T, pd.DataFrame(Q).T


px_same = R.to_monthly(px)
px_next = R.next_close_prices(px)


def run(alpha, r2, variant, pxm, cost=R.COST_BPS, borrow=0.0):
    a, q = alpha.reindex(pxm.index), r2.reindex(pxm.index)
    lg, sh = R.legs(a, q, variant)
    o = R.run_legs(pxm, lg, sh, cost_bps=cost, start=FIRST)[0]
    if borrow and sh.to_numpy().any():
        o["net"] = o["net"] - borrow / 100 / 12
    return o, lg, sh


def bench(alpha, pxm, cost=R.COST_BPS):
    ew = alpha.reindex(pxm.index).notna()
    return R.run_legs(pxm, ew, ew & False, cost_bps=cost, start=FIRST)[0]


def st(series, idx, label):
    return R.stats(series.reindex(idx), label)


out = {"params": {"funds": len(F), "first_signal": str(FIRST.date()), "last_factor_date": str(ff.index[-1].date()),
                  "window": R.WINDOW, "cost_bps": R.COST_BPS, "last_price": str(px.index[-1].date())}}
alpha, tstat, r2 = fit()
res = {v: run(alpha, r2, v, px_same) for v in ("r2_alpha", "long_leg", "alpha_only", "r2_only")}
idx = res["r2_alpha"][0].index
ewb = bench(alpha, px_same)
spy_m = R.pct(px_same["SPY"])
out["turnover_pct"] = {}
for v, (o, _, _) in res.items():
    out[v] = st(o["net"], idx, v)
    out[v + "_gross"] = st(o["gross"], idx, v + " gross")
    out["turnover_pct"][v] = round(float(o["turnover"].reindex(idx).mean()) * 100, 1)
out["equal_weight"] = st(ewb["net"], idx, "equal weight")
out["spy"] = st(spy_m, idx, "SPY")
out["turnover_pct"]["equal_weight"] = round(float(ewb["turnover"].reindex(idx).mean()) * 100, 1)
out["leg_sizes"] = {"avg_long": round(float(res["r2_alpha"][1].loc[idx].sum(axis=1).mean()), 1),
                    "avg_short": round(float(res["r2_alpha"][2].loc[idx].sum(axis=1).mean()), 1),
                    "avg_ranked": round(float(alpha.reindex(idx).notna().sum(axis=1).mean()), 0)}
out["diagnostics"] = {
    "avg_rank_corr_alpha_r2": round(float(np.nanmean([alpha.loc[t].rank().corr(r2.loc[t].rank()) for t in idx if t in alpha.index])), 3),
    "mean_r2": round(float(np.nanmean(r2.reindex(idx).to_numpy())), 3)}
series = {"r2_alpha": res["r2_alpha"][0]["net"], "long_leg": res["long_leg"][0]["net"],
          "alpha_only": res["alpha_only"][0]["net"], "r2_only": res["r2_only"][0]["net"],
          "equal_weight": ewb["net"], "spy": spy_m}
cum = pd.DataFrame({k: (1 + v.reindex(idx)).cumprod() * 100 for k, v in series.items()})
cum.to_csv(ROOT / "data/monthly_cum_funds.csv")
out["annual_returns_pct"] = {k: {str(y): round(float(((1 + g).prod() - 1) * 100), 2)
                                 for y, g in v.reindex(idx).groupby(idx.year)} for k, v in series.items()}
sub = {}
for name, a, b in (("2012-2019", "2012-01-01", "2019-12-31"), ("2020-2025", "2020-01-01", "2025-12-31")):
    ii = idx[(idx >= a) & (idx <= b)]
    sub[name] = {k: st(series[k], ii, k) for k in ("r2_alpha", "long_leg", "equal_weight", "spy")}
out["subperiods"] = sub
grid = {}
for w in (126, 252, 504):
    a_, t_, q_ = (alpha, tstat, r2) if w == 252 else fit(w)
    for v in ("r2_alpha", "long_leg", "alpha_only", "r2_only"):
        grid[f"{w}_{v}"] = st(run(a_, q_, v, px_same)[0]["net"], idx, f"{w} {v}")
out["grid"] = grid
timing = {}
for lag in (0, 5, 21):
    a_, t_, q_ = (alpha, tstat, r2) if lag == 0 else fit(R.WINDOW, lag)
    for ex, pm in (("same", px_same), ("next", px_next)):
        o = run(a_, q_, "r2_alpha", pm)[0]
        lgo = run(a_, q_, "long_leg", pm)[0]
        e = bench(a_, pm)
        timing[f"lag{lag}_{ex}"] = {"r2_alpha": st(o["net"], idx, "r2_alpha"), "long_leg": st(lgo["net"], idx, "long_leg"),
                                    "equal_weight": st(e["net"], idx, "ew"), "spy": st(R.pct(pm["SPY"]), idx, "spy")}
out["timing"] = timing
out["costs"] = {str(c): st(run(alpha, r2, "r2_alpha", px_same, cost=c)[0]["net"], idx, f"cost{c}") for c in (0, 10, 25, 50)}
out["borrow"] = {str(b): st(run(alpha, r2, "r2_alpha", px_same, borrow=b)[0]["net"], idx, f"borrow{b}") for b in (0, 1, 2, 5)}

mf = (1 + ff).resample("ME").prod() - 1
Xm = mf.reindex(idx)[R.FACTORS]


def reg(s):
    y = s.reindex(idx).to_numpy()
    X = np.column_stack([np.ones(len(idx)), Xm.to_numpy()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"alpha_pct_yr": round(float(b[0] * 1200), 2), "t": round(float(b[0] / se[0]), 2),
            "betas": {n: round(float(v), 3) for n, v in zip(R.FACTORS, b[1:])},
            "r2": round(float(1 - e @ e / ((y - y.mean()) @ (y - y.mean()))), 3)}


rf_m = (1 + ff["RF"]).resample("ME").prod() - 1
out["factor_regression"] = {"r2_alpha": reg(series["r2_alpha"]),
                            "long_leg_excess": reg(series["long_leg"] - rf_m.reindex(idx)),
                            "equal_weight_excess": reg(series["equal_weight"] - rf_m.reindex(idx)),
                            "long_leg_minus_equal_weight": reg(series["long_leg"] - series["equal_weight"])}
json.dump(out, open(ROOT / "data/results_funds_python.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("params", "r2_alpha", "long_leg", "equal_weight", "spy", "turnover_pct",
                                      "leg_sizes", "factor_regression")}, indent=1, default=str)[:3500])
