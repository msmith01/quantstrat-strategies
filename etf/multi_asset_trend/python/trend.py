"""4.6 Strategy: Multi-asset trend following -- implementation of Kakushadze & Serur (2018), "151 Trading Strategies", Section 4.6, Eq. (371)-(373).

Each month-end: R_i = P_i(t)/P_i(t-T) - 1 over T months; keep ETFs with R_i > 0 and (optionally) P_i > its 200-day moving average;
weights w_i = g1*R_i (371), g2*R_i/sigma_i (372) or g3*R_i/sigma_i^2 (373), normalised to sum to 1 over the kept ETFs.
sigma_i = annualised standard deviation of daily returns over the previous 252 trading days. If no ETF qualifies the portfolio holds the Treasury bill ETF (cash).
Trade at the month-end close, earn the next month, pay costs on drift-adjusted turnover (as in the 4.1 article).
Writes data/results_python.json and data/monthly_returns.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "DBC"]
CASH = "SHV"
COST_BPS = 10
MA_DAYS, VOL_DAYS = 200, 252
MAIN = dict(T=12, ma=True, wt="371", cap=None)


def load():
    px = pd.read_csv(ROOT / "data/prices.csv", index_col=0, parse_dates=True)
    daily = px[UNIVERSE + [CASH]]
    pm = daily.resample("ME").last()
    # inception months: a fund has no price before it exists; resample keeps NaN
    pm = pm.where(daily.resample("ME").count() > 0)
    ret_d = daily[UNIVERSE].pct_change(fill_method=None)
    sig = ret_d.rolling(VOL_DAYS, min_periods=VOL_DAYS).std().resample("ME").last() * np.sqrt(252)
    ma_ok = (daily[UNIVERSE] > daily[UNIVERSE].rolling(MA_DAYS, min_periods=MA_DAYS).mean()).where(
        daily[UNIVERSE].rolling(MA_DAYS, min_periods=MA_DAYS).mean().notna()).resample("ME").last()
    return pm, sig, ma_ok


def load_exec():
    """Close of the first trading day after each month-end (next-day execution test), indexed by the month-end it follows."""
    px = pd.read_csv(ROOT / "data/prices.csv", index_col=0, parse_dates=True)[UNIVERSE + [CASH]]
    fm = px.resample("MS").first()
    fm.index = fm.index - pd.Timedelta(days=1)
    return fm


def cap_weights(w, cap):
    """Iteratively cap each weight at `cap`; weight that cannot be placed under the cap goes to cash."""
    w = w.copy()
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum(); w[over] = cap
        free = (w < cap - 1e-12) & (w > 0)
        if free.sum() == 0 or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def weights(pm, sig, ma_ok, T, ma, wt, cap):
    rcum = pm[UNIVERSE] / pm[UNIVERSE].shift(T) - 1
    ok = rcum > 0
    if ma:
        ok = ok & ma_ok.fillna(False).astype(bool)
    ok = ok & sig.notna() & rcum.notna()
    raw = {"371": rcum, "372": rcum / sig, "373": rcum / sig ** 2, "ew": rcum * 0 + 1}[wt]
    raw = raw.where(ok, 0.0).clip(lower=0)
    s = raw.sum(axis=1)
    w = raw.div(s.where(s > 0), axis=0).fillna(0.0)
    if cap is not None:
        w = w.apply(lambda r: cap_weights(r, cap), axis=1)
    return w, rcum


def start_month(pm, sig):
    rcum = pm[UNIVERSE] / pm[UNIVERSE].shift(12) - 1
    return rcum.dropna(how="any").index[0]


def run(pm, sig, ma_ok, T=12, ma=True, wt="371", cap=None, cost_bps=COST_BPS, start=None, pm_ex=None):
    w, rcum = weights(pm, sig, ma_ok, T, ma, wt, cap)
    px = pm if pm_ex is None else pm_ex
    r = px[UNIVERSE].pct_change(fill_method=None)
    rc = px[CASH].pct_change(fill_method=None)
    wl = w.shift(1)                                   # trade at month-end t, earn t+1
    cash_w = (1 - wl.sum(axis=1)).clip(lower=0)
    gross = (wl * r).sum(axis=1, min_count=1) + cash_w * rc
    drift = wl * (1 + r)
    tot = drift.sum(axis=1) + cash_w * (1 + rc)
    drift = drift.div(tot, axis=0)
    drift_cash = cash_w * (1 + rc) / tot
    turn = (wl - drift.shift(1)).abs().sum(axis=1) + (cash_w - drift_cash.shift(1)).abs()   # includes the SHV leg
    first = pm.index[pm.index > start][0]             # first month earned = month after the first signal
    turn.loc[first] = wl.loc[first].sum() + cash_w.loc[first]   # initial purchase of the whole portfolio
    net = gross - turn * cost_bps / 1e4
    held = (wl > 0).sum(axis=1)
    out = pd.DataFrame({"gross": gross, "rf": rc, "net": net, "turnover": turn, "cash": cash_w, "held": held}).loc[first:]
    return out.dropna(subset=["net"]), w


def stats(ret, rf=None):
    ret = ret.dropna(); yrs = len(ret) / 12
    eq = np.r_[1.0, (1 + ret).cumprod().to_numpy()]
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=(eq[-1] ** (1 / yrs) - 1) * 100, vol=ret.std(ddof=1) * np.sqrt(12) * 100,
                sharpe=ret.mean() / ret.std(ddof=1) * np.sqrt(12),
                sharpe_rf=(np.nan if rf is None else ((ret - rf.reindex(ret.index)).mean() / (ret - rf.reindex(ret.index)).std(ddof=1) * np.sqrt(12))), maxdd=dd.min() * 100, months=len(ret),
                start=str(ret.index[0].date()), end=str(ret.index[-1].date()))


def main():
    pm, sig, ma_ok = load()
    start = start_month(pm, sig)
    res = {"universe": UNIVERSE, "cash": CASH}
    # benchmarks, on the same months
    r = pm[UNIVERSE].pct_change(fill_method=None)
    # true equal-weight: hold all ten every month
    wl = pd.DataFrame(1 / len(UNIVERSE), index=pm.index, columns=UNIVERSE)
    def fixed(wdf, name):
        wl = wdf.shift(1); gross = (wl * r).sum(axis=1, min_count=1)
        drift = wl * (1 + r); drift = drift.div(drift.sum(axis=1), axis=0)
        turn = (wl - drift.shift(1)).abs().sum(axis=1)
        f = wl.dropna(how="all").index[0]; f = max(f, ret_first(wl, start)); turn.loc[f] = wl.loc[f].sum()
        return (gross - turn * COST_BPS / 1e4).loc[f:].dropna()
    def ret_first(wl, st):
        return wl.index[wl.index > st][0]
    first = pm.index[pm.index > start][0]
    bm = {}
    bm["Equal weight, all ten"] = fixed(wl, "ew")
    spy = r["SPY"].loc[first:]; bm["SPY buy and hold"] = spy
    w6040 = pd.DataFrame(0.0, index=pm.index, columns=UNIVERSE); w6040["SPY"] = 0.6; w6040["IEF"] = 0.4
    bm["60/40 SPY and IEF, monthly"] = fixed(w6040, "6040")
    cashr = pm[CASH].pct_change(fill_method=None).loc[first:]; bm["Treasury bill ETF"] = cashr
    res["benchmarks"] = {k: stats(v, cashr) for k, v in bm.items()}
    rets = {k: v for k, v in bm.items()}
    # main variants and the sensitivity grid
    grid = {}; gridret = {}
    for T in (6, 9, 12):
        for ma in (False, True):
            for wt in ("371", "372", "373"):
                o, _ = run(pm, sig, ma_ok, T=T, ma=ma, wt=wt, start=start)
                s = stats(o["net"], cashr); s["gross_cagr"] = stats(o["gross"])["cagr"]; s["turnover"] = o["turnover"].mean() * 100
                s["avg_held"] = o["held"].mean(); s["cash_months_pct"] = (o["cash"] > 0.999).mean() * 100
                grid[f"T{T}|ma{int(ma)}|{wt}"] = s; gridret[f"T{T}|ma{int(ma)}|{wt}"] = o["net"]
                if T == 12 and ma:
                    rets[f"Eq.({ {'371': 371, '372': 372, '373': 373}[wt] })"] = o["net"]
    res["grid"] = grid; pd.DataFrame(gridret).to_csv(ROOT / "data/grid_returns_python.csv")
    caps = {}
    for wt in ("371", "372", "373"):
        o, _ = run(pm, sig, ma_ok, T=12, ma=True, wt=wt, cap=0.35, start=start)
        s = stats(o["net"], cashr); s["turnover"] = o["turnover"].mean() * 100; s["cash_months_pct"] = (o["cash"] > 0.999).mean() * 100
        s["avg_cash_pct"] = o["cash"].mean() * 100; s["cash_any_pct"] = (o["cash"] > 0.001).mean() * 100
        u, _ = run(pm, sig, ma_ok, T=12, ma=True, wt=wt, start=start)
        s["uncapped_avg_cash_pct"] = u["cash"].mean() * 100; s["uncapped_cash_any_pct"] = (u["cash"] > 0.001).mean() * 100
        caps[wt] = s
    res["cap35"] = caps
    costs = {}
    for c in (0, 10, 25, 50):
        o, _ = run(pm, sig, ma_ok, T=12, ma=True, wt="372", cost_bps=c, start=start)
        costs[str(c)] = stats(o["net"])["cagr"]
    res["costs_372"] = costs
    pmx = load_exec().reindex(pm.index)
    nd = {}
    for wt in ("371", "372", "373"):
        o, _ = run(pm, sig, ma_ok, T=12, ma=True, wt=wt, start=start, pm_ex=pmx)
        nd[wt] = stats(o["net"], cashr)
    res["next_day"] = nd
    # yearly returns for the three main variants and benchmarks
    df = pd.DataFrame(rets)
    df.to_csv(ROOT / "data/monthly_returns.csv")
    yr = (1 + df).groupby(df.index.year).prod() - 1
    res["yearly"] = (yr * 100).round(2).to_dict(orient="index")
    res["start"] = str(first.date())
    json.dump(res, open(ROOT / "data/results_python.json", "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("benchmarks",)}, indent=1)[:1500])
    for k, v in grid.items():
        print(k, round(v["cagr"], 2), round(v["sharpe"], 2), round(v["maxdd"], 1), round(v["turnover"]), round(v["avg_held"], 1), round(v["cash_months_pct"]))


if __name__ == "__main__":
    main()
