"""4.5 Leveraged ETFs: volatility decay of leveraged and inverse ETFs, and the book's strategy of shorting a leveraged ETF together with its
inverse (same leverage, same index) and holding a Treasury bill ETF. Writes data/results_python.json and data/*.csv.
Analysis 1 follows Avellaneda and Zhang (2010): a daily-reset L-times ETF compounds as ln(V_T/V_0) = L ln(S_T/S_0) - L(L-1)/2 * sum(r_u^2) per period.
Leverage history: ERX/ERY were +-3x until 31 Mar 2020 and +-2x after; UYG/SKF tracked the Dow Jones US Financials until 17 Mar 2023 (IYF as proxy), then the
S&P Financial Select Sector index (XLF)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASH = "SHV"
ERX_CHANGE = pd.Timestamp("2020-04-01")      # first day at +-2x
UYG_CHANGE = pd.Timestamp("2023-03-20")      # first trading day on the S&P Financial Select Sector index
SPEC = {"borrow_pct": [0.0, 1.0, 3.0, 5.0], "rebalance": ["daily", "monthly", "annual"], "min_days_year": 120, "full_year_days": 240, "trade_bps": 5.0,
        "margin_rate": {"2x": 0.6, "3x": 0.9},
        "stress": {"2008-09": ("2008-09-01", "2009-03-31"), "Covid crash 2020": ("2020-02-19", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}}
PAIRS = [("S&P 500 2x", "SPY", "SSO", "SDS", 2), ("S&P 500 3x", "SPY", "UPRO", "SPXU", 3), ("Nasdaq-100 2x", "QQQ", "QLD", "QID", 2), ("Nasdaq-100 3x", "QQQ", "TQQQ", "SQQQ", 3),
         ("Russell 2000 2x", "IWM", "UWM", "TWM", 2), ("Russell 2000 3x", "IWM", "TNA", "TZA", 3), ("Dow 30 2x", "DIA", "DDM", "DXD", 2), ("Dow 30 3x", "DIA", "UDOW", "SDOW", 3),
         ("S&P MidCap 2x", "MDY", "MVV", "MZZ", 2), ("S&P MidCap 3x", "MDY", "UMDD", "SMDD", 3), ("Financials 2x", "XLF", "UYG", "SKF", 2), ("Financials 3x", "XLF", "FAS", "FAZ", 3),
         ("Energy 3x to 2x", "XLE", "ERX", "ERY", 3), ("Technology 3x", "XLK", "TECL", "TECS", 3), ("Semiconductors 3x", "SOXX", "SOXL", "SOXS", 3),
         ("20+yr Treasuries 3x", "TLT", "TMF", "TMV", 3)]

px = pd.read_csv(ROOT / "data/prices_all.csv", index_col=0, parse_dates=True, float_precision="round_trip").sort_index()


def pair_frame(p):
    name, u, lp, lm, L = p
    c = [u, lp, lm, CASH] + (["IYF"] if name == "Financials 2x" else [])
    if any(x not in px for x in c):
        return None
    r = px[c].dropna().pct_change().iloc[1:]
    if name == "Financials 2x":                      # Dow Jones US Financials proxy (IYF) before the benchmark change, XLF after
        r[u] = np.where(r.index < UYG_CHANGE, r["IYF"], r[u])
        r = r[[u, lp, lm, CASH]]
    r.columns = ["u", "p", "m", "c"]
    r["L"] = float(L)
    if name.startswith("Energy"):
        r.loc[r.index >= ERX_CHANGE, "L"] = 2.0
    return r


F = {p[0]: pair_frame(p) for p in PAIRS}
F = {k: v for k, v in F.items() if v is not None and len(v) > 500}
meta = {p[0]: {"proxy": p[1], "long": p[2], "inverse": p[3], "L": p[4], "start": str(F[p[0]].index.min().date()), "days": len(F[p[0]])} for p in PAIRS if p[0] in F}
out = {"params": {"spec": SPEC, "pairs": meta, "last_date": str(max(v.index.max() for v in F.values()).date()), "cash": CASH}}

# ---------- 1. volatility decay regression ----------
rows = []
for p in PAIRS:
    if p[0] not in F:
        continue
    f = F[p[0]]
    for leg, sign in (("p", 1), ("m", -1)):
        for y, g in f.groupby(f.index.year):
            if len(g) < SPEC["min_days_year"] or g.L.nunique() > 1:        # a year with a change of leverage (ERX/ERY 2020) is left out
                continue
            lev = sign * float(g.L.iloc[0])
            rows.append({"pair": p[0], "etf": p[2] if leg == "p" else p[3], "L": lev, "year": int(y), "n": len(g),
                         "y": float(np.log1p(g[leg]).sum() - lev * np.log1p(g.u).sum()), "x": float(lev * (lev - 1) / 2 * (g.u ** 2).sum()),
                         "etf_ret": float(np.expm1(np.log1p(g[leg]).sum())), "u_ret": float(np.expm1(np.log1p(g.u).sum()))})
D = pd.DataFrame(rows)
D.to_csv(ROOT / "data/decay_etf_years.csv", index=False)


def ols(y, x, cluster):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(res @ res / (n - k) * XtXi))
    meat = np.zeros((k, k))
    for g in np.unique(cluster):
        s = X[cluster == g].T @ res[cluster == g]
        meat += np.outer(s, s)
    G = len(np.unique(cluster))
    se_c = np.sqrt(np.diag(XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)))
    return b, se, se_c, float(1 - (res @ res) / ((y - y.mean()) ** 2).sum())


dec = {}
FULL = D[D.n >= SPEC["full_year_days"]]
for label, sub in (("all", D), ("full", FULL), ("long", FULL[FULL.L > 0]), ("inverse", FULL[FULL.L < 0]), ("2x", FULL[FULL.L.abs() == 2]), ("3x", FULL[FULL.L.abs() == 3])):
    b, se, sec, r2 = ols(sub.y.to_numpy(), sub.x.to_numpy(), sub.year.to_numpy())
    dec[label] = {"n": int(len(sub)), "intercept": float(b[0]), "intercept_se": float(se[0]), "slope": float(b[1]), "slope_se": float(se[1]), "slope_se_cluster_year": float(sec[1]),
                  "intercept_se_cluster_year": float(sec[0]), "r2": r2, "n_years": int(sub.year.nunique())}
out["decay_regression"] = dec
out["decay_summary"] = {"n_etf_years_all": int(len(D)), "n_etf_years_full": int(len(FULL))}


# ---------- 2. the book's strategy: short the leveraged ETF and its inverse, hold the Treasury ETF ----------
def sim(f, freq, borrow_pct=0.0, trade_bps=0.0, margin=False, share=1.0):
    """Per $1 of equity at the start of each period: short $0.5 of each leg ($1 gross short), the sale proceeds in the Treasury ETF, $1 of collateral
    earning nothing. Positions reset to target at the first day of each period (daily, monthly or annual); between resets the legs drift.
    borrow_pct: annual fee on the short value, charged daily. trade_bps: cost on the value traded at each reset. share: fraction of the Treasury
    ETF return credited on the proceeds (0 = no interest). margin: maintenance requirement on the short value (0.6 for 2x, 0.9 for 3x); if equity falls
    below it the positions are closed that day and the equity stays in cash until the next reset."""
    key = {"daily": np.arange(len(f)), "monthly": f.index.year * 100 + f.index.month, "annual": f.index.year}[freq]
    p, m, c, Lr = f.p.to_numpy(), f.m.to_numpy(), f.c.to_numpy(), f.L.to_numpy()
    fee, n = borrow_pct / 100 / 252, len(f)
    r, E, prev, a, b, cc, col, closed, dead, liq = np.zeros(n), 1.0, None, 0.0, 0.0, 0.0, 1.0, False, False, 0
    for t in range(n):
        if dead:
            continue
        E_pre = E
        if prev is None or key[t] != prev:                                   # reset to target weights; pay for the trades
            traded = abs(0.5 * E - a) + abs(0.5 * E - b)
            E = E - trade_bps / 1e4 * traded
            a, b, cc, col, closed, prev = 0.5 * E, 0.5 * E, E, E, False, key[t]
        if closed:
            r[t] = E / E_pre - 1
            continue
        fee_t = fee * (a + b)
        a, b, cc, col = a * (1 + p[t]), b * (1 + m[t]), cc * (1 + share * c[t]), col - fee_t
        En = col + cc - a - b
        if En <= 0:
            r[t], dead = -1.0, True
            continue
        if margin and En < (SPEC["margin_rate"]["3x"] if Lr[t] >= 3 else SPEC["margin_rate"]["2x"]) * (a + b):
            col, a, b, cc, closed = En, 0.0, 0.0, 0.0, True                    # forced liquidation at the close
            liq += 1
        r[t], E = En / E_pre - 1, En
    return pd.Series(r, index=f.index), liq


def stats(r):
    r = r.dropna()
    eqc = pd.concat([pd.Series([1.0]), (1 + r.reset_index(drop=True)).cumprod()], ignore_index=True)       # start the curve at 1 so an opening loss counts
    yrs = len(r) / 252
    m = r.groupby(r.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    return {"cagr": float((eqc.iloc[-1] ** (1 / yrs) - 1) * 100) if eqc.iloc[-1] > 0 else -100.0, "vol": float(r.std() * np.sqrt(252) * 100),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"), "max_dd": float((eqc / eqc.cummax() - 1).min() * 100),
            "worst_day": float(r.min() * 100), "worst_month": float(m.min() * 100), "best_day": float(r.max() * 100),
            "pct_months_pos": float((m > 0).mean() * 100), "days": int(len(r)), "final": float(eqc.iloc[-1]), "wiped_out": bool((r <= -1.0).any())}


SCEN = {"base": {}, "no_interest": {"share": 0.0}, "trade_cost": {"trade_bps": SPEC["trade_bps"]}, "margin": {"margin": True},
        "all": {"share": 0.0, "trade_bps": SPEC["trade_bps"], "margin": True}}
res, rets = {}, {}
for name, f in F.items():
    res[name] = {}
    for freq in SPEC["rebalance"]:
        for bp in SPEC["borrow_pct"]:
            r_, _ = sim(f, freq, bp)
            res[name][f"{freq}|{bp:g}"] = stats(r_)
            rets[(name, freq, bp)] = r_
    res[name]["scen"] = {}
    for sc, kw in SCEN.items():
        for bp in (0.0, 3.0):
            r_, liq = sim(f, "monthly", bp, **kw)
            res[name]["scen"][f"{sc}|{bp:g}"] = {**stats(r_), "liquidations": liq}
            rets[(name, "monthly", bp, sc)] = r_
    res[name]["cash_only"] = stats(f.c)
out["strategy"] = res
KEYS = ("cagr", "vol", "sharpe", "max_dd", "worst_day", "worst_month", "pct_months_pos")
out["strategy_avg"] = {f"{freq}|{bp:g}": {k: float(np.nanmean([res[n][f"{freq}|{bp:g}"][k] for n in res])) for k in KEYS} for freq in SPEC["rebalance"] for bp in SPEC["borrow_pct"]}
out["strategy_counts"] = {"n_wiped_out_annual_0": int(sum(res[n]["annual|0"]["wiped_out"] for n in res)), "n_wiped_out_monthly_0": int(sum(res[n]["monthly|0"]["wiped_out"] for n in res)),
                          "n_pairs": len(res), "n_positive_daily_0": int(sum(res[n]["daily|0"]["cagr"] > 0 for n in res)), "n_positive_daily_3": int(sum(res[n]["daily|3"]["cagr"] > 0 for n in res)),
                          "cash_cagr": float(np.nanmean([res[n]["cash_only"]["cagr"] for n in res]))}


def combine(R):
    """Portfolio of all pairs with $1 put into each pair when it starts and no rebalancing between pairs; return on total equity."""
    E_prev = (1 + R.fillna(0)).cumprod().shift(1).fillna(1.0).where(R.notna())
    return (R * E_prev).sum(axis=1) / E_prev.sum(axis=1)


def portfolio(freq, bp, sc=None):
    return combine(pd.concat({n: (rets[(n, freq, bp, sc)] if sc else rets[(n, freq, bp)]) for n in F}, axis=1))


port, portm = {}, {}
for bp in SPEC["borrow_pct"]:
    port[bp], portm[bp] = portfolio("daily", bp), portfolio("monthly", bp)
    out.setdefault("portfolio", {})[f"{bp:g}"] = stats(port[bp])
    out.setdefault("portfolio_monthly", {})[f"{bp:g}"] = stats(portm[bp])
cash = combine(pd.concat({n: f.c for n, f in F.items()}, axis=1))
out["portfolio"]["cash"] = stats(cash)
out["portfolio_scen"] = {}
for sc in SCEN:
    for bp in (0.0, 3.0):
        pr = portfolio("monthly", bp, sc)
        out["portfolio_scen"][f"{sc}|{bp:g}"] = {**stats(pr), "liquidations": int(sum(res[n]["scen"][f"{sc}|{bp:g}"]["liquidations"] for n in F))}
start_all = max(f.index.min() for f in F.values())                              # common period: only the dates on which all pairs exist
cp = {f"{bp:g}": stats(portm[bp][start_all:]) for bp in (0.0, 3.0)}
cp["cash"], cp["start"] = stats(cash[start_all:]), str(start_all.date())
out["portfolio_common"] = cp
st = {}
for nm, (a, b) in SPEC["stress"].items():
    st[nm] = {}
    for bp in (0.0, 3.0):
        r = portm[bp][a:b]
        st[nm][f"{bp:g}"] = {"ret": float(((1 + r).prod() - 1) * 100), "worst_day": float(r.min() * 100),
                             "max_dd": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), "days": int(len(r))}
out["stress_monthly"] = st
out["portfolio_monthly_yearly"] = {f"{bp:g}": {str(y): float(((1 + x).prod() - 1) * 100) for y, x in portm[bp].groupby(portm[bp].index.year)} for bp in (0.0, 3.0)}
pd.DataFrame({f"net_{bp:g}": port[bp] for bp in SPEC["borrow_pct"]}).assign(cash=cash).to_csv(ROOT / "data/portfolio_daily.csv")
pd.DataFrame({f"net_{bp:g}": portm[bp] for bp in SPEC["borrow_pct"]}).to_csv(ROOT / "data/portfolio_monthly_daily.csv")
json.dump(out, open(ROOT / "data/results_python.json", "w"), indent=1, default=str)
print("decay", {k: (round(v["slope"], 3), round(v["slope_se"], 3), round(v["slope_se_cluster_year"], 3), v["n"]) for k, v in dec.items()})
print("portfolio daily", {k: round(v["cagr"], 2) for k, v in out["portfolio"].items()}, "monthly", {k: round(v["cagr"], 2) for k, v in out["portfolio_monthly"].items()})
print("scen", {k: (round(v["cagr"], 2), round(v["max_dd"], 1), v["liquidations"]) for k, v in out["portfolio_scen"].items()})
print("common", {k: (round(v["cagr"], 2) if isinstance(v, dict) else v) for k, v in cp.items()})
