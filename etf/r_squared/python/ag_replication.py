"""Replication of the empirical design of Amihud and Goyenko (2013), "Mutual Fund's R2 as Predictor of Performance",
Review of Financial Studies 26(3), on Yahoo US equity mutual funds, 2012-2026 (their sample: CRSP, 1990-2007).
Design (from the paper's Sections 2-5): for each fund and calendar year, regress daily excess returns on RM-Rf, SMB,
HML, MOM and their one-day lags (Dimson 1979); alpha, R2, RMSE, InfRatio = alpha/RMSE. R2 and alpha of year y-1 predict
alpha and InfRatio of year y. Tables reproduced: II, III (without fund-characteristic controls Yahoo does not have),
V (Fama-MacBeth), VIII (dependent 5x5 sort) and IX (independent 5x5 sort). Writes data/results_ag_python.json and
data/ag_fund_years.csv. Everything the article quotes comes from them."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QS_RAW_DIR", str(ROOT / "data" / "raw"))
import r_squared as R
from ag_replication_lib import cluster_ols, design

D = ROOT / "data/funds"
MIN_PRIOR, MIN_TEST = 125, 50                      # daily observations needed in year y-1 and in year y (paper, Section 3)
meta = pd.read_csv(D / "yahoo_us_equity_funds_deduped.csv").set_index("ticker")
is_index = meta["name"].str.contains(r"\bindex\b|\bind\b", case=False, regex=True)   # paper: names with "index" or "ind"
tick = meta.index[~is_index].tolist()
px = pd.read_parquet(D / "fund_prices.parquet").loc["2010-06-01":, tick]
spy = R.spy_prices("2010-06-01")
ff = R.download_factors()
last = min(px.index[-1], ff.index[-1])
px, ff, spy = px.loc[:last], ff.loc[:last], spy.loc[:last]
fac = ff[R.FACTORS]
Z = pd.concat([fac, fac.shift(1).add_suffix("_L")], axis=1)          # current and one-lag factors (Dimson)
ret = R.pct(px)
d = ret.join(Z, how="inner").join(ff["RF"]).dropna(subset=list(Z.columns) + ["RF"])
yrs = d.index.year


def year_fit(y):
    """Per fund: alpha (annualised %), R2, RMSE (daily), SDR (daily), InfRatio, n days for calendar year y."""
    m = yrs == y
    X = np.column_stack([np.ones(m.sum()), d.loc[m, list(Z.columns)].to_numpy()])
    rf = d.loc[m, "RF"].to_numpy()
    Y = d.loc[m, tick].to_numpy() - rf[:, None]
    out = {}
    for j, t in enumerate(tick):
        ok = np.isfinite(Y[:, j])
        n = int(ok.sum())
        if n < MIN_TEST:
            continue
        A, y_ = X[ok], Y[ok, j]
        b = np.linalg.lstsq(A, y_, rcond=None)[0]
        e = y_ - A @ b
        ssr, sst = float(e @ e), float(((y_ - y_.mean()) ** 2).sum())
        rmse = np.sqrt(ssr / (n - A.shape[1]))
        out[t] = (b[0] * 252 * 100, 1 - ssr / sst, rmse, y_.std(ddof=1), b[0] / rmse, n)
    return pd.DataFrame(out, index=["alpha", "r2", "rmse", "sdr", "ir", "n"]).T


fits = {y: year_fit(y) for y in range(2011, 2027)}
rows = []
for y in range(2012, 2027):
    a, b = fits[y - 1], fits[y]
    j = a[a.n >= MIN_PRIOR].join(b, lsuffix="_p", rsuffix="")
    j = j.dropna(subset=["alpha"])
    j["year"], j["fund"] = y, j.index
    rows.append(j)
fy = pd.concat(rows, ignore_index=True)
n_before = len(fy)
lo, hi = fy["r2_p"].quantile(0.01), fy["r2_p"].quantile(0.99)                     # symmetric 1% trim of R2 (paper, Section 3)
fy = fy[(fy.r2_p >= lo) & (fy.r2_p <= hi)].reset_index(drop=True)
fy["tr2"] = np.log(np.sqrt(fy.r2_p) / (1 - np.sqrt(fy.r2_p)))
fy["style"] = meta.loc[fy.fund, "yahoo_category"].to_numpy()
fy.to_csv(ROOT / "data/ag_fund_years.csv", index=False)


# ---------- helpers ----------
def panel(dep, cols, df):
    X = design(df, cols)
    b, se, r2 = cluster_ols(df[dep].to_numpy(float), X, df["fund"].to_numpy())
    return {"coef": {c: float(b[1 + i]) for i, c in enumerate(cols)}, "t": {c: float(b[1 + i] / se[1 + i]) for i, c in enumerate(cols)},
            "n_funds": int(df.fund.nunique()), "n_fund_years": int(len(df)), "r2": float(r2)}


def welch(a, b):
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def sort_tables(df, dep, lag, independent):
    """5x5 tables: rows = quintile of `lag` (alpha_p or ir_p), columns = quintile of R2_p. Cell = mean of year-y `dep`."""
    df = df.copy()
    df["qr"] = 0
    df["qa"] = 0
    for y, g in df.groupby("year"):
        rr = g["r2_p"].rank(method="first")
        qr = R.bin_of(rr, len(g))
        df.loc[g.index, "qr"] = qr
        if independent:
            df.loc[g.index, "qa"] = R.bin_of(g[lag].rank(method="first"), len(g))
        else:
            for q in range(1, 6):
                mem = g.index[qr == q]
                df.loc[mem, "qa"] = R.bin_of(df.loc[mem, lag].rank(method="first"), len(mem))
    cell = {}
    for a in range(1, 6):
        for q in range(1, 6):
            s = df.loc[(df.qa == a) & (df.qr == q), dep]
            cell[(a, q)] = (float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), int(len(s)))
    margin_a = {a: (float(df.loc[df.qa == a, dep].mean()), float(df.loc[df.qa == a, dep].mean() / (df.loc[df.qa == a, dep].std(ddof=1) / np.sqrt((df.qa == a).sum())))) for a in range(1, 6)}
    margin_r = {q: (float(df.loc[df.qr == q, dep].mean()), float(df.loc[df.qr == q, dep].mean() / (df.loc[df.qr == q, dep].std(ddof=1) / np.sqrt((df.qr == q).sum())))) for q in range(1, 6)}
    allm = float(df[dep].mean()), float(df[dep].mean() / (df[dep].std(ddof=1) / np.sqrt(len(df))))
    lowhigh = {a: (float(df.loc[(df.qa == a) & (df.qr == 1), dep].mean() - df.loc[(df.qa == a) & (df.qr == 5), dep].mean()),
                   welch(df.loc[(df.qa == a) & (df.qr == 1), dep], df.loc[(df.qa == a) & (df.qr == 5), dep])) for a in range(1, 6)}
    lowhigh["All"] = (float(df.loc[df.qr == 1, dep].mean() - df.loc[df.qr == 5, dep].mean()), welch(df.loc[df.qr == 1, dep], df.loc[df.qr == 5, dep]))
    highlow = {q: (float(df.loc[(df.qa == 5) & (df.qr == q), dep].mean() - df.loc[(df.qa == 1) & (df.qr == q), dep].mean()),
                   welch(df.loc[(df.qa == 5) & (df.qr == q), dep], df.loc[(df.qa == 1) & (df.qr == q), dep])) for q in range(1, 6)}
    highlow["All"] = (float(df.loc[df.qa == 5, dep].mean() - df.loc[df.qa == 1, dep].mean()), welch(df.loc[df.qa == 5, dep], df.loc[df.qa == 1, dep]))
    # stricter inference: mean of the 15 annual averages of the Low-R2 / High-alpha cell and its t over years
    yearly = df[(df.qa == 5) & (df.qr == 1)].groupby("year")[dep].mean()
    ts = {"mean": float(yearly.mean()), "t": float(yearly.mean() / (yearly.std(ddof=1) / np.sqrt(len(yearly)))), "years": int(len(yearly))}
    return {"cell": {f"{a}_{q}": v for (a, q), v in cell.items()}, "margin_alpha": margin_a, "margin_r2": margin_r, "all": allm,
            "low_minus_high_r2": lowhigh, "high_minus_low_alpha": highlow, "ts_lowR2_highAlpha": ts, "df": df}


out = {"params": {"first_test_year": 2012, "last_test_year": 2026, "last_price_date": str(px.index[-1].date()), "last_factor_date": str(ff.index[-1].date()),
                  "funds_after_index_filter": len(tick), "index_named_excluded": int(is_index.sum()),
                  "pairs_before_trim": int(n_before), "pairs_after_trim": int(len(fy)), "r2_trim_low": float(lo), "r2_trim_high": float(hi),
                  "min_days_prior": MIN_PRIOR, "min_days_test": MIN_TEST}}
out["table1"] = {c: {"mean": float(fy[c].mean()), "median": float(fy[c].median()), "sd": float(fy[c].std()), "min": float(fy[c].min()), "max": float(fy[c].max())}
                 for c in ("r2_p", "alpha_p", "rmse_p", "sdr_p", "alpha", "ir")}
out["table1"]["n_funds"], out["table1"]["n_fund_years"] = int(fy.fund.nunique()), int(len(fy))
fy["alpha_lag"], fy["ir_lag"] = fy["alpha_p"], fy["ir_p"]
for tname, dep, lagc in (("table2", "alpha", "alpha_lag"), ("table3", "ir", "ir_lag")):
    t = {}
    for per, (a, b) in {"full": (2012, 2026), "early": (2012, 2018), "late": (2019, 2026)}.items():
        sub = fy[(fy.year >= a) & (fy.year <= b)]
        t[per] = {"tr2": panel(dep, ["tr2"], sub), "tr2_lag": panel(dep, ["tr2", lagc], sub),
                  "comp": panel(dep, ["rmse_p", "sdr_p"], sub), "comp_lag": panel(dep, ["rmse_p", "sdr_p", lagc], sub)}
    out[tname] = t
fm = {}
for dep in ("alpha", "ir"):
    for name, cols in (("tr2", ["tr2"]), ("tr2_comp", ["tr2", "rmse_p", "sdr_p"])):
        cl = [("alpha_lag" if dep == "alpha" else "ir_lag")]
        coefs = []
        for y, g in fy.groupby("year"):
            X = np.column_stack([np.ones(len(g)), g[cols + cl].to_numpy(float)])
            coefs.append(np.linalg.lstsq(X, g[dep].to_numpy(float), rcond=None)[0])
        C = np.array(coefs)
        mean, se = C.mean(0), C.std(0, ddof=1) / np.sqrt(len(C))
        names = ["const"] + cols + cl
        fm[f"{dep}_{name}"] = {n: {"mean": float(mean[i]), "t": float(mean[i] / se[i]), "pos_share": float((C[:, i] > 0).mean())} for i, n in enumerate(names)}
out["table5"] = fm
res_dep = sort_tables(fy, "alpha", "alpha_p", False)
res_ir = sort_tables(fy, "ir", "ir_p", False)
ind_dep = sort_tables(fy, "alpha", "alpha_p", True)
ind_ir = sort_tables(fy, "ir", "ir_p", True)
for k, v in (("table8_alpha", res_dep), ("table8_ir", res_ir), ("table9_alpha", ind_dep), ("table9_ir", ind_ir)):
    out[k] = {kk: vv for kk, vv in v.items() if kk != "df"}

# ---------- year by year: the Low-R2 / High-alpha portfolio (dependent sort) ----------
dfq = res_dep["df"]
yp = px.copy()
yp["SPY"] = spy
ye = yp.groupby(yp.index.year).tail(1)
ye.index = ye.index.year
yret = (ye / ye.shift(1) - 1) * 100                                         # calendar-year total return by fund and SPY
rows = []
for y in range(2012, 2027):
    g = dfq[dfq.year == y]
    port = g[(g.qr == 1) & (g.qa == 5)]
    short = g[(g.qr == 5) & (g.qa == 1)]
    allr = yret.loc[y, g.fund].astype(float)
    pr = yret.loc[y, port.fund].astype(float)
    sr = yret.loc[y, short.fund].astype(float)
    rows.append({"year": y, "n_funds": int(len(g)), "n_portfolio": int(len(port)),
                 "alpha_portfolio": float(port.alpha.mean()), "alpha_all": float(g.alpha.mean()),
                 "ret_portfolio": float(pr.mean()), "ret_all_funds": float(allr.mean()), "ret_spy": float(yret.loc[y, "SPY"]),
                 "ret_hiR2_lowalpha": float(sr.mean()), "ret_long_short": float(pr.mean() - sr.mean()),
                 "partial_year": bool(y == 2026)})
yy = pd.DataFrame(rows)
yy.to_csv(ROOT / "data/ag_yearly.csv", index=False)
out["yearly"] = rows
cum = pd.DataFrame({"low_r2_high_alpha": ((1 + yy.ret_portfolio / 100).cumprod() * 100).to_numpy(), "all_funds": ((1 + yy.ret_all_funds / 100).cumprod() * 100).to_numpy(),
                    "spy": ((1 + yy.ret_spy / 100).cumprod() * 100).to_numpy()}, index=yy.year.to_numpy())
out["annual_summary"] = {k: {"cagr_pct": float(((cum[k].iloc[-1] / 100) ** (1 / (len(cum) - 0.25)) - 1) * 100),
                             "years_beating_all_funds": int((yy.ret_portfolio > yy.ret_all_funds).sum()) if k == "low_r2_high_alpha" else None}
                         for k in cum.columns}
cum.to_csv(ROOT / "data/ag_cum_annual.csv")
json.dump(out, open(ROOT / "data/results_ag_python.json", "w"), indent=1, default=str)
c = out["table8_alpha"]["cell"]
print("pairs", n_before, "->", len(fy), " funds", fy.fund.nunique())
print("Table VIII panel A (mean alpha_y %, t):")
for a in range(5, 0, -1):
    print(a, " ".join(f"{c[f'{a}_{q}'][0]:7.2f}({c[f'{a}_{q}'][1]:5.1f})" for q in range(1, 6)), " All", f"{out['table8_alpha']['margin_alpha'][a][0]:.2f}")
print("Low-High R2 by alpha quintile:", {k: round(v[0], 2) for k, v in out["table8_alpha"]["low_minus_high_r2"].items()})
print("yearly:\n", yy[["year", "n_portfolio", "alpha_portfolio", "alpha_all", "ret_portfolio", "ret_all_funds", "ret_spy"]].round(2).to_string(index=False))
print("TR2 coef (alpha, full):", out["table2"]["full"]["tr2"]["coef"], out["table2"]["full"]["tr2"]["t"])
