"""4.3 Strategy: R-squared -- implementation of Kakushadze & Serur (2018), "151 Trading Strategies",
Section 4.3, Eq. (365)-(369).

    R_i(t) = alpha_i + b1_i MKT(t) + b2_i SMB(t) + b3_i HML(t) + b4_i MOM(t) + eps_i(t)        (365)
    R2_i   = 1 - SSres_i / SStot_i                                                              (366)

R_i(t) is the ETF's DAILY return in excess of the risk-free rate; MKT, SMB, HML and Carhart's MOM are Kenneth
French's daily factors. The regression uses the last year of daily returns (252 trading days).
At each month-end the ETFs are sorted into quintiles by R2 and, inside each R2 quintile, into sub-quintiles by
alpha (25 groups). The strategy buys the group with the lowest R2 quintile and the highest alpha sub-quintile
and sells the group with the highest R2 quintile and the lowest alpha sub-quintile, equal weights, held for one
month. Variants (the paper: "other variations are possible"): long leg only; alpha-only quintiles (the 4.2 rule
on the same universe); R2-only quintiles.
"""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf   # pip install yfinance

# 36 US-listed equity ETFs, all with Yahoo data from 2006-06-22 or earlier; list fixed before any result was seen.
UNIVERSE = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",                 # 9 SPDR sectors
            "IYR", "IYZ", "IYT", "IGV", "SOXX", "IBB", "OIH", "SMH", "KBE", "KRE", "KIE",   # industries
            "XBI", "XHB", "XME", "XOP", "XRT", "ITB", "GDX",
            "IWM", "IWD", "IWF", "IWN", "IWO", "IWS", "IWP", "IJH", "IJR", "IJK", "IJJ", "IJS", "IJT",   # size and style
            "MDY", "QQQ", "DIA"]
FACTORS = ["MKT", "SMB", "HML", "MOM"]
WINDOW = 252        # regression window in trading days (paper: typically 1 year)
NQ = 5              # quintiles, and sub-quintiles (paper: 5 x 5 = 25 groups)
COST_BPS = 10       # cost per unit traded, in basis points
START, END = "2005-06-01", "2026-10-01"   # fixed end date so the sample is reproducible
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF3_ZIP, MOM_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"


pd.set_option("compute.use_numexpr", False)   # numexpr corrupts/segfaults on 3000-column frames in this environment


def pct(df):
    """Simple returns. Not DataFrame.pct_change: on frames with ~500+ columns that call returned wrong values (and
    segfaulted) in this pandas 2.3.3 / numpy 2.5.3 environment."""
    return df / df.shift(1) - 1


def spy_prices(start):
    """SPY adjusted close, cached in data/funds/spy.csv so reruns do not depend on a live Yahoo call (R18)."""
    f = Path(__file__).resolve().parents[1] / "data/funds/spy.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].loc[start:]
    s = yf.download("SPY", start="2009-01-01", end=END, auto_adjust=True, progress=False)["Close"].squeeze()
    assert len(s) > 1000, "SPY download failed"
    s.rename("SPY").to_csv(f)
    return s.loc[start:]


def download_prices():
    """Daily adjusted closes for the ETFs and SPY from Yahoo Finance."""
    tickers = UNIVERSE + ["SPY"]
    px = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    return px[tickers].dropna(how="all")


def read_french_zip(name):
    """Bytes of one Ken French zip: from the folder in $QS_RAW_DIR if set (archived copy), else the library."""
    raw = os.environ.get("QS_RAW_DIR")
    for n in (name, name.replace("_CSV.zip", "_TXT.zip")):      # the archived copy may be the CSV or the TXT download
        if raw and (Path(raw) / n).exists():
            return (Path(raw) / n).read_bytes()
    req = urllib.request.Request(FF_BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req).read()


def parse_french_daily(blob, columns):
    """Parse a daily CSV from Ken French's library (values in percent) into a decimal DataFrame."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    rows = []
    for l in lines:                                  # data rows only: 8-digit date then the values (comma or space separated)
        p = l.replace(",", " ").split()
        if len(p) == len(columns) + 1 and len(p[0]) == 8 and p[0].isdigit():
            rows.append(p)
        elif rows:
            break
    df = pd.DataFrame(rows, columns=["date"] + columns).astype(float)
    df.index = pd.to_datetime(df.pop("date").astype(int).astype(str), format="%Y%m%d")
    return df / 100                                  # percent -> decimal


def download_factors():
    """Daily MKT, SMB, HML, RF and Carhart's MOM, aligned on the days both files cover."""
    ff3 = parse_french_daily(read_french_zip(FF3_ZIP), ["MKT", "SMB", "HML", "RF"])
    mom = parse_french_daily(read_french_zip(MOM_ZIP), ["MOM"])
    return pd.concat([ff3, mom], axis=1, join="inner")[["MKT", "SMB", "HML", "MOM", "RF"]]


def ols(y, X):
    """OLS of excess returns y (n x k, one column per ETF) on [1, factors] (n x 4).
    Returns coefficients (5 x k: alpha, 4 betas), t-statistics and R2 = 1 - SSres/SStot (Eq. 366-369)."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.solve(A.T @ A, A.T @ y)
    e = y - A @ b
    ss_res = (e ** 2).sum(axis=0)                           # Eq. (367)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)        # Eq. (368)-(369)
    dof = len(A) - A.shape[1]
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(A.T @ A)), ss_res / dof))
    return b, b / se, 1 - ss_res / ss_tot


def rolling_fit(px, ff, window=WINDOW, lag=0):
    """Eq. (365)-(366) at each month-end: alpha, its t-statistic and R2 of every ETF over the last `window`
    trading days, dated by calendar month-end. lag > 0 = factor-availability test: the window ends `lag`
    trading days before month-end, as if the factors were published with that delay."""
    d = pd.concat([pct(px[UNIVERSE]), ff], axis=1, join="inner").dropna()
    y = d[UNIVERSE].sub(d["RF"], axis=0)                    # excess returns (paper's footnote 50)
    X = d[FACTORS]
    alpha, tstat, r2 = {}, {}, {}
    for t in d.groupby(d.index.to_period("M")).tail(1).index:
        i = d.index.get_loc(t) - lag
        if i + 1 < window:
            continue
        s = slice(i + 1 - window, i + 1)
        b, tt, rr = ols(y.iloc[s].to_numpy(), X.iloc[s].to_numpy())
        m = t + pd.offsets.MonthEnd(0)
        alpha[m], tstat[m], r2[m] = b[0], tt[0], rr
    f = lambda dct: pd.DataFrame(dct, index=UNIVERSE).T
    return f(alpha), f(tstat), f(r2)


def bin_of(rank, n):
    """Bin 1..NQ of a rank 1..n (1 = lowest): ceil(rank * NQ / n). Same rule in the R script."""
    return np.ceil(rank * NQ / n).astype(int)


def legs(alpha, r2, variant):
    """Long and short membership (boolean DataFrames, month-end signals) for one sorting variant."""
    lg, sh = pd.DataFrame(False, index=alpha.index, columns=alpha.columns), pd.DataFrame(False, index=alpha.index, columns=alpha.columns)
    for t in alpha.index:
        a, r = alpha.loc[t], r2.loc[t]
        ok = a.notna() & r.notna()
        if ok.sum() < NQ:
            continue
        a, r = a[ok], r[ok]
        ra, rr = a.rank(method="first"), r.rank(method="first")      # rank 1 = lowest
        qa, qr = bin_of(ra, len(a)), bin_of(rr, len(r))
        if variant in ("r2_alpha", "long_leg"):
            sub = pd.Series(0, index=a.index)
            for q in (1, NQ):                                       # alpha sub-quintiles inside the lowest / highest R2 quintile
                members = a.index[qr == q]
                inner = a[members].rank(method="first")
                sub[members] = bin_of(inner, len(members))
            lg.loc[t, a.index] = ((qr == 1) & (sub == NQ)).to_numpy()   # lowest R2, highest alpha
            if variant == "r2_alpha":
                sh.loc[t, a.index] = ((qr == NQ) & (sub == 1)).to_numpy()   # highest R2, lowest alpha
        elif variant == "alpha_only":
            lg.loc[t, a.index], sh.loc[t, a.index] = (qa == NQ).to_numpy(), (qa == 1).to_numpy()
        elif variant == "r2_only":
            lg.loc[t, a.index], sh.loc[t, a.index] = (qr == 1).to_numpy(), (qr == NQ).to_numpy()
        else:
            raise ValueError(variant)
    return lg, sh


def run_legs(px_m, lg, sh, cost_bps=COST_BPS, start=None):
    """Same engine as 4.1/4.2, for a long leg and an optional short leg. Each leg is equal-weighted and held for one
    month; each leg is rebalanced to 100% of its own notional every month, costs are charged on both legs on
    drift-adjusted turnover. Return = long-leg return - short-leg return (no borrow fee or interest on proceeds)."""
    r = pct(px_m[UNIVERSE])
    if start is not None:
        keep = pd.Series(lg.index >= start, index=lg.index)
        lg, sh = lg.where(keep, False, axis=0), sh.where(keep, False, axis=0)
    def leg_w(mask):
        w = mask.astype(float)
        n = w.sum(axis=1)
        return w.div(n.where(n > 0), axis=0)                        # NaN rows where the leg is empty
    wl, ws = leg_w(lg).shift(1), leg_w(sh).shift(1)                  # trade at month-end, earn NEXT month
    def leg_ret_turn(w):
        gross = (w * r).sum(axis=1, min_count=1)
        drifted = w * (1 + r)
        drifted = drifted.div(drifted.sum(axis=1), axis=0)
        turn = (w - drifted.shift(1)).abs().sum(axis=1, min_count=1)
        first = w.dropna(how="all").index[0] if w.notna().any().any() else None
        if first is not None:
            turn.loc[first] = 1.0                                   # initial purchase
        return gross, turn, first
    gl, tl, fl = leg_ret_turn(wl)
    if sh.to_numpy().any():
        gs, ts, _ = leg_ret_turn(ws)
        gross, turn = gl - gs, tl + ts
    else:
        gross, turn = gl, tl
    net = gross - turn * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "net": net, "turnover": turn}).loc[fl:]
    return out, wl.loc[fl:], ws.loc[fl:]


def to_monthly(px):
    return px.resample("ME").last()


def next_close_prices(px):
    """Month-end prices replaced by the close of the NEXT trading day: execution one day after the signal."""
    last = px.groupby(px.index.to_period("M")).tail(1).index
    nxt = px.shift(-1).loc[last]
    nxt.index = nxt.index + pd.offsets.MonthEnd(0)
    return nxt


def stats(ret, label=""):
    ret = ret.dropna()
    yrs = len(ret) / 12
    eq = np.r_[1.0, (1 + ret).cumprod().to_numpy()]      # starts at 1 so an initial loss counts
    dd = eq / np.maximum.accumulate(eq) - 1
    return {
        "label": label,
        "start": str(ret.index[0].date()), "end": str(ret.index[-1].date()),
        "months": int(len(ret)),
        "total_return_pct": round(float(eq[-1] - 1) * 100, 2),
        "cagr_pct": round(float(eq[-1] ** (1 / yrs) - 1) * 100, 2),
        "ann_vol_pct": round(float(ret.std(ddof=1) * np.sqrt(12)) * 100, 2),
        "sharpe": round(float(ret.mean() / ret.std(ddof=1) * np.sqrt(12)), 3),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "hit_rate_pct": round(float((ret > 0).mean()) * 100, 1),
    }


def regression_table(px, ff, start=None, end=None):
    """Eq. (365) for each ETF over one period: annualised alpha (%), t-statistic, betas and R-squared."""
    d = pd.concat([pct(px[UNIVERSE]), ff], axis=1, join="inner").dropna().loc[start:end]
    b, t, r2 = ols(d[UNIVERSE].sub(d["RF"], axis=0).to_numpy(), d[FACTORS].to_numpy())
    return pd.DataFrame({"alpha_pct_yr": b[0] * 252 * 100, "alpha_t": t[0], "beta_mkt": b[1], "beta_smb": b[2],
                         "beta_hml": b[3], "beta_mom": b[4], "r2": r2, "days": len(d)}, index=UNIVERSE)


def main():
    px = download_prices()
    ff = download_factors()
    last = min(px.index[-1], ff.index[-1])          # both must cover the same last day
    px, ff = px.loc[:last], ff.loc[:last]
    px_m = to_monthly(px)

    alpha, tstat, r2 = rolling_fit(px, ff)          # Eq. (365)-(366), one-year daily window
    alpha, tstat, r2 = (x.reindex(px_m.index) for x in (alpha, tstat, r2))
    main_l, main_s = legs(alpha, r2, "r2_alpha")
    res = {"r2_alpha": run_legs(px_m, main_l, main_s)}
    for v in ("long_leg", "alpha_only", "r2_only"):
        lg, sh = legs(alpha, r2, v)
        res[v] = run_legs(px_m, lg, sh)
    idx = res["r2_alpha"][0].index                  # one common sample for everything
    ew_l = pd.DataFrame(True, index=alpha.index, columns=alpha.columns).where(alpha.notna(), False)
    ew = run_legs(px_m, ew_l, ew_l & False)[0].reindex(idx)      # equal-weight universe, same engine and costs
    spy = pct(px_m["SPY"]).loc[idx]

    names = {"r2_alpha": "R2 x alpha long-short (paper's example)", "long_leg": "Long leg only",
             "alpha_only": "Alpha-only long-short", "r2_only": "R2-only long-short"}
    summary = {"params": {"window_days": WINDOW, "quantiles": NQ, "cost_bps": COST_BPS, "universe": UNIVERSE,
                          "factors": "Fama-French 3 + Carhart momentum (daily, Ken French library)",
                          "last_factor_date": str(ff.index[-1].date())}}
    for v, (o, _, _) in res.items():
        summary[v] = stats(o["net"].reindex(idx), names[v] + " (net)")
        summary[v + "_gross"] = stats(o["gross"].reindex(idx), names[v] + " (gross)")
    summary["equal_weight"] = stats(ew["net"], f"Equal-weight {len(UNIVERSE)} (net)")
    summary["spy"] = stats(spy, "SPY buy & hold")
    summary["avg_monthly_turnover_pct"] = {v: round(float(o["turnover"].reindex(idx).mean()) * 100, 1) for v, (o, _, _) in res.items()}
    summary["avg_monthly_turnover_pct"]["equal_weight"] = round(float(ew["turnover"].mean()) * 100, 1)
    summary["leg_sizes"] = {"avg_long": round(float((main_l.loc[idx]).sum(axis=1).mean()), 2),
                            "avg_short": round(float((main_s.loc[idx]).sum(axis=1).mean()), 2)}
    # how much do R2 and alpha overlap? (cross-sectional rank correlation, averaged over months)
    summary["signal_diagnostics"] = {
        "avg_rank_corr_alpha_r2": round(float(np.nanmean([alpha.loc[t].rank().corr(r2.loc[t].rank()) for t in idx])), 3),
        "mean_r2": round(float(r2.reindex(idx).to_numpy().mean()), 3),
        "pct_alpha_abs_t_above_2": round(float((tstat.reindex(idx).abs() > 2).to_numpy().mean() * 100), 1)}
    print(pd.DataFrame({k: summary[k] for k in ("r2_alpha", "long_leg", "alpha_only", "r2_only", "equal_weight", "spy")}).T[
        ["cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct", "months"]])
    return summary


if __name__ == "__main__":
    main()
