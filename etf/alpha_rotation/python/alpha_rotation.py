"""4.2 Strategy: Alpha rotation -- implementation of Kakushadze & Serur (2018),
"151 Trading Strategies", Section 4.2, Eq. (364).

    R_i(t) = alpha_i + b1_i * MKT(t) + b2_i * SMB(t) + b3_i * HML(t) + eps_i(t)

R_i(t) is the ETF's DAILY return in excess of the risk-free rate, and MKT, SMB, HML are the three
Fama-French factors (Kenneth French's data library). The regression uses the last year of daily
returns (252 trading days; the paper's footnote 77: "typically 1 year, daily or weekly returns").
At each month-end: rank the ETFs by alpha_i, buy the top N equal-weighted, hold for one month.
This is the sector momentum rotation of Section 4.1 with the cumulative return replaced by alpha.
"""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf   # pip install yfinance

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
WINDOW = 252        # regression window in trading days (paper: 1 year)
TOP_N = 3           # diversified version; N = 1 is the closest to the paper's top decile
COST_BPS = 10       # cost per unit of turnover, in basis points
T_MOM = 12          # look-back of the Section 4.1 momentum signal used as the comparison
START, END = "1998-12-01", "2026-10-01"   # fixed end date so the sample is reproducible
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def download_prices():
    """Daily adjusted closes for the 9 sector ETFs + SPY, straight from Yahoo Finance."""
    tickers = SECTORS + ["SPY"]
    px = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    return px[tickers].dropna(how="all")


def download_factors():
    """Daily Fama-French 3 factors and the risk-free rate (all in percent) from Ken French's library."""
    req = urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req).read())) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[head + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) != 5 or not p[0].isdigit():
            break
        rows.append(p)
    ff = pd.DataFrame(rows, columns=["date", "MKT", "SMB", "HML", "RF"]).astype(float)
    ff.index = pd.to_datetime(ff.pop("date").astype(int).astype(str), format="%Y%m%d")
    return ff / 100                              # percent -> decimal


def ff3(y, X):
    """OLS of excess returns y (n x k, one column per ETF) on [1, MKT, SMB, HML] (n x 4).
    Returns coefficients (4 x k: alpha, b_MKT, b_SMB, b_HML), their t-statistics and R-squared."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.solve(A.T @ A, A.T @ y)
    e = y - A @ b
    dof = len(A) - A.shape[1]
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(A.T @ A)), (e ** 2).sum(axis=0) / dof))
    r2 = 1 - (e ** 2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return b, b / se, r2


def rolling_alpha(px, ff, window=WINDOW, lag=0):
    """Eq. (364) at each month-end: alpha (and its t-statistic) of every sector ETF over the last
    `window` trading days. Dated by calendar month-end, like the monthly prices.
    lag > 0 is the factor-availability test: the window ends `lag` trading days before month-end,
    as if the factor data were published with that delay (lag = 0 is the main, idealised strategy)."""
    d = pd.concat([px[SECTORS].pct_change(fill_method=None), ff], axis=1, join="inner").dropna()
    y = d[SECTORS].sub(d["RF"], axis=0)          # excess returns (paper's footnote 50)
    X = d[["MKT", "SMB", "HML"]]
    last_day = d.groupby(d.index.to_period("M")).tail(1).index
    alpha, tstat = {}, {}
    for t in last_day:
        i = d.index.get_loc(t) - lag
        if i + 1 < window:
            continue
        s = slice(i + 1 - window, i + 1)
        b, tt, _ = ff3(y.iloc[s].to_numpy(), X.iloc[s].to_numpy())
        m = t + pd.offsets.MonthEnd(0)
        alpha[m], tstat[m] = b[0], tt[0]
    return (pd.DataFrame(alpha, index=SECTORS).T, pd.DataFrame(tstat, index=SECTORS).T)


def to_monthly(px):
    return px.resample("ME").last()             # last trading day of each month


def run(px_m, signal, n=TOP_N, cost_bps=COST_BPS):
    """Same engine as the 4.1 article: `signal` is a month-end score per ETF (higher = better)."""
    sect = px_m[SECTORS]
    r = sect.pct_change(fill_method=None)                       # month-m return of each ETF
    rank = signal.rank(axis=1, ascending=False, method="first")
    w_signal = (rank <= n).astype(float).where(signal.notna())
    w_signal = w_signal.div(w_signal.sum(axis=1), axis=0)   # equal weight
    w = w_signal.shift(1)                       # trade at month-end, earn NEXT month
    gross = (w * r).sum(axis=1, min_count=1)
    drifted = w * (1 + r)                       # last month's weights after the month's price moves
    drifted = drifted.div(drifted.sum(axis=1), axis=0)
    turnover = (w - drifted.shift(1)).abs().sum(axis=1)   # trades needed to get back to target
    first = w.dropna(how="all").index[0]
    turnover.loc[first] = 1.0                   # initial purchase
    net = gross - turnover * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "net": net, "turnover": turnover}).loc[first:]
    return out, w.loc[first:]


def from_month(signal, start):
    """Fresh-capital start: ignore the signal before `start`, so the first purchase (and its cost) is at `start`."""
    s = signal.copy()
    s.loc[s.index < start] = np.nan
    return s


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
    """Eq. (364) for each ETF over one period: annualised alpha (%), betas, t-statistics, R-squared."""
    d = pd.concat([px[SECTORS].pct_change(fill_method=None), ff], axis=1, join="inner").dropna().loc[start:end]
    b, t, r2 = ff3(d[SECTORS].sub(d["RF"], axis=0).to_numpy(), d[["MKT", "SMB", "HML"]].to_numpy())
    return pd.DataFrame({"alpha_pct_yr": b[0] * 252 * 100, "alpha_t": t[0], "beta_mkt": b[1], "beta_smb": b[2],
                         "beta_hml": b[3], "t_mkt": t[1], "t_smb": t[2], "t_hml": t[3], "r2": r2,
                         "days": len(d)}, index=SECTORS)


def results_table(summary):
    """Net-of-cost results side by side, as a DataFrame."""
    cols = {"Alpha top 3": "alpha3", "Alpha top 1": "alpha1", "Momentum top 3 (4.1)": "mom3",
            "Equal-weight 9 (net)": "equal_weight", "SPY": "spy"}
    return pd.DataFrame({name: {
        "$100 became": 100 * (1 + summary[k]["total_return_pct"] / 100),
        "CAGR (%)": summary[k]["cagr_pct"], "Volatility (%)": summary[k]["ann_vol_pct"],
        "Sharpe": summary[k]["sharpe"], "Max drawdown (%)": summary[k]["max_drawdown_pct"],
        "Months with a gain (%)": summary[k]["hit_rate_pct"],
        "Monthly turnover (%)": summary["avg_monthly_turnover_pct"].get(k, 0.0)}
        for name, k in cols.items()})


def plot_results(series, grid, windows, ns):
    eq = 100 * (1 + pd.DataFrame(series)).cumprod()
    eq = pd.concat([pd.DataFrame(100.0, index=[eq.index[0] - pd.offsets.MonthEnd(1)], columns=eq.columns), eq])  # start at $100
    colors, styles = ["#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "grey"], ["-", "-", "-", "-", "--"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 12))
    for c, col, ls in zip(eq.columns, colors, styles):
        ax[0].plot(eq.index, eq[c], color=col, ls=ls, lw=2, label=c)
        ax[1].plot(eq.index, 100 * (eq[c] / eq[c].cummax() - 1), color=col, ls=ls, lw=2)
    ax[0].set_yscale("log"); ax[0].set_title("Growth of $100 (log scale)"); ax[0].legend(frameon=False)
    ax[1].set_title("Drawdown from previous peak (%)")
    sharpe = np.array([[grid[f"W{w}_N{n}"]["sharpe"] for n in ns] for w in windows])
    ax[2].imshow(sharpe, cmap="Blues")
    ax[2].set_xticks(range(len(ns)), ns); ax[2].set_yticks(range(len(windows)), windows)
    ax[2].set_xlabel("ETFs held (n)"); ax[2].set_ylabel("Regression window (trading days)")
    ax[2].set_title("Sharpe ratio by setting (same sample for every cell)")
    for i in range(len(windows)):
        for j in range(len(ns)):
            ax[2].text(j, i, f"{sharpe[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    plt.show()


def main():
    px = download_prices()
    ff = download_factors()
    last = min(px.index[-1], ff.index[-1])      # both series must cover the same last day
    px, ff = px.loc[:last], ff.loc[:last]
    px_m = to_monthly(px)

    alpha, tstat = rolling_alpha(px, ff)        # Eq. (364), one-year daily window
    alpha = alpha.reindex(px_m.index)
    a3, w = run(px_m, alpha)                    # alpha rotation, top 3 of 9
    a1 = run(px_m, alpha, n=1)[0]               # top 1 of 9: closest to the paper's top decile
    mom = px_m[SECTORS] / px_m[SECTORS].shift(T_MOM) - 1     # Section 4.1 signal, Eq. (361)
    idx = a3.index                              # one common sample for everything
    m3 = run(px_m, mom)[0].reindex(idx)         # 4.1 momentum top 3 on the same months
    ew = run(px_m, mom, n=9)[0].reindex(idx)    # all 9 equally, rebalanced monthly, same costs
    spy = px_m["SPY"].pct_change(fill_method=None).loc[idx]

    summary = {
        "params": {"window_days": WINDOW, "top_n": TOP_N, "cost_bps": COST_BPS, "universe": SECTORS,
                   "factors": "Fama-French 3 (daily, Ken French library)", "last_factor_date": str(ff.index[-1].date())},
        "alpha3": stats(a3["net"], "Alpha top 3 of 9 (net)"),
        "alpha3_gross": stats(a3["gross"], "Alpha top 3 of 9 (gross)"),
        "alpha1": stats(a1["net"].reindex(idx), "Alpha top 1 of 9 (net)"),
        "mom3": stats(m3["net"], "Momentum top 3 of 9 (net)"),
        "equal_weight": stats(ew["net"], "Equal-weight 9 (net)"),
        "spy": stats(spy, "SPY buy & hold"),
        "avg_monthly_turnover_pct": {"alpha3": round(float(a3["turnover"].mean()) * 100, 1),
                                     "alpha1": round(float(a1["turnover"].reindex(idx).mean()) * 100, 1),
                                     "mom3": round(float(m3["turnover"].mean()) * 100, 1),
                                     "equal_weight": round(float(ew["turnover"].mean()) * 100, 1)},
    }
    # how alike are the two signals? average overlap of the top-3 sets, and how noisy is alpha?
    top_a = w.loc[idx] > 0
    top_m = (run(px_m, mom)[1].reindex(idx)) > 0
    summary["signal_diagnostics"] = {
        "avg_top3_overlap_with_momentum": round(float((top_a & top_m).sum(axis=1).mean()), 2),
        "pct_alpha_abs_t_above_2": round(float((tstat.reindex(idx).abs() > 2).to_numpy().mean() * 100), 1),
        "median_abs_alpha_t": round(float(tstat.reindex(idx).abs().stack().median()), 2),
        "avg_top3_alpha_pct_yr": round(float((alpha.reindex(idx).where(top_a).stack().mean()) * 252 * 100), 2),
    }
    # exploratory grid: every cell is evaluated on the same months as the main result
    # Every cell, and the equal-weight benchmark, starts with fresh capital in the same month: the first month in
    # which the longest window (504 days) has an alpha. So every cell covers exactly the same months.
    windows, ns, grid = (126, 252, 504), (1, 2, 3, 4), {}
    alphas = {wd: rolling_alpha(px, ff, wd)[0].reindex(px_m.index) for wd in windows}
    g0 = max(a.dropna(how="any").index[0] for a in alphas.values())
    gidx = idx[idx > g0]
    for wd in windows:
        for n in ns:
            s = stats(run(px_m, from_month(alphas[wd], g0), n)[0]["net"].reindex(gidx))
            grid[f"W{wd}_N{n}"] = {"cagr_pct": s["cagr_pct"], "sharpe": s["sharpe"], "months": s["months"]}
    summary["sensitivity"] = grid
    summary["sensitivity_sample"] = {
        "start": str(gidx[0].date()), "end": str(gidx[-1].date()), "months": len(gidx),
        "equal_weight": stats(run(px_m, from_month(mom, g0), n=9)[0]["net"].reindex(gidx), "Equal-weight 9 (net)")}
    # Factor-availability test: alpha from a window that ends `lag` trading days before month-end, and the
    # trade at the same close or at the next trading day's close. Same months and fresh capital for all.
    px_next, variants = next_close_prices(px), {}
    for lag in (0, 5, 21):
        al = rolling_alpha(px, ff, WINDOW, lag)[0].reindex(px_m.index)
        for ex, prices in (("same", px_m), ("next", px_next)):
            variants[(lag, ex)] = (al, prices)
    t0 = max(a.dropna(how="any").index[0] for a, _ in variants.values())
    nets, tops = {}, {}
    for (lag, ex), (al, prices) in variants.items():
        o, wv = run(prices, from_month(al, t0))
        nets[(lag, ex)], tops[(lag, ex)] = o["net"], wv > 0
    tidx = idx[idx > t0]
    for v in nets.values():
        tidx = tidx.intersection(v.dropna().index)
    base = tops[(0, "same")].reindex(tidx)
    summary["timing_test"] = {"start": str(tidx[0].date()), "end": str(tidx[-1].date()), "months": len(tidx), "variants": {
        f"lag{lag}_{ex}": {**stats(nets[(lag, ex)].reindex(tidx)),
                           "top3_overlap_with_main": round(float((tops[(lag, ex)].reindex(tidx) & base).sum(axis=1).mean()), 2)}
        for (lag, ex) in nets},
        "equal_weight": stats(run(px_m, from_month(mom, t0), n=9)[0]["net"].reindex(tidx), "Equal-weight 9 (net)"),
        "mom3": stats(run(px_m, from_month(mom, t0))[0]["net"].reindex(tidx), "Momentum top 3 (net)"),
        # the same two benchmarks traded at the next trading day's close, to compare like with like with the "next" variants
        "equal_weight_next": stats(run(px_next, from_month(mom, t0), n=9)[0]["net"].reindex(tidx), "Equal-weight 9, next-day close (net)"),
        "mom3_next": stats(run(px_next, from_month(mom, t0))[0]["net"].reindex(tidx), "Momentum top 3, next-day close (net)")}
    # calendar-year returns
    yr = pd.DataFrame({"alpha3": a3["net"], "alpha1": a1["net"].reindex(idx), "mom3": m3["net"],
                       "equal_weight": ew["net"], "spy": spy})
    summary["annual_returns_pct"] = (
        ((1 + yr).groupby(yr.index.year).prod() - 1) * 100).round(2).to_dict("index")
    # the regressions themselves: whole sample, and the latest one-year window (what the rule sees today)
    full = regression_table(px, ff)
    latest = regression_table(px, ff, start=px.index[-WINDOW:][0])
    summary["regression_full"] = full.round(4).to_dict("index")
    summary["regression_latest"] = latest.round(4).to_dict("index")

    # ---- SAVE-BEGIN: optional, uncomment to write the results to ./data ----
    # import json
    # from pathlib import Path
    # out = Path("data"); out.mkdir(exist_ok=True)
    # px.to_csv(out / "prices_python.csv"); ff.to_csv(out / "factors_python.csv")
    # (out / "results_python.json").write_text(json.dumps(summary, indent=2, default=str))
    # pd.DataFrame({"alpha3_net": a3["net"], "alpha3_gross": a3["gross"], "alpha1_net": a1["net"].reindex(idx),
    #               "mom3_net": m3["net"], "ew_net": ew["net"], "spy": spy, "turnover_alpha3": a3["turnover"]}
    #              ).to_csv(out / "monthly_returns_python.csv", float_format="%.10f")
    # w.to_csv(out / "weights_python.csv", float_format="%.6f")
    # alpha.to_csv(out / "alphas_python.csv", float_format="%.10f")
    # tstat.to_csv(out / "alpha_tstats_python.csv", float_format="%.6f")
    # full.to_csv(out / "regression_full_python.csv", float_format="%.6f")
    # latest.to_csv(out / "regression_latest_python.csv", float_format="%.6f")
    # ---- SAVE-END ----

    print("Eq. (364) over the whole sample, daily excess returns (alpha in % per year):")
    print(full.round(2).to_string())
    print("\nEq. (364) over the latest one-year window:")
    print(latest.round(2).to_string())
    print("\n" + results_table(summary).round(2).to_string())
    tt = summary["timing_test"]
    print(f"\nFactor-availability test, {tt['start']} to {tt['end']} ({tt['months']} months):")
    print(pd.DataFrame({k: {"CAGR (%)": v["cagr_pct"], "Sharpe": v["sharpe"], "Max drawdown (%)": v["max_drawdown_pct"],
                            "Top-3 overlap with main": v["top3_overlap_with_main"]} for k, v in tt["variants"].items()}).to_string())
    plot_results({"Alpha top 3": a3["net"], "Alpha top 1": a1["net"].reindex(idx), "Momentum top 3 (4.1)": m3["net"],
                  "Equal-weight 9 (net)": ew["net"], "SPY": spy}, grid, windows, ns)
    return a3, w, alpha, px_m, ew, spy, summary


if __name__ == "__main__":
    main()
