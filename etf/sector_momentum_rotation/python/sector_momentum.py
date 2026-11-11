"""4.1 Strategy: Sector momentum rotation -- implementation of Kakushadze & Serur (2018),
"151 Trading Strategies", Section 4.1, Eq. (361).

    R_cum_i(t) = P_i(t) / P_i(t - T) - 1      (T months back from t)

At each month-end t: rank the sector ETFs by R_cum_i(t), buy the top N equal-weighted,
hold for one month (the paper's 1-3 month holding period), then repeat.
N = 1 is the closest to the paper's "top decile" with nine ETFs; N = 3 is a diversified version.
N = 9 holds every ETF equally, which is the no-rotation benchmark.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf   # pip install yfinance

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
T_MONTHS = 12      # formation period (paper: 6-12 months)
TOP_N = 3          # diversified version; N = 1 is the closest to the paper's top decile
COST_BPS = 10      # cost per unit of turnover, in basis points
START, END = "1998-12-01", "2026-11-01"   # fixed end date so the sample is reproducible


def download_prices():
    """Daily adjusted closes for the 9 sector ETFs + SPY, straight from Yahoo Finance."""
    tickers = SECTORS + ["SPY"]
    px = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    return px[tickers].dropna(how="all")


def to_monthly(px):
    return px.resample("ME").last()             # last trading day of each month


def run(px_m, T=T_MONTHS, n=TOP_N, cost_bps=COST_BPS):
    sect = px_m[SECTORS]
    r = sect.pct_change()                       # month-m return of each ETF
    rcum = sect / sect.shift(T) - 1             # Eq. (361), known at end of month m
    rank = rcum.rank(axis=1, ascending=False, method="first")
    w_signal = (rank <= n).astype(float).where(rcum.notna())
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
    return out, w.loc[first:], rcum


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


def results_table(summary):
    """Net-of-cost results side by side, as a DataFrame."""
    cols = {"Top 3 of 9": "top3", "Top 1 of 9": "top1", "Equal-weight 9 (net)": "equal_weight", "SPY": "spy"}
    return pd.DataFrame({name: {
        "$100 became": 100 * (1 + summary[k]["total_return_pct"] / 100),
        "CAGR (%)": summary[k]["cagr_pct"], "Volatility (%)": summary[k]["ann_vol_pct"],
        "Sharpe": summary[k]["sharpe"], "Max drawdown (%)": summary[k]["max_drawdown_pct"],
        "Months with a gain (%)": summary[k]["hit_rate_pct"],
        "Monthly turnover (%)": summary["avg_monthly_turnover_pct"].get(k, 0.0)}
        for name, k in cols.items()})


def plot_results(top3, top1, ew, spy, grid):
    eq = 100 * (1 + pd.DataFrame({"Top 3 of 9": top3, "Top 1 of 9": top1,
                                  "Equal-weight 9 (net)": ew, "SPY": spy})).cumprod()
    colors, styles = ["#2a78d6", "#7b4fc4", "#eb6834", "grey"], ["-", "-", "-", "--"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 12))
    for c, col, ls in zip(eq.columns, colors, styles):
        ax[0].plot(eq.index, eq[c], color=col, ls=ls, lw=2, label=c)
        ax[1].plot(eq.index, 100 * (eq[c] / eq[c].cummax() - 1), color=col, ls=ls, lw=2)
    ax[0].set_yscale("log"); ax[0].set_title("Growth of $100 (log scale)"); ax[0].legend(frameon=False)
    ax[1].set_title("Drawdown from previous peak (%)")
    Ts, Ns = (3, 6, 9, 12), (1, 2, 3, 4)
    sharpe = np.array([[grid[f"T{T}_N{n}"]["sharpe"] for n in Ns] for T in Ts])
    ax[2].imshow(sharpe, cmap="Blues")
    ax[2].set_xticks(range(4), Ns); ax[2].set_yticks(range(4), Ts)
    ax[2].set_xlabel("ETFs held (n)"); ax[2].set_ylabel("Look-back T (months)")
    ax[2].set_title("Sharpe ratio by setting (same sample for every cell)")
    for i in range(4):
        for j in range(4):
            ax[2].text(j, i, f"{sharpe[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    plt.show()


def main():
    px = download_prices()
    px_m = to_monthly(px)
    top3, w, rcum = run(px_m)                   # top 3 of 9
    top1 = run(px_m, n=1)[0]                    # top 1 of 9: closest to the paper's top decile
    ew = run(px_m, n=9)[0]                      # all 9 equally, rebalanced monthly, same costs
    idx = top3.index                            # one common sample for everything
    spy = px_m["SPY"].pct_change().loc[idx]

    summary = {
        "params": {"T_months": T_MONTHS, "top_n": TOP_N, "cost_bps": COST_BPS, "universe": SECTORS},
        "top3": stats(top3["net"], "Top 3 of 9 (net)"),
        "top3_gross": stats(top3["gross"], "Top 3 of 9 (gross)"),
        "top1": stats(top1["net"], "Top 1 of 9 (net)"),
        "equal_weight": stats(ew["net"], "Equal-weight 9 (net)"),
        "equal_weight_gross": stats(ew["gross"], "Equal-weight 9 (gross)"),
        "spy": stats(spy, "SPY buy & hold"),
        "avg_monthly_turnover_pct": {"top3": round(float(top3["turnover"].mean()) * 100, 1),
                                     "top1": round(float(top1["turnover"].mean()) * 100, 1),
                                     "equal_weight": round(float(ew["turnover"].mean()) * 100, 1)},
    }
    # sensitivity grid: every cell is evaluated on the same months as the main result
    grid = {}
    for T in (3, 6, 9, 12):
        for n in (1, 2, 3, 4):
            s = stats(run(px_m, T, n)[0]["net"].reindex(idx))
            grid[f"T{T}_N{n}"] = {"cagr_pct": s["cagr_pct"], "sharpe": s["sharpe"]}
    summary["sensitivity"] = grid
    # calendar-year returns
    yr = pd.DataFrame({"top3": top3["net"], "top1": top1["net"], "equal_weight": ew["net"], "spy": spy})
    summary["annual_returns_pct"] = (
        ((1 + yr).groupby(yr.index.year).prod() - 1) * 100).round(2).to_dict("index")

    # ---- SAVE-BEGIN: optional, uncomment to write the results to ./data ----
    # import json
    # from pathlib import Path
    # out = Path("data"); out.mkdir(exist_ok=True)
    # px.to_csv(out / "prices_python.csv")
    # (out / "results_python.json").write_text(json.dumps(summary, indent=2, default=str))
    # pd.DataFrame({"top3_net": top3["net"], "top3_gross": top3["gross"], "top1_net": top1["net"],
    #               "ew_net": ew["net"], "ew_gross": ew["gross"], "spy": spy, "turnover_top3": top3["turnover"]}
    #              ).to_csv(out / "monthly_returns_python.csv", float_format="%.10f")
    # w.to_csv(out / "weights_python.csv", float_format="%.6f")
    # ---- SAVE-END ----

    print(results_table(summary).round(2).to_string())
    plot_results(top3["net"], top1["net"], ew["net"], spy, grid)
    return top3, w, rcum, px_m, ew, spy, summary


if __name__ == "__main__":
    main()
