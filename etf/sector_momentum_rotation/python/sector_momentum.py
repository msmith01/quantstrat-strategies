"""Sector momentum rotation  --  replication of Kakushadze & Serur (2018),
"151 Trading Strategies", Section 4.1, Eq. (361).

    R_cum_i(t) = P_i(t) / P_i(t - T) - 1

At each month-end t: rank the sector ETFs by R_cum_i(t), buy the top N
equal-weighted, hold for one month (the paper's 1-3 month holding period).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
T_MONTHS = 12      # formation period (paper: 6-12 months)
TOP_N = 3          # 9 ETFs -> "top decile" would be <1 ETF; top third used
COST_BPS = 10      # one-way cost per unit of turnover


def monthly_prices(path=ROOT / "data" / "prices.csv"):
    px = pd.read_csv(path, index_col="date", parse_dates=True)
    return px.resample("ME").last()


def run(px_m, T=T_MONTHS, n=TOP_N, cost_bps=COST_BPS):
    sect = px_m[SECTORS]
    r = sect.pct_change()                       # month-m return of each ETF
    rcum = sect / sect.shift(T) - 1             # Eq. (361), known at end of month m
    rank = rcum.rank(axis=1, ascending=False, method="first")
    w_signal = (rank <= n).astype(float).where(rcum.notna())
    w_signal = w_signal.div(w_signal.sum(axis=1), axis=0)   # equal weight
    w = w_signal.shift(1)                       # trade at month-end, earn NEXT month
    gross = (w * r).sum(axis=1, min_count=1)
    turnover = w.diff().abs().sum(axis=1)
    first = w.dropna(how="all").index[0]
    turnover.loc[first] = 1.0                   # initial purchase
    net = gross - turnover * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "net": net, "turnover": turnover}).loc[first:]
    return out, w.loc[first:], rcum


def stats(ret, label=""):
    ret = ret.dropna()
    yrs = len(ret) / 12
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "label": label,
        "start": str(ret.index[0].date()), "end": str(ret.index[-1].date()),
        "months": int(len(ret)),
        "total_return_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
        "ann_vol_pct": round(float(ret.std(ddof=1) * np.sqrt(12)) * 100, 2),
        "sharpe": round(float(ret.mean() / ret.std(ddof=1) * np.sqrt(12)), 3),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "hit_rate_pct": round(float((ret > 0).mean()) * 100, 1),
    }


def main():
    px_m = monthly_prices()
    res, w, rcum = run(px_m)
    spy = px_m["SPY"].pct_change().loc[res.index]
    ew = px_m[SECTORS].pct_change().mean(axis=1).loc[res.index]

    summary = {
        "params": {"T_months": T_MONTHS, "top_n": TOP_N, "cost_bps": COST_BPS,
                   "universe": SECTORS},
        "strategy_net": stats(res["net"], "Sector momentum (net of costs)"),
        "strategy_gross": stats(res["gross"], "Sector momentum (gross)"),
        "equal_weight_sectors": stats(ew, "Equal-weight 9 sectors"),
        "spy": stats(spy, "SPY buy & hold"),
        "avg_monthly_turnover_pct": round(float(res["turnover"].mean()) * 100, 1),
    }
    # sensitivity grid: net CAGR and Sharpe over (T, N)
    grid = {}
    for T in (3, 6, 9, 12):
        for n in (1, 2, 3, 4):
            s = stats(run(px_m, T, n)[0]["net"])
            grid[f"T{T}_N{n}"] = {"cagr_pct": s["cagr_pct"], "sharpe": s["sharpe"]}
    summary["sensitivity"] = grid
    # calendar-year table
    yr = pd.DataFrame({"strategy": res["net"], "equal_weight": ew, "spy": spy})
    summary["annual_returns_pct"] = (
        ((1 + yr).groupby(yr.index.year).prod() - 1) * 100).round(2).to_dict("index")

    out = ROOT / "data"
    (out / "results_python.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame({"strategy_net": res["net"], "strategy_gross": res["gross"],
                  "equal_weight": ew, "spy": spy, "turnover": res["turnover"]}
                 ).to_csv(out / "monthly_returns_python.csv", float_format="%.10f")
    w.to_csv(out / "weights_python.csv", float_format="%.6f")
    for k in ("strategy_net", "equal_weight_sectors", "spy"):
        print(summary[k])
    return res, w, rcum, px_m, ew, spy, summary


if __name__ == "__main__":
    main()
