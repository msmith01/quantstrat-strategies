"""Pilot: the 4.3 R-squared strategy on a random sample of Yahoo US equity mutual funds, 2012 onward.
Usage: python fund_test.py [n_funds] [seed]. Reuses the engine in r_squared.py with the universe swapped to funds."""
import json, os, sys
import numpy as np, pandas as pd, yfinance as yf
os.environ.setdefault("QS_RAW_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "raw"))
import r_squared as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
START_PX, FIRST_RET = "2010-06-01", "2012-01-31"
uni = json.load(open(R.Path(__file__).resolve().parents[1] / "data/funds/universe_raw.json"))
cand = sorted(k for k, v in uni.items() if v["cur"] == "USD")
tick = list(np.random.default_rng(SEED).choice(cand, N, replace=False)) if N else []
cache = R.Path(__file__).resolve().parents[1] / f"data/funds/sample_{N}_{SEED}.parquet"
if N == 0:                                      # full filtered universe (fund_prices.py output) plus SPY
    px = pd.read_parquet(cache.parent / "fund_prices.parquet").loc[START_PX:]
    ded = pd.read_csv(cache.parent / "yahoo_us_equity_funds_deduped.csv").ticker.tolist()   # one class per fund, bad prices out
    px = px[ded]
    px["SPY"] = yf.download("SPY", start=START_PX, end=R.END, auto_adjust=True, progress=False)["Close"].squeeze()
elif cache.exists():
    px = pd.read_parquet(cache)
else:
    px = yf.download(tick + ["SPY"], start=START_PX, end=R.END, auto_adjust=True, progress=False)["Close"]
    px.to_parquet(cache)
px = px.dropna(how="all", axis=1)
ff = R.download_factors()
last = min(px.index[-1], ff.index[-1]); px, ff = px.loc[:last], ff.loc[:last]
F = [c for c in px.columns if c != "SPY"]
v = px[F].to_numpy(); mx = np.nanmax(np.abs(v[1:] / v[:-1] - 1), axis=0)      # data-error screen: a fund with any one-day move
bad = [f for f, m in zip(F, mx) if m > 0.30]                                  # beyond +-30% has an unadjusted split / bad print
F = [f for f in F if f not in bad]; px = px[F + ["SPY"]]
print("excluded for implausible daily moves:", len(bad))
pd.Series(bad, name="ticker").to_csv(R.Path(__file__).resolve().parents[1] / "data/funds/excluded_bad_prices.csv", index=False)
R.UNIVERSE = F
print("funds with any data:", len(F), "of", N)


def fit(px, ff, window=R.WINDOW):
    """Like R.rolling_fit but each fund needs a complete window of daily returns (funds start at different dates)."""
    ret = R.pct(px[F])
    d = ret.join(ff, how="inner")
    y = d[F].sub(d["RF"], axis=0); X = d[R.FACTORS]
    A, Rr, T = {}, {}, {}
    for t in d.groupby(d.index.to_period("M")).tail(1).index:
        i = d.index.get_loc(t)
        if i + 1 < window: continue
        s = slice(i + 1 - window, i + 1)
        ys = y.iloc[s]; ok = ys.notna().all().to_numpy()
        if ok.sum() < R.NQ * 2: continue
        b, tt, rr = R.ols(ys.to_numpy()[:, ok], X.iloc[s].to_numpy())
        m = t + pd.offsets.MonthEnd(0)
        for dct, v in ((A, b[0]), (Rr, rr), (T, tt[0])):
            r = pd.Series(np.nan, index=F); r[ok] = v; dct[m] = r
    return pd.DataFrame(A).T, pd.DataFrame(T).T, pd.DataFrame(Rr).T


px_m = R.to_monthly(px)
alpha, tstat, r2 = (x.reindex(px_m.index) for x in fit(px, ff))
res = {}
for v in ("r2_alpha", "long_leg", "alpha_only", "r2_only"):
    lg, sh = R.legs(alpha, r2, v)
    res[v] = R.run_legs(px_m, lg, sh, start=pd.Timestamp(FIRST_RET))[0]
idx = res["r2_alpha"].index
idx = idx[idx >= FIRST_RET]
ew_l = alpha.notna()
res["equal_weight"] = R.run_legs(px_m, ew_l, ew_l & False, start=pd.Timestamp(FIRST_RET))[0]
spy = R.pct(px_m["SPY"])
rows = {k: R.stats(o["net"].reindex(idx), k) for k, o in res.items()}
rows["spy"] = R.stats(spy.reindex(idx), "spy")
df = pd.DataFrame(rows).T[["start", "end", "months", "cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct"]]
df["turnover_pct"] = [round(res[k]["turnover"].reindex(idx).mean() * 100) if k in res else 0 for k in df.index]
print(df)
print("avg funds ranked per month:", round(alpha.loc[idx].notna().sum(axis=1).mean(), 1),
      " avg long/short size:", round(R.legs(alpha, r2, "r2_alpha")[0].loc[idx].sum(axis=1).mean(), 2),
      round(R.legs(alpha, r2, "r2_alpha")[1].loc[idx].sum(axis=1).mean(), 2))
