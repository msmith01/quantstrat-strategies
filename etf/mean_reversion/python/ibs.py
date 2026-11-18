"""4.4 Mean reversion: the IBS effect in equity ETFs. Replicates Pagonidis (2014), "The IBS Effect: Mean Reversion in Equity ETFs"
(NAAIM working paper) on Yahoo daily OHLC for the paper's ETFs, for the paper's own window (to 5 Dec 2013) and for the period after,
and adds the book's cross-sectional decile rule (Section 4.4, Eq. 370). Writes data/results_python.json and data/*.csv.
Definitions follow the paper; where it is silent the choice is noted in SPEC below and in the article."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUT = pd.Timestamp("2013-12-05")            # last day of the paper's sample
SPEC = {"buckets": 5, "long_below": 0.2, "short_above": 0.8, "vol_window": 20, "range_window": 500, "range_min": 250, "bull_window": 200,
        "rsi_n": 3, "rsi_entry": 10, "rsi_hold": 40, "ibs_filter": 0.5, "decile_share": 0.1, "cost_bps": [0, 2, 5, 10]}

raw = pd.read_csv(ROOT / "data/ohlc_all.csv", parse_dates=["Date"], float_precision="round_trip")
# Days with High == Low (no range) or a close outside the range have no IBS; they stay in the calendar so returns are true one-day returns.
bad = raw[(raw.High <= raw.Low)]
raw = raw.copy()
raw["ibs"] = (raw.Close - raw.Low) / (raw.High - raw.Low).where(raw.High > raw.Low)
n_out = int(((raw.ibs < -1e-6) | (raw.ibs > 1 + 1e-6)).sum())
raw.loc[(raw.ibs < -1e-6) | (raw.ibs > 1 + 1e-6), "ibs"] = np.nan
raw["ibs"] = raw.ibs.clip(0, 1)
D = {}
for t, g in raw.groupby("ticker"):
    g = g.sort_values("Date").set_index("Date").copy()
    g["ret"] = g.Close.pct_change()
    g["nxt"] = g.ret.shift(-1)                                     # close-to-close return of the next trading day
    g["ndate"] = g.index.to_series().shift(-1)                      # date on which that return is earned
    g["vol10"] = g.ret.rolling(SPEC["vol_window"]).std()
    rng = (g.High - g.Low) / g.Close
    g["range_rel"] = rng / rng.rolling(SPEC["range_window"], min_periods=SPEC["range_min"]).median().shift(1)
    vm = g.Volume.rolling(SPEC["range_window"], min_periods=SPEC["range_min"]).median().shift(1)
    g["vol_rel"] = np.where(g.Volume > 0, g.Volume / vm, np.nan)
    g["trend"] = g.Close / g.Close.shift(SPEC["bull_window"]) - 1
    g["dow"] = g.index.dayofweek
    d = g.Close.diff()
    u, dn = d.clip(lower=0).rolling(SPEC["rsi_n"]).mean(), (-d.clip(upper=0)).rolling(SPEC["rsi_n"]).mean()
    g["rsi"] = np.where(dn == 0, 100.0, 100 - 100 / (1 + u / dn.replace(0, np.nan)))
    g.loc[u.isna() | dn.isna(), "rsi"] = np.nan
    D[t] = g
TICK = sorted(D)
PER = {"paper_window": lambda i: i <= CUT, "after": lambda i: i > CUT, "all": lambda i: i == i}
BUCKET = lambda x: np.minimum((x * SPEC["buckets"]).astype(int), SPEC["buckets"] - 1) + 1     # 1 = lowest IBS


def tt(x, mu=0.0):
    x = x.dropna()
    return float((x.mean() - mu) / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def welch(a, b):
    a, b = a.dropna(), b.dropna()
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) if len(a) > 2 and len(b) > 2 else np.nan


def valid(g, per):
    g = g[PER[per](g.index) & PER[per](pd.DatetimeIndex(g.ndate))]    # signal date AND the return date must both lie in the window
    return g[g.nxt.notna() & g.ibs.notna()]


out = {"params": {"spec": SPEC, "tickers": TICK, "n_tickers": len(TICK), "last_date": str(max(g.index.max() for g in D.values()).date()),
                  "rows_without_ibs_high_eq_low": int(len(bad)), "rows_ibs_outside_0_1_set_missing": n_out,
                  "first_dates": {t: str(D[t].index.min().date()) for t in TICK}}}

# ---------- Table 2: descriptive statistics ----------
t2 = {}
for per in PER:
    rows = {}
    for t in TICK:
        r = valid(D[t], per).nxt
        if len(r) > 30:
            rows[t] = {"n": int(len(r)), "mean": float(r.mean() * 100), "sd": float(r.std() * 100), "skew": float(r.skew()), "up": float((r > 0).mean() * 100)}
    t2[per] = rows
out["table2"] = t2

# ---------- Tables 3 and 4: next-day return and P(up) by IBS bucket ----------
t3, t4 = {}, {}
for per in PER:
    rows3, rows4 = {}, {}
    for t in TICK:
        g = valid(D[t], per)
        if len(g) < 250:
            continue
        b = BUCKET(g.ibs)
        mu = g.nxt.mean()
        r3 = {"n": [int((b == k).sum()) for k in range(1, 6)]}
        r3["mean"] = [float(g.nxt[b == k].mean() * 100) for k in range(1, 6)]
        r3["t_vs_avg"] = [tt(g.nxt[b == k], mu) for k in range(1, 6)]
        r3["diff_1_5"] = r3["mean"][0] - r3["mean"][4]
        r3["t_diff"] = welch(g.nxt[b == 1], g.nxt[b == 5])
        rows3[t] = r3
        up = (g.nxt > 0).astype(float)
        rows4[t] = {"up": [float(up[b == k].mean() * 100) for k in range(1, 6)]}
        rows4[t]["diff_1_5"] = rows4[t]["up"][0] - rows4[t]["up"][4]
    t3[per], t4[per] = rows3, rows4
out["table3"], out["table4"] = t3, t4


def avg_row(rows, key):
    return [float(np.mean([r[key][k] for r in rows.values()])) for k in range(5)]


out["table3_avg"] = {per: {"mean": avg_row(t3[per], "mean"), "diff_1_5": float(np.mean([r["diff_1_5"] for r in t3[per].values()])),
                           "n_etfs": len(t3[per]),
                           "n_sig_diff": int(sum(abs(r["t_diff"]) > 2.576 and r["diff_1_5"] > 0 for r in t3[per].values())),
                           "n_pos_diff": int(sum(r["diff_1_5"] > 0 for r in t3[per].values()))} for per in PER}
out["table4_avg"] = {per: {"up": avg_row(t4[per], "up"), "diff_1_5": float(np.mean([r["diff_1_5"] for r in t4[per].values()]))} for per in PER}

# ---------- conditional tables: volatility, range, trend, volume, weekday ----------
def cond_table(per, label, split):
    """Average over ETFs of the 1st and 5th bucket mean (raw and vol-adjusted) in each regime."""
    res = {}
    for reg in split["levels"]:
        a_raw, a_adj = [], []
        for t in TICK:
            g = valid(D[t], per)
            if len(g) < 500:
                continue
            m = split["fn"](g)
            if m is None:
                continue
            s = g[m == reg]
            b = BUCKET(s.ibs)
            adj = s.nxt / s.vol10
            if min((b == 1).sum(), (b == 5).sum()) < 20:
                continue
            a_raw.append((s.nxt[b == 1].mean() * 100, s.nxt[b == 5].mean() * 100))
            a_adj.append((adj[b == 1].mean(), adj[b == 5].mean()))
        if a_raw:
            r = np.array(a_raw)
            ad = np.array(a_adj)
            res[str(reg)] = {"n_etfs": len(a_raw), "b1": float(r[:, 0].mean()), "b5": float(r[:, 1].mean()), "diff": float((r[:, 0] - r[:, 1]).mean()),
                             "adj_b1": float(ad[:, 0].mean()), "adj_b5": float(ad[:, 1].mean()), "adj_diff": float((ad[:, 0] - ad[:, 1]).mean())}
    return res


def by_median(col):
    def fn(g):
        x = g[col]
        if x.notna().sum() < 500:
            return None
        return pd.Series(np.where(x.isna(), "na", np.where(x >= x.median() if col == "vol10" else x >= 1.0, "high", "low")), index=g.index)
    return fn


SPLITS = {"volatility": {"levels": ["high", "low"], "fn": by_median("vol10")},
          "range": {"levels": ["high", "low"], "fn": by_median("range_rel")},
          "trend": {"levels": ["bull", "bear"], "fn": lambda g: pd.Series(np.where(g.trend.isna(), "na", np.where(g.trend > 0, "bull", "bear")), index=g.index)},
          "volume": {"levels": ["high", "low"], "fn": by_median("vol_rel")},
          "weekday": {"levels": ["Mon", "Tue", "Wed", "Thu", "Fri"], "fn": lambda g: pd.Series(g.dow.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}).to_numpy(), index=g.index)}}
out["conditional"] = {per: {k: cond_table(per, k, v) for k, v in SPLITS.items()} for per in PER}

# ---------- skewness by bucket ----------
sk = {}
for per in PER:
    rows = []
    for t in TICK:
        g = valid(D[t], per)
        if len(g) < 500:
            continue
        b = BUCKET(g.ibs)
        rows.append([float(g.nxt[b == k].skew()) for k in range(1, 6)])
    sk[per] = [float(x) for x in np.nanmean(np.array(rows), axis=0)]
out["skew_by_bucket"] = sk

# ---------- Table 5: IBS long/short per ETF (long if IBS <= 0.2, short if >= 0.8), yearly compounded returns ----------
def pos_ibs(g):
    return pd.Series(np.where(g.ibs <= SPEC["long_below"], 1.0, np.where(g.ibs >= SPEC["short_above"], -1.0, 0.0)), index=g.index)


def strat_returns(g, pos, cost_bps=0.0):
    """Position set at the close of day t earns day t+1's close-to-close return; cost on the change of position."""
    p = pos.shift(1).fillna(0.0)                                  # position held during day t (decided at close t-1)
    gross = (p * g.ret).where(g.ret.notna(), 0.0)
    turn = (pos.shift(1).fillna(0) - pos.shift(2).fillna(0)).abs()
    return gross - turn * cost_bps / 1e4


yearly, summary = {}, {}
for t in TICK:
    g = D[t]
    r = strat_returns(g, pos_ibs(g))
    yearly[t] = {str(y): float(((1 + x).prod() - 1) * 100) for y, x in r.groupby(r.index.year)}
out["table5_yearly"] = yearly
_py = [v for t in TICK for y, v in yearly[t].items() if int(y) <= CUT.year and D[t].index[D[t].index.year == int(y)].size > 100]
out["table5_avg_etf_year_to_2013"] = float(np.mean(_py))
out["table5_avg_etf_year_to_2013_n"] = len(_py)
for per in PER:
    tot = []
    for t in TICK:
        g = D[t]
        m = PER[per](g.index)
        r = strat_returns(g, pos_ibs(g))[m]
        if len(r) < 250:
            continue
        yrs = len(r) / 252
        tot.append({"t": t, "cagr": float(((1 + r).prod() ** (1 / yrs) - 1) * 100), "in_mkt": float((pos_ibs(g)[m] != 0).mean() * 100)})
    summary[per] = {"avg_cagr": float(np.mean([x["cagr"] for x in tot])), "median_cagr": float(np.median([x["cagr"] for x in tot])),
                    "n_pos": int(sum(x["cagr"] > 0 for x in tot)), "n": len(tot), "avg_in_market_pct": float(np.mean([x["in_mkt"] for x in tot])),
                    "by_etf": {x["t"]: x["cagr"] for x in tot}}
out["ibs_strategy_gross"] = summary
costs = {}
for per in PER:
    costs[per] = {}
    for c in SPEC["cost_bps"]:
        v = []
        for t in TICK:
            g = D[t]
            m = PER[per](g.index)
            r = strat_returns(g, pos_ibs(g), c)[m]
            if len(r) >= 250:
                v.append(((1 + r).prod() ** (252 / len(r)) - 1) * 100)
        costs[per][str(c)] = {"avg_cagr": float(np.mean(v)), "n_pos": int(sum(x > 0 for x in v)), "n": len(v)}
out["ibs_strategy_costs"] = costs

# ---------- RSI(3) with and without the IBS filter ----------
def rsi_pos(g, filt):
    pos, prev = np.zeros(len(g)), 0.0
    for i, (rs, ib) in enumerate(zip(g.rsi.to_numpy(), g.ibs.to_numpy())):
        if np.isnan(rs):
            prev = 0.0
            continue
        ok = (not filt) or ib <= SPEC["ibs_filter"]
        prev = 1.0 if ok and (rs < SPEC["rsi_entry"] or (prev == 1.0 and rs <= SPEC["rsi_hold"])) else 0.0
        pos[i] = prev
    return pd.Series(pos, index=g.index)


rsi = {}
for per in PER:
    rows = []
    for t in TICK:
        g = D[t]
        m = PER[per](g.index)
        if m.sum() < 750:
            continue
        rr = {}
        for name, f in (("rsi", False), ("rsi_ibs", True)):
            p = rsi_pos(g, f)
            r = strat_returns(g, p)[m]
            rr[name] = {"total": float(((1 + r).prod() - 1) * 100), "sum": float(r.sum() * 100), "days": int((p[m] != 0).sum()), "in_mkt": float((p[m] != 0).mean() * 100), "trades": int(((p[m] == 1) & (p[m].shift(1).fillna(0) == 0)).sum())}
        rows.append({"t": t, **rr})
    rsi[per] = {"n": len(rows), "avg_sum_rsi": float(np.mean([x["rsi"]["sum"] for x in rows])), "avg_sum_filtered": float(np.mean([x["rsi_ibs"]["sum"] for x in rows])),
                "avg_days_rsi": float(np.mean([x["rsi"]["days"] for x in rows])), "avg_days_filtered": float(np.mean([x["rsi_ibs"]["days"] for x in rows])),
                "n_better_sum": int(sum(x["rsi_ibs"]["sum"] > x["rsi"]["sum"] for x in rows)),
                "avg_total_rsi": float(np.mean([x["rsi"]["total"] for x in rows])), "avg_total_filtered": float(np.mean([x["rsi_ibs"]["total"] for x in rows])),
                "avg_in_mkt_rsi": float(np.mean([x["rsi"]["in_mkt"] for x in rows])), "avg_in_mkt_filtered": float(np.mean([x["rsi_ibs"]["in_mkt"] for x in rows])),
                "avg_trades_rsi": float(np.mean([x["rsi"]["trades"] for x in rows])), "avg_trades_filtered": float(np.mean([x["rsi_ibs"]["trades"] for x in rows])),
                "n_better": int(sum(x["rsi_ibs"]["total"] > x["rsi"]["total"] for x in rows))}
out["rsi_filter"] = rsi

# ---------- the book's rule: cross-sectional decile sort on IBS, long bottom decile / short top decile ----------
ibs_w = pd.DataFrame({t: D[t].ibs for t in TICK})
ret_w = pd.DataFrame({t: D[t].ret for t in TICK})
nxt_w = ret_w.shift(-1)
# Eligibility uses only what is known at the close of the signal day (a valid IBS). A selected ETF with no next-day return
# (missing data, delisting) is counted below and its weight is spread over the other selected ETFs of the same leg.
cnt = ibs_w.notna().sum(axis=1)
k = np.maximum((cnt * SPEC["decile_share"]).round().astype(int), 1)
# ties (IBS of exactly 0 or 1 is common) are broken by a ticker order that rotates every day, so no ticker is systematically favoured
nT = len(TICK)
rot = (np.arange(nT)[None, :] + np.arange(len(ibs_w))[:, None]) % nT
ibs_t = ibs_w + pd.DataFrame(rot / nT * 1e-9, index=ibs_w.index, columns=ibs_w.columns)
rk = ibs_t.where(ibs_w.notna()).rank(axis=1, method="first")
long_m = rk.le(k, axis=0) & ibs_w.notna()
short_m = rk.gt(cnt - k, axis=0) & ibs_w.notna()
lw = long_m.div(long_m.sum(axis=1).replace(0, np.nan), axis=0)
sw = short_m.div(short_m.sum(axis=1).replace(0, np.nan), axis=0)
have = nxt_w.notna()


def leg_return(w):
    wa = w.where(have)
    return (wa * nxt_w).sum(axis=1, min_count=1) / wa.sum(axis=1).replace(0, np.nan)


rl, rs_ = leg_return(lw), leg_return(sw)
n_missing = int(((long_m | short_m) & ~have & nxt_w.index.to_series().lt(ret_w.index[-1]).to_numpy()[:, None]).sum().sum())
ew = nxt_w.where(ibs_w.notna()).mean(axis=1)
ok = cnt >= 10
dec = {}


def drift_turnover(w, r_leg):
    """Trades needed each day to go from yesterday's drifted holdings to today's target weights (one leg, weights sum to 1)."""
    prev_t = w.shift(1).fillna(0)
    drifted = prev_t.mul(1 + nxt_w.shift(1).fillna(0)).div((1 + r_leg.shift(1)).fillna(1), axis=0)
    return (w.fillna(0) - drifted).abs().sum(axis=1)


turn = ((lw.fillna(0) - lw.shift(1).fillna(0)).abs().sum(axis=1) + (sw.fillna(0) - sw.shift(1).fillna(0)).abs().sum(axis=1))
turn_d = drift_turnover(lw, rl) + drift_turnover(sw, rs_)
BORROW = 0.01                                                    # 1% a year on the short leg, charged daily, for the robustness scenario
for per in PER:
    m = pd.Series(PER[per](ibs_w.index), index=ibs_w.index) & ok & rl.notna() & rs_.notna()
    res = {}
    for c in SPEC["cost_bps"]:
        ls = (rl - rs_)[m] - turn.shift(0)[m] * c / 1e4
        yrs = m.sum() / 252
        res[str(c)] = {"cagr": float(((1 + ls).prod() ** (1 / yrs) - 1) * 100), "vol": float(ls.std() * np.sqrt(252) * 100), "sharpe": float(ls.mean() / ls.std() * np.sqrt(252)),
                       "mean_bp_day": float(ls.mean() * 1e4), "t": tt(ls), "max_dd": float(((1 + ls).cumprod() / (1 + ls).cumprod().cummax() - 1).min() * 100)}
        lsd = (rl - rs_)[m] - turn_d[m] * c / 1e4
        res[str(c) + "_drift"] = {"cagr": float(((1 + lsd).prod() ** (1 / yrs) - 1) * 100)}
        lsb = lsd - BORROW / 252
        res[str(c) + "_drift_borrow"] = {"cagr": float(((1 + lsb).prod() ** (1 / yrs) - 1) * 100)}
    res["long_leg_cagr"] = float(((1 + rl[m]).prod() ** (1 / (m.sum() / 252)) - 1) * 100)
    res["short_leg_cagr"] = float(((1 + rs_[m]).prod() ** (1 / (m.sum() / 252)) - 1) * 100)
    res["equal_weight_cagr"] = float(((1 + ew[m]).prod() ** (1 / (m.sum() / 252)) - 1) * 100)
    res["days"] = int(m.sum())
    res["avg_etfs"] = float(cnt[m].mean())
    res["avg_per_leg"] = float(k[m].mean())
    res["avg_daily_turnover"] = float(turn[m].mean())
    res["avg_daily_turnover_drift"] = float(turn_d[m].mean())
    res["selected_missing_next_return"] = n_missing if per == "all" else None
    dec[per] = res
out["decile_rule"] = dec
spy = D["SPY"].ret
okk = ok & rl.notna() & rs_.notna()
cum = pd.DataFrame({"decile_long_short": (1 + (rl - rs_)[okk]).cumprod() * 100,
                    "equal_weight_etfs": (1 + ew[okk]).cumprod() * 100})
cum["spy"] = (1 + spy.reindex(cum.index).fillna(0)).cumprod() * 100
cum.to_csv(ROOT / "data/cum_decile.csv")
daily = pd.DataFrame({f"net_{c}bp": ((rl - rs_) - turn * c / 1e4)[okk] for c in (0, 2, 5)})
daily["equal_weight"] = ew[okk]
daily["spy"] = spy.reindex(daily.index)
daily.to_csv(ROOT / "data/decile_daily.csv")
yr_avg = pd.DataFrame(out["table5_yearly"]).mean(axis=1)
yr_avg.to_csv(ROOT / "data/table5_avg_by_year.csv", header=["avg_return_pct"])
yd = pd.DataFrame({"decile_ls": (rl - rs_)[okk]})
out["decile_yearly"] = {str(y): float(((1 + x).prod() - 1) * 100) for y, x in yd.decile_ls.groupby(yd.index.year)}

pd.DataFrame({t: pd.Series(out["table3"]["paper_window"].get(t, {}).get("mean", [np.nan] * 5), index=range(1, 6)) for t in TICK}).T.to_csv(ROOT / "data/table3_paper_window.csv")
pd.DataFrame({t: pd.Series(out["table3"]["after"].get(t, {}).get("mean", [np.nan] * 5), index=range(1, 6)) for t in TICK}).T.to_csv(ROOT / "data/table3_after.csv")
json.dump(out, open(ROOT / "data/results_python.json", "w"), indent=1, default=str)
a = out["table3_avg"]
for per in PER:
    print(per, [round(x, 3) for x in a[per]["mean"]], "1-5:", round(a[per]["diff_1_5"], 3), "etfs", a[per]["n_etfs"], "pos", a[per]["n_pos_diff"], "sig", a[per]["n_sig_diff"],
          "| up", [round(x, 1) for x in out["table4_avg"][per]["up"]])
print("IBS strategy avg CAGR gross", {p: round(out["ibs_strategy_gross"][p]["avg_cagr"], 1) for p in PER}, "costs", {p: {c: round(v["avg_cagr"], 1) for c, v in out["ibs_strategy_costs"][p].items()} for p in PER})
print("decile", {p: {c: round(dec[p][c]["cagr"], 2) for c in ("0", "2", "5", "10")} | {"ew": round(dec[p]["equal_weight_cagr"], 2)} for p in PER})
print("rsi", out["rsi_filter"])
