"""Table II / III with proxy controls from Yahoo's CURRENT fund snapshot (fund_info.json): log total net assets and its square,
expense ratio, turnover and age. These are today's values used for every year, so they carry look-ahead and survivorship
(the paper uses end-of-year values). The article labels them proxies and keeps the no-controls result as the headline.
Writes data/results_ag_controls.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ag_replication_lib import cluster_ols, design  # noqa: E402  (helpers shared with ag_replication.py)

D = ROOT / "data/funds"
info = json.load(open(D / "fund_info.json"))
groups = pd.read_csv(D / "share_class_groups.csv")
fy = pd.read_csv(ROOT / "data/ag_fund_years.csv")
fy["alpha_lag"], fy["ir_lag"] = fy["alpha_p"], fy["ir_p"]


def fund_row(classes):
    recs = [info.get(c) for c in classes.split() if info.get(c)]
    tna = [r["totalAssets"] or r["netAssets"] for r in recs if (r.get("totalAssets") or r.get("netAssets"))]
    exp = [r["netExpenseRatio"] if r.get("netExpenseRatio") is not None else (r["annualReportExpenseRatio"] * 100 if r.get("annualReportExpenseRatio") is not None else None) for r in recs]
    exp = [e for e in exp if e is not None]
    tov = [r["annualHoldingsTurnover"] * 100 for r in recs if r.get("annualHoldingsTurnover") is not None]
    inc = [r["fundInceptionDate"] for r in recs if r.get("fundInceptionDate")]
    return pd.Series({"tna": max(tna) / 1e6 if tna else np.nan, "expenses": float(np.mean(exp)) if exp else np.nan,
                      "turnover": float(np.mean(tov)) if tov else np.nan, "inception": pd.to_datetime(min(inc), unit="s") if inc else pd.NaT})


ctl = groups.set_index("kept")["classes"].apply(fund_row)
fy = fy.join(ctl, on="fund")
fy["age"] = (pd.to_datetime(fy.year.astype(str) + "-01-01") - fy.inception).dt.days / 365.25
fy["ltna"] = np.log(fy.tna)
fy["ltna2"] = fy.ltna ** 2
cov = {c: float(fy[c].notna().mean()) for c in ("tna", "expenses", "turnover", "age")}
use = fy.dropna(subset=["ltna", "expenses", "turnover", "age"])
use = use[use.age > 0]
out = {"coverage_share_of_fund_years": cov, "n_used": int(len(use)), "n_total": int(len(fy)),
       "summary": {c: {"mean": float(use[c].mean()), "median": float(use[c].median())} for c in ("tna", "expenses", "turnover", "age")}}


def panel(dep, cols, df):
    X = design(df, cols)
    b, se, r2 = cluster_ols(df[dep].to_numpy(float), X, df["fund"].to_numpy())
    return {"coef": {c: float(b[1 + i]) for i, c in enumerate(cols)}, "t": {c: float(b[1 + i] / se[1 + i]) for i, c in enumerate(cols)},
            "n_funds": int(df.fund.nunique()), "n_fund_years": int(len(df)), "r2": float(r2)}


ctrl = ["expenses", "ltna", "ltna2", "turnover", "age"]
for tname, dep, lag in (("table2", "alpha", "alpha_lag"), ("table3", "ir", "ir_lag")):
    t = {}
    for per, (a, b) in {"full": (2012, 2025), "early": (2012, 2018), "late": (2019, 2025)}.items():
        sub = use[(use.year >= a) & (use.year <= b)]
        t[per] = {"tr2": panel(dep, ["tr2", lag] + ctrl, sub), "comp": panel(dep, ["rmse_p", "sdr_p", lag] + ctrl, sub)}
    out[tname] = t
json.dump(out, open(ROOT / "data/results_ag_controls.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("coverage_share_of_fund_years", "n_used", "n_total")}))
print("alpha ~ TR2 + controls (full):", {k: round(v, 3) for k, v in out["table2"]["full"]["tr2"]["coef"].items()})
print("t:", {k: round(v, 2) for k, v in out["table2"]["full"]["tr2"]["t"].items()})
