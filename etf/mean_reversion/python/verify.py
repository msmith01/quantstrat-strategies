"""Compares the Python and R results for the IBS study. Exits non-zero on failure."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
py = json.load(open(ROOT / "data/results_python.json"))
ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, detail)
rb = pd.read_csv(ROOT / "data/ibs_r_buckets.csv")
d = 0
for per, rows in py["table3"].items():
    for t, r in rows.items():
        for k in range(5):
            x = rb[(rb.period == per) & (rb.ticker == t) & (rb.bucket == k + 1)]
            d = max(d, abs(float(x["mean"].iloc[0]) - r["mean"][k]), abs(float(x["up"].iloc[0]) - py["table4"][per][t]["up"][k]))
check("bucket mean return and P(up) for every ETF, bucket and period", d < 1e-8, f"max abs diff {d:.2e}")
rs = pd.read_csv(ROOT / "data/ibs_r_strategy.csv"); d = 0
for per, cs in py["ibs_strategy_costs"].items():
    for c, v in cs.items():
        d = max(d, abs(float(rs[(rs.period == per) & (rs.cost == int(c))].avg_cagr.iloc[0]) - v["avg_cagr"]))
check("single-ETF IBS strategy average CAGR at each cost and period", d < 1e-6, f"max abs diff {d:.2e}")
rd = pd.read_csv(ROOT / "data/ibs_r_decile.csv"); d = 0
for per, v in py["decile_rule"].items():
    for c in ("0", "2", "5", "10"):
        x = rd[(rd.period == per) & (rd.cost == int(c))].iloc[0]
        d = max(d, abs(float(x.cagr) - v[c]["cagr"]), abs(float(x.cagr_drift) - v[c + "_drift"]["cagr"]), abs(float(x.cagr_drift_borrow) - v[c + "_drift_borrow"]["cagr"]))
check("decile rule CAGR (target-weight and drifting-holdings turnover, with borrow fee) at each cost and period", d < 1e-6, f"max abs diff {d:.2e}")
ru = pd.read_csv(ROOT / "data/ibs_r_up_probabilities.csv"); d = 0
for per, rows in py["table4"].items():
    for tk, r in rows.items():
        for k in range(5):
            d = max(d, abs(float(ru[(ru.period == per) & (ru.ticker == tk) & (ru.bucket == k + 1)].up.iloc[0]) - r["up"][k]))
check("Table 4 style P(up) by bucket, exported separately from R", d < 1e-8, f"max abs diff {d:.2e}")
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
