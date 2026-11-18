"""Compares Python and R results for the 4.5 study. Exits non-zero on failure."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
py = json.load(open(ROOT / "data/results_python.json")); ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, detail)
rd = pd.read_csv(ROOT / "data/letf_r_decay.csv").set_index("sample"); d = 0
for k in ("all", "full"):
    a, r = py["decay_regression"][k], rd.loc[k]
    d = max(d, abs(r.n - a["n"]), abs(r.intercept - a["intercept"]), abs(r.slope - a["slope"]), abs(r.r2 - a["r2"]))
check("decay regression, all years and full years (n, intercept, slope, R-squared)", d < 1e-9, f"max abs diff {d:.2e}; slope R {rd.loc['full'].slope:.6f} Python {py['decay_regression']['full']['slope']:.6f}")
rs = pd.read_csv(ROOT / "data/letf_r_strategy.csv"); d = 0
for _, r in rs.iterrows():
    s = py["strategy"][r.pair]
    v = s["scen"][r.scen[5:]]["cagr"] if r.scen.startswith("scen:") else s[r.scen]["cagr"]
    d = max(d, abs(r.cagr - v))
check("strategy return for every pair, reset schedule, borrow fee and scenario (trade cost, no interest, margin call)", d < 1e-6, f"max abs diff {d:.2e} over {len(rs)} cells")
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
