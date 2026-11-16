"""Checks that the independent R implementation (r/ag_replication.R) reproduces the Python replication, and that one
fund-year regression matches statsmodels. Exits non-zero on failure. Run after both scripts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
py = pd.read_csv(ROOT / "data/ag_fund_years.csv")
r = pd.read_csv(ROOT / "data/ag_r_fund_years.csv")
res = json.load(open(ROOT / "data/results_ag_python.json"))
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    if not ok:
        fails.append(name)


check("same fund-years", len(py) == len(r) and set(zip(py.fund, py.year)) == set(zip(r.fund, r.year)), f"{len(py)} vs {len(r)}")
m = py.merge(r, on=["fund", "year"], suffixes=("_py", "_r"))
for c in ("alpha_p", "r2_p", "rmse_p", "sdr_p", "ir_p", "alpha", "ir", "tr2"):
    d = float(np.abs(m[c + "_py"] - m[c + "_r"]).max())
    check(f"fund-year {c} agrees", d < 1e-6, f"max abs diff {d:.2e}")
for key, f in (("table8_alpha", "ag_r_table8_alpha.csv"), ("table8_ir", "ag_r_table8_ir.csv"),
               ("table9_alpha", "ag_r_table9_alpha.csv"), ("table9_ir", "ag_r_table9_ir.csv")):
    t = pd.read_csv(ROOT / "data" / f)
    dm = dt = 0.0
    for _, row in t.iterrows():
        pm, pt, pn = res[key]["cell"][f"{int(row.alpha_q)}_{int(row.r2_q)}"]
        dm, dt = max(dm, abs(pm - row["mean"])), max(dt, abs(pt - row["t"]))
        assert pn == row["n"], (key, row.alpha_q, row.r2_q)
    check(f"{key} 25 cells agree (mean, t, n)", dm < 1e-6 and dt < 1e-6, f"max diff mean {dm:.1e}, t {dt:.1e}")
reg = pd.read_csv(ROOT / "data/ag_r_regressions.csv").set_index(["spec", "var"])
pairs = [(("alpha_tr2", "tr2"), res["table2"]["full"]["tr2"], "tr2"),
         (("alpha_comp_lag", "sdr_p"), res["table2"]["full"]["comp_lag"], "sdr_p"),
         (("ir_tr2_lag", "tr2"), res["table3"]["full"]["tr2_lag"], "tr2")]
for (k, p, v) in pairs:
    rc, rt = reg.loc[k, "coef"], reg.loc[k, "t"]
    check(f"regression {k[0]}:{v} (coef, clustered t)", abs(rc - p["coef"][v]) < 1e-6 * max(1, abs(rc)) and abs(rt - p["t"][v]) < 1e-6, f"coef {rc:.5g} vs {p['coef'][v]:.5g}; t {rt:.4f} vs {p['t'][v]:.4f}")
# independent hand check of one fund-year (statsmodels): R2, alpha for the first fund in 2020
try:
    import statsmodels.api as sm
    sys.path.insert(0, str(ROOT / "python"))
    import os
    os.environ.setdefault("QS_RAW_DIR", str(ROOT / "data/raw"))
    import r_squared as R
    R.pd.set_option("compute.use_numexpr", False)
    px = pd.read_csv(ROOT / "data/funds/ag_prices.csv", index_col=0, parse_dates=True)
    ff = R.download_factors()
    fund = r.loc[r.year == 2021, "fund"].iloc[0]
    fac = ff[R.FACTORS]
    Z = pd.concat([fac, fac.shift(1).add_suffix("_L")], axis=1)
    d = (px[[fund]] / px[[fund]].shift(1) - 1).join(Z, how="inner").join(ff["RF"]).dropna()
    d = d[d.index.year == 2020]
    fit = sm.OLS(d[fund] - d["RF"], sm.add_constant(d[list(Z.columns)])).fit()
    row = py[(py.fund == fund) & (py.year == 2021)].iloc[0]
    check("statsmodels hand check of one fund-year (alpha_p, R2_p)", abs(fit.params["const"] * 252 * 100 - row.alpha_p) < 1e-6 and abs(fit.rsquared - row.r2_p) < 1e-9,
          f"{fund}: alpha {fit.params['const'] * 25200:.4f} vs {row.alpha_p:.4f}; R2 {fit.rsquared:.6f} vs {row.r2_p:.6f}")
except ImportError:
    print("statsmodels not installed: hand check skipped")
print("ALL PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
