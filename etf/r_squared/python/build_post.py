"""Builds post.html (WordPress content) and preview.html for the 4.3 article from post_template.html.
Every number comes from data/results_ag_python.json, data/results_funds_python.json, data/results_ag_controls.json and
data/verification_ag.txt. Run after ag_replication.py, fund_study.py, ag_controls.py, verify_ag.py and make_charts.py."""
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
from pygments.lexers import PythonLexer, SLexer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import tables as T
from postlib import code_box, eq, fn, frac, hl, hl_function, mi, mn, mo, mop, row, sub, sup, sqrt, tabs

THEME_ASSETS = ROOT.parents[2] / "wp_theme_build" / "theme" / "genebook" / "assets"
WP_IMG = (ROOT / "wp_img_prefix.txt").read_text().strip() if (ROOT / "wp_img_prefix.txt").exists() else "charts/"
GH_TREE = "https://github.com/msmith01/quantstrat-strategies/tree/main/etf/r_squared"
ag = json.load(open(ROOT / "data/results_ag_python.json"))
mf = json.load(open(ROOT / "data/results_funds_python.json"))
ct = json.load(open(ROOT / "data/results_ag_controls.json"))
ver = (ROOT / "data/verification_ag.txt").read_text(encoding="utf-8")
man = json.load(open(ROOT / "data/MANIFEST.json")) if (ROOT / "data/MANIFEST.json").exists() else {}
P = ag["params"]
yy = ag["yearly"]
n = lambda x: f"{x:,}"
MINUS = "&minus;"


def sg(x, d=2):
    return (MINUS if x < 0 else "") + f"{abs(x):.{d}f}"


# ---------- derived numbers ----------
c85 = ag["table8_alpha"]["cell"]["5_1"]
t2 = ag["table2"]
t3 = ag["table3"]
beat = sum(1 for r in yy if r["ret_portfolio"] > r["ret_all_funds"])
abeat = sum(1 for r in yy if r["alpha_portfolio"] > r["alpha_all"])
apos = sum(1 for r in yy if r["alpha_portfolio"] > 0)
gap = sorted(yy, key=lambda r: r["ret_portfolio"] - r["ret_all_funds"])
cum = pd.read_csv(ROOT / "data/ag_cum_annual.csv", index_col=0)
YRS = len(cum) - 0.25                                          # 2026 is January to September
cagr = lambda s: (cum[s].iloc[-1] / 100) ** (1 / YRS) * 100 - 100
hl_all = ag["table8_alpha"]["high_minus_low_alpha"]
hl_pos = sum(1 for q in "12345" if hl_all[q][0] > 0)
ver_diffs = [float(x) for x in re.findall(r"max abs diff ([\d.e+-]+)", ver)]
ver_fund = re.search(r"AAAGX: (alpha [-\d.]+ vs [-\d.]+; R2 [\d.]+ vs [\d.]+)", ver)
sdr = t2["full"]["comp"]
lag = t2["full"]["tr2_lag"]
fr = mf["factor_regression"]
mfm = lambda k, f: mf[k][f]
sub_p = mf["subperiods"]
V = {
    "n_screen": "5,607", "n_hist": "3,156", "n_bad": "83", "n_bad_left": "3,073", "n_dedupe": "1,458", "n_index": str(P["index_named_excluded"]),
    "n_funds": n(P["funds_after_index_filter"]), "n_fy": n(P["pairs_after_trim"]), "n_pairs": n(P["pairs_before_trim"]),
    "r2_lo": f"{P['r2_trim_low']:.2f}", "r2_hi": f"{P['r2_trim_high']:.2f}",
    "ff_vintage": re.search(r"(\d{6}) CRSP", man.get("factor_file_header", "202609 CRSP")).group(1) if man else "202609",
    "best_alpha": sg(c85[0]), "best_t": sg(c85[1], 1), "best_n": n(c85[2]),
    "all_alpha": sg(ag["table8_alpha"]["all"][0]),
    "tr2_coef": sg(t2["full"]["tr2"]["coef"]["tr2"], 3), "tr2_t": sg(t2["full"]["tr2"]["t"]["tr2"], 2),
    "tr2_early": f"{sg(t2['early']['tr2']['coef']['tr2'], 3)} (t-statistic {sg(t2['early']['tr2']['t']['tr2'], 2)})",
    "tr2_late": f"{sg(t2['late']['tr2']['coef']['tr2'], 3)} (t-statistic {sg(t2['late']['tr2']['t']['tr2'], 2)})",
    "sdr_coef": sg(sdr["coef"]["sdr_p"] / 100, 2), "sdr_t": sg(sdr["t"]["sdr_p"], 2),
    "ir_tr2": sg(t3["full"]["tr2"]["coef"]["tr2"], 4), "ir_tr2_t": sg(t3["full"]["tr2"]["t"]["tr2"], 2),
    "lag_coef": sg(lag["coef"]["alpha_lag"], 3), "lag_t": sg(lag["t"]["alpha_lag"], 2),
    "yrs_beat": str(beat), "nyears": str(len(yy)), "yrs_alpha_beat": str(abeat), "yrs_alpha_pos": str(apos),
    "ret_gap": f"{abs(sum(r['ret_portfolio'] - r['ret_all_funds'] for r in yy) / len(yy)):.1f}",
    "cagr_best": f"{cagr('low_r2_high_alpha'):.2f}", "cagr_all": f"{cagr('all_funds'):.2f}", "cagr_spy": f"{cagr('spy'):.2f}",
    "g_best": f"{cum['low_r2_high_alpha'].iloc[-1]:,.0f}", "g_all": f"{cum['all_funds'].iloc[-1]:,.0f}", "g_spy": f"{cum['spy'].iloc[-1]:,.0f}",
    "mean_r2": f"{ag['table1']['r2_p']['mean']:.2f}",
    "best_ts": sg(ag["table8_alpha"]["ts_lowR2_highAlpha"]["mean"]), "best_ts_t": sg(ag["table8_alpha"]["ts_lowR2_highAlpha"]["t"], 2),
    "lh_top": sg(ag["table8_alpha"]["low_minus_high_r2"]["5"][0]), "lh_top_t": sg(ag["table8_alpha"]["low_minus_high_r2"]["5"][1], 2),
    "lh_all": sg(ag["table8_alpha"]["low_minus_high_r2"]["All"][0]), "lh_all_t": sg(ag["table8_alpha"]["low_minus_high_r2"]["All"][1], 2),
    "lh_bot": sg(ag["table8_alpha"]["low_minus_high_r2"]["1"][0]),
    "hl_all": sg(hl_all["All"][0]), "hl_all_t": sg(hl_all["All"][1], 2), "hl_pos": {4: "four", 5: "all five", 3: "three"}.get(hl_pos, str(hl_pos)),
    "ir_lh": sg(ag["table8_ir"]["low_minus_high_r2"]["All"][0], 3), "ir_lh_t": sg(ag["table8_ir"]["low_minus_high_r2"]["All"][1], 2),
    "ind_best": sg(ag["table9_alpha"]["cell"]["5_1"][0]), "ind_best_t": sg(ag["table9_alpha"]["cell"]["5_1"][1], 2),
    "gap_worst": ", ".join(str(r["year"]) for r in gap[:3]), "gap_best": ", ".join(str(r["year"]) for r in gap[::-1][:3] if r["ret_portfolio"] > r["ret_all_funds"]),
    # monthly extension
    "m_cagr_abs": f"{abs(mf['r2_alpha']['cagr_pct']):.2f}", "m_gross_abs": f"{abs(mf['r2_alpha_gross']['cagr_pct']):.2f}",
    "ctl_tr2": sg(ct["table2"]["full"]["tr2"]["coef"]["tr2"], 3), "ctl_tr2_t": sg(ct["table2"]["full"]["tr2"]["t"]["tr2"], 2),
    "ctl_exp": sg(ct["table2"]["full"]["tr2"]["coef"]["expenses"], 3), "ctl_exp_t": sg(ct["table2"]["full"]["tr2"]["t"]["expenses"], 2), "ctl_to_t": sg(ct["table2"]["full"]["tr2"]["t"]["turnover"], 2), "m_months": str(mf["r2_alpha"]["months"]),
    "m_ranked": n(int(mf["leg_sizes"]["avg_ranked"])), "m_long": f"{mf['leg_sizes']['avg_long']:.0f}",
    "g_min": sg(min(v["cagr_pct"] for k, v in mf["grid"].items() if k.endswith("r2_alpha")), 1),
    "g_max": sg(max(v["cagr_pct"] for k, v in mf["grid"].items() if k.endswith("r2_alpha")), 1),
    "t_min": sg(min(v["r2_alpha"]["cagr_pct"] for v in mf["timing"].values()), 1), "t_max": sg(max(v["r2_alpha"]["cagr_pct"] for v in mf["timing"].values()), 1),
    "c0": sg(mf["costs"]["0"]["cagr_pct"], 1), "c10": sg(mf["costs"]["10"]["cagr_pct"], 1), "c25": sg(mf["costs"]["25"]["cagr_pct"], 1), "c50": sg(mf["costs"]["50"]["cagr_pct"], 1),
    "b1": sg(mf["borrow"]["1"]["cagr_pct"], 1), "b2": sg(mf["borrow"]["2"]["cagr_pct"], 1), "b5": sg(mf["borrow"]["5"]["cagr_pct"], 1),
    "fr_alpha": sg(fr["r2_alpha"]["alpha_pct_yr"]), "fr_t": sg(fr["r2_alpha"]["t"], 2),
    "fl_alpha": sg(fr["long_leg_minus_equal_weight"]["alpha_pct_yr"]), "fl_t": sg(fr["long_leg_minus_equal_weight"]["t"], 2),
    "sub_txt": "; ".join(f"{p}: long-short {sg(d['r2_alpha']['cagr_pct'], 1)}%, long leg {sg(d['long_leg']['cagr_pct'], 1)}%, all funds {sg(d['equal_weight']['cagr_pct'], 1)}%, SPY {sg(d['spy']['cagr_pct'], 1)}% a year" for p, d in sub_p.items()) + ".",
    "ctl_cover": f"Proxies were available for {ct['n_used'] / ct['n_total'] * 100:.0f}% of fund-years.",
    "ctl_n": n(ct["n_used"]),
    "ver_tol": "10<sup>&minus;{}</sup>".format(int(-__import__('math').ceil(__import__('math').log10(max(ver_diffs))))) if ver_diffs else "n/a",
    "ver_fund": ver_fund.group(1) if ver_fund else "n/a",
    "versions": (ROOT / "data/versions.txt").read_text().strip() if (ROOT / "data/versions.txt").exists() else "Python, pandas, numpy, statsmodels; R, data.table, sandwich",
    "retrieved": man.get("retrieved", "16 November 2026"), "gh_tree": GH_TREE,
}

# ---------- equations ----------
t_ = mi("t")
tm1 = row(t_, mo("−"), mn("1"))
F = lambda k, arg: fn(sub(mi("F"), k), arg)
EQ365 = eq("365", row(fn(sub("R", "j"), t_), mo("="), f"<msub>{mi('α')}{mi('j')}</msub>", mo("+"),
                      f"<munder>{mo('∑')}{row(mi('k'), mo('∈'), mo('{'), mi('MKT'), mo(','), mi('SMB'), mo(','), mi('HML'), mo(','), mi('MOM'), mo('}'))}</munder>",
                      mop("["), f"<msub>{mi('β')}{row(mi('j'), mo(','), mi('k'))}</msub>", F("k", t_), mo("+"),
                      f"<msub>{mi('γ')}{row(mi('j'), mo(','), mi('k'))}</msub>", F("k", tm1), mop("]"), mo("+"), fn(sub("ε", "j"), t_)),
           "R sub j of t equals alpha sub j plus the sum over the four factors of beta times the factor today plus gamma times the factor yesterday, plus epsilon sub j of t")
# the compact form of Eq. 366 to 369: both sums are over the days of the year
EQ366 = eq("366-369", row(sup(mi("R"), mn("2")), mo("="), mn("1"), mo("−"),
                           frac(row(f"<munder>{mo('∑')}{mi('t')}</munder>", sup(fn(sub("ε", "j"), t_), mn("2"))),
                                row(f"<munder>{mo('∑')}{mi('t')}</munder>", sup(row(mop("("), sub("R", "j") + row(mop("("), t_, mop(")")), mo("−"), f"<mover>{sub('R', 'j')}{mo('¯')}</mover>", mop(")")), mn("2")))),
                           ), "R squared equals one minus the sum of squared residuals over the sum of squared deviations of the excess return from its mean")
EQIR = eq("IR", row(sub("InfRatio", "j") if False else f"<msub>{mi('InfRatio')}{mi('j')}</msub>", mo("="),
                    frac(f"<msub>{mi('α')}{mi('j')}</msub>", f"<msub>{mi('RMSE')}{mi('j')}</msub>"), mo(","), f"<msub>{mi('RMSE')}{mi('j')}</msub>", mo("="),
                    sqrt(frac(row(sub("SS", "res")), row(mi("n"), mo("−"), mi("k"))))),
          "Information ratio equals alpha over RMSE, where RMSE is the square root of the residual sum of squares over n minus k")
r2 = sup(mi("R"), mn("2"))
EQTR2 = eq("TR", row(f"<msup>{mi('TR')}{mn('2')}</msup>", mo("="), mi("ln"), mop("("), frac(sqrt(r2), row(mn("1"), mo("−"), sqrt(r2))), mop(")")),
           "TR squared equals the natural log of the square root of R squared over one minus the square root of R squared")
EQPRED = eq("P", row(f"<msub>{mi('α')}{row(mi('j'), mo(','), mi('y'))}</msub>", mo("="), mi("a"), mo("+"), mi("b"),
                     f"<msubsup>{mi('TR')}{row(mi('j'), mo(','), mi('y'), mo('−'), mn('1'))}{mn('2')}</msubsup>", mo("+"), mi("c"),
                     f"<msub>{mi('α')}{row(mi('j'), mo(','), mi('y'), mo('−'), mn('1'))}</msub>", mo("+"),
                     f"<msub>{mi('D')}{mi('y')}</msub>", mo("+"), f"<msub>{mi('D')}{mi('s')}</msub>", mo("+"), f"<msub>{mi('u')}{row(mi('j'), mo(','), mi('y'))}</msub>"),
           "alpha of fund j in year y equals a plus b times TR squared in the previous year plus c times the previous year's alpha plus year dummies and style dummies plus an error")
V.update({"EQ365": EQ365, "EQ366": EQ366, "EQIR": EQIR, "EQTR2": EQTR2, "EQPRED": EQPRED})

# ---------- tables ----------
V["TABLE1"] = T.table1_html()
V["T8A"], V["T8B"] = T.sort_html("table8_alpha", 3, "Alpha"), T.sort_html("table8_ir", 3, "InfRatio")
V["T9A"], V["T9B"] = T.sort_html("table9_alpha", 3, "Alpha"), T.sort_html("table9_ir", 3, "InfRatio")
V["T2"], V["T3"], V["T5"], V["YEARLY"] = T.reg_html("table2"), T.reg_html("table3"), T.fm_html(), T.yearly_html()


def ctl_table():
    cols = [("full", "tr2"), ("full", "comp"), ("early", "tr2"), ("early", "comp"), ("late", "tr2"), ("late", "comp")]
    names = [("tr2", "TR&sup2;", 1), ("rmse_p", "RMSE (% a day)", 100), ("sdr_p", "SDR (% a day)", 100), ("alpha_lag", "Alpha (lag)", 1), ("expenses", "Expense ratio (%)", 1),
             ("ltna", "Log(TNA)", 1), ("ltna2", "Log(TNA)&sup2;", 1), ("turnover", "Turnover (%)", 1), ("age", "Age (years)", 1)]
    head = ("<tr><th rowspan='2'>Alpha regression with proxy controls</th><th colspan='2'>2012-2026</th><th colspan='2'>2012-2018</th><th colspan='2'>2019-2026</th></tr>"
            "<tr>" + "".join(f"<th>({i})</th>" for i in range(1, 7)) + "</tr>")
    body = ""
    for v, lab, sc in names:
        body += f"<tr><th>{lab}</th>"
        for per, sp in cols:
            p = ct["table2"][per][sp]
            body += (f"<td>{p['coef'][v] / sc:.3f}<span class='qs-t'>({p['t'][v]:.2f})</span></td>" if v in p["coef"] else "<td></td>")
        body += "</tr>"
    body += "<tr><th>Fund-years</th>" + "".join(f"<td>{ct['table2'][per][sp]['n_fund_years']:,}</td>" for per, sp in cols) + "</tr>"
    return f"<div class='qs-tw'><table class='qs-table qs-num'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


V["CTLTABLE"] = ctl_table()
rows_m = [("R&sup2; &times; alpha long-short (the book&rsquo;s rule)", "r2_alpha"), ("Long leg only", "long_leg"), ("Alpha-only long-short", "alpha_only"),
          ("R&sup2;-only long-short", "r2_only"), ("All funds, equal weight", "equal_weight"), ("SPY", "spy")]
tb = ["<div class='qs-tw'><table class='qs-table qs-num'><tr><th></th><th class='num'>CAGR</th><th class='num'>Volatility</th><th class='num'>Sharpe</th><th class='num'>Max drawdown</th><th class='num'>Monthly turnover</th></tr>"]
for lab, k in rows_m:
    d = mf[k]
    to = "0%" if k == "spy" else f"{mf['turnover_pct'][k]:.0f}%"
    tb.append(f"<tr><td>{lab}</td><td class='num'>{sg(d['cagr_pct'])}%</td><td class='num'>{d['ann_vol_pct']:.1f}%</td><td class='num'>{sg(d['sharpe'])}</td><td class='num'>{sg(d['max_drawdown_pct'], 1)}%</td><td class='num'>{to}</td></tr>")
tb.append("</table></div>")
V["MONTHLY"] = "".join(tb)

# ---------- code blocks ----------
py_src = (ROOT / "python/ag_replication.py").read_text(encoding="utf-8")
r_src = (ROOT / "r/ag_replication.R").read_text(encoding="utf-8")
defs = {nd.name: nd for nd in ast.parse(py_src).body if isinstance(nd, ast.FunctionDef)}
py_a, py_b = defs["year_fit"].lineno, defs["year_fit"].end_lineno
m1 = re.search(r"^year_fit <- function.*?^\}", r_src, re.S | re.M)
r_a, r_b = r_src[:m1.start()].count("\n") + 1, r_src[:m1.end()].count("\n") + 1
V["FULL_TABS"] = tabs([
    ("Python", code_box(hl_function(py_src, py_a, py_b, PythonLexer(), "Fund-year regression: Eq. 365 to 368"), "Python", "ag_replication.py")),
    ("R", code_box(hl_function(r_src, r_a, r_b, SLexer(), "Fund-year regression: Eq. 365 to 368"), "R", "ag_replication.R")),
])

# ---------- assemble ----------
for _k, _v in list(V.items()):
    if _v.startswith("<div class='qs-tw'>"):
        V[_k] = re.sub(r"(?<=[>(\[])-(?=\d)", "&minus;", _v)
tpl = (ROOT / "post_template.html").read_text(encoding="utf-8")
for k, v in V.items():
    tpl = tpl.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{(\w+)\}\}", tpl.replace("{{IMG}}", ""))
assert not left, f"unfilled placeholders: {left}"
(ROOT / "post.html").write_text(tpl.replace("{{IMG}}", WP_IMG), encoding="utf-8")
css = (THEME_ASSETS / "qs-article.css").read_text(encoding="utf-8")
js = (THEME_ASSETS / "qs-article.js").read_text(encoding="utf-8")
prev = tpl.replace("{{IMG}}", "charts/")
page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>preview</title>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap">'
        "<style>body{margin:0;font-family:Inter,sans-serif;background:#fff}.aib-post-content h2{font-family:Poppins,sans-serif;font-weight:700;font-size:24px;margin:48px 0 16px}"
        ".aib-post-content h3{font-weight:600;font-size:19px;margin:36px 0 14px}.aib-post-content p{margin:0 0 20px}.aib-post-content img{max-width:100%;border-radius:14px;margin:16px 0 32px}"
        ".aib-post-content ul,.aib-post-content ol{margin:0 0 20px;padding-left:24px}.aib-post-content li{margin-bottom:8px}.aib-post-content a{color:#2563EB}</style>"
        f"<style>{css}</style><div style=\"max-width:1200px;margin:0 auto;padding:24px\"><div class=\"aib-post-content\" style=\"font-size:16px;line-height:1.75;color:#334155\">{prev}</div></div><script>{js}</script>")
(ROOT / "preview.html").write_text(page, encoding="utf-8")
print("built post.html", len(tpl), "chars")
