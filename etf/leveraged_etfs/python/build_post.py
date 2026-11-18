"""Builds post.html (WordPress content) and preview.html for the 4.5 leveraged-ETF article from post_template.html.
Every number comes from data/results_python.json. Run after letf.py, make_charts.py and verify.py."""
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
from pygments.lexers import PythonLexer, SLexer

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT.parent / "r_squared"
sys.path.insert(0, str(RS / "python"))
from postlib import code_box, eq, frac, hl_function, mi, mn, mo, row, sub, sup, tabs   # noqa: E402

THEME_ASSETS = ROOT.parents[2] / "wp_theme_build" / "theme" / "genebook" / "assets"
WP_IMG = (ROOT / "wp_img_prefix.txt").read_text().strip() if (ROOT / "wp_img_prefix.txt").exists() else "charts/"
GH_TREE = "https://github.com/msmith01/quantstrat-strategies/tree/main/etf/leveraged_etfs"
O = json.load(open(ROOT / "data/results_python.json"))
P, S, DR = O["params"], O["strategy"], O["decay_regression"]
SA, PF, PM = O["strategy_avg"], O["portfolio"], O["portfolio_monthly"]
MINUS = "&minus;"


def sg(x, d=2):
    return (MINUS if round(x, d) < 0 else "") + f"{abs(x):.{d}f}"


def tbl(head, rows, cls="qs-table qs-num"):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    b = "".join("<tr>" + f"<th>{r[0]}</th>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows)
    return f"<div class='qs-tw'><table class='{cls}'><thead>{h}</thead><tbody>{b}</tbody></table></div>"


last = pd.Timestamp(P["last_date"])
pairs = P["pairs"]
SC, PS, CP = O["strategy"], O["portfolio_scen"], O["portfolio_common"]
FULLR = DR["full"]
npos = lambda fk: sum(S[n][fk]["cagr"] > 0 for n in S)                         # noqa: E731
nbeat = lambda fk: sum(S[n][fk]["cagr"] > S[n]["cash_only"]["cagr"] for n in S)  # noqa: E731
worst_dd = min(S, key=lambda n: S[n]["monthly|0"]["max_dd"])
V = {
    "n_pairs": str(len(pairs)), "n_etfs": str(2 * len(pairs)), "n_etfyears": str(FULLR["n"]), "last_date": last.strftime("%d %B %Y").lstrip("0"),
    "first_start": min(v["start"] for v in pairs.values())[:4],
    "slope": sg(FULLR["slope"]), "slope_se": f"{FULLR['slope_se_cluster_year']:.2f}", "r2": f"{FULLR['r2']:.2f}", "icpt": sg(FULLR["intercept"] * 100),
    "slope2": sg(DR["2x"]["slope"]), "slope3": sg(DR["3x"]["slope"]), "slopeL": sg(DR["long"]["slope"]), "slopeI": sg(DR["inverse"]["slope"]),
    "slope_all": sg(DR["all"]["slope"]), "n_all": str(DR["all"]["n"]),
    "d_daily0": sg(PF["0"]["cagr"], 1), "d_daily3": sg(PF["3"]["cagr"], 1), "cash": sg(PF["cash"]["cagr"], 1), "d_excess": f"{PF['0']['cagr'] - PF['cash']['cagr']:.1f}",
    "m0": sg(PM["0"]["cagr"], 1), "m1": sg(PM["1"]["cagr"], 1), "m3": sg(PM["3"]["cagr"], 1), "m5": sg(PM["5"]["cagr"], 1),
    "m_vol": f"{PM['0']['vol']:.1f}", "m_dd": sg(PM["0"]["max_dd"], 1), "m_wd": sg(PM["0"]["worst_day"], 1),
    "pos_m0": str(npos("monthly|0")), "beat_m3": str(nbeat("monthly|3")), "pos_m5": str(npos("monthly|5")), "wiped_a": str(O["strategy_counts"]["n_wiped_out_annual_0"]),
    "worst_pair": worst_dd, "worst_pair_dd": f"{abs(S[worst_dd]['monthly|0']['max_dd']):.0f}",
    "stress_gfc": sg(O["stress_monthly"]["2008-09"]["0"]["ret"], 1), "stress_cov": sg(O["stress_monthly"]["Covid crash 2020"]["0"]["ret"], 1),
    "stress_cov_dd": f"{abs(O['stress_monthly']['Covid crash 2020']['0']['max_dd']):.1f}",
    "ni0": sg(PS["no_interest|0"]["cagr"], 1), "tc0": sg(PS["trade_cost|0"]["cagr"], 1), "mg0": sg(PS["margin|0"]["cagr"], 1), "mg3": sg(PS["margin|3"]["cagr"], 1),
    "all0": sg(PS["all|0"]["cagr"], 1), "all3": sg(PS["all|3"]["cagr"], 1), "liq0": str(PS["margin|0"]["liquidations"]),
    "cp0": sg(CP["0"]["cagr"], 1), "cp3": sg(CP["3"]["cagr"], 1), "cp_cash": sg(CP["cash"]["cagr"], 1), "cp_start": CP["start"][:4], "gh_tree": GH_TREE,
}
b0, b5 = PM["0"]["cagr"], PM["5"]["cagr"]
V["be_borrow"] = f"{(b0 - PF['cash']['cagr']) / (b0 - b5) * 5:.1f}"

# ---------- tables ----------
rows = []
for lab, st in (("Reset daily, no borrow fee", PF["0"]), ("Reset monthly, no borrow fee", PM["0"]), ("Reset monthly, 3% borrow fee", PM["3"]), ("Reset monthly, 5% borrow fee", PM["5"]), ("Treasury bill ETF alone", PF["cash"])):
    rows.append([lab, sg(st["cagr"], 1), sg(st["vol"], 1), sg(st["max_dd"], 1), sg(st["worst_day"], 1)])
V["T_PORT"] = tbl(["All pairs together", "Return, % a year", "Volatility, %", "Max drawdown, %", "Worst day, %"], rows)
rows = [[lab, sg(PS[k + "|0"]["cagr"], 1), sg(PS[k + "|3"]["cagr"], 1), sg(PS[k + "|0"]["max_dd"], 1), str(PS[k + "|0"]["liquidations"])] for lab, k in
        (("Idealised model", "base"), ("No interest on the short-sale proceeds", "no_interest"), ("5 bps on each scheduled reset of the shorts", "trade_cost"), ("Margin calls (60% or 90% maintenance)", "margin"), ("All three together", "all"))]
V["T_SCEN"] = tbl(["Monthly reset, all pairs", "Return, % a year, no borrow fee", "Return with 3% borrow fee", "Max drawdown, %", "Forced liquidations"], rows)
rows = []
for n in S:
    rows.append([n, f"{pairs[n]['long']} / {pairs[n]['inverse']}", pairs[n]["start"][:4], sg(S[n]["daily|0"]["cagr"], 1), sg(S[n]["monthly|0"]["cagr"], 1), sg(S[n]["monthly|3"]["cagr"], 1),
                 sg(S[n]["monthly|0"]["max_dd"], 0), sg(S[n]["annual|0"]["cagr"], 1), sg(S[n]["cash_only"]["cagr"], 1)])
V["T_PAIRS"] = tbl(["Pair", "ETFs (long / inverse)", "From", "Daily reset, %", "Monthly reset, %", "Monthly, 3% fee, %", "Monthly: max drawdown, %", "Annual reset, %", "T-bill ETF, %"], rows)
rows = []
for lab, k in (("Calendar years with at least 120 trading days", "all"), ("Full calendar years (240 days or more)", "full"), ("Full years: leveraged long ETFs", "long"), ("Full years: inverse ETFs", "inverse")):
    d = DR[k]
    rows.append([lab, str(d["n"]), sg(d["slope"], 2) + f" ({d['slope_se']:.2f})", f"{d['slope_se_cluster_year']:.2f}", f"{d['r2']:.2f}"])
V["T_DECAY"] = tbl(["Sample", "ETF-years", "Slope (standard error)", "Standard error, clustered by year", "R-squared"], rows)

# ---------- equations ----------
t_ = mi("t")
LEV = mi("L")
EQDECAY = eq("1", row(mi("ln"), mo("("), frac(sub("V", mi("T")), sub("V", mn("0"))), mo(")"), mo("≈"), LEV, mi("ln"), mo("("), frac(sub("S", mi("T")), sub("S", mn("0"))), mo(")"), mo("−"),
                      frac(row(LEV, mo("("), LEV, mo("−"), mn("1"), mo(")")), mn("2")), f"<munder>{mo('∑')}{t_}</munder>", sup(sub("r", row(mi("u"), mo(","), t_)), mn("2"))),
             "the log return of a daily-reset L times ETF is L times the log return of the index minus L times L minus one over two times the sum of squared daily index returns")
EQSTRAT = eq("2", row(sub("r", row(mi("s"), mo(","), t_)), mo("="), sub("r", row(mi("c"), mo(","), t_)), mo("−"), frac(mn("1"), mn("2")), sub("r", row(mi("p"), mo(","), t_)), mo("−"), frac(mn("1"), mn("2")), sub("r", row(mi("m"), mo(","), t_)), mo("−"), mi("f")),
                "strategy return equals the Treasury ETF return minus half the leveraged ETF return minus half the inverse ETF return minus the borrow fee")
V.update({"EQDECAY": EQDECAY, "EQSTRAT": EQSTRAT})

# ---------- code ----------
py_src = (ROOT / "python/letf.py").read_text(encoding="utf-8")
r_src = (ROOT / "r/letf.R").read_text(encoding="utf-8")
defs = {nd.name: nd for nd in ast.parse(py_src).body if isinstance(nd, ast.FunctionDef)}
py_a, py_b = defs["sim"].lineno, defs["sim"].end_lineno
m1 = re.search(r"^sim <- function.*?^\}", r_src, re.S | re.M)
r_a, r_b = r_src[:m1.start()].count("\n") + 1, r_src[:m1.end()].count("\n") + 1
V["FULL_TABS"] = tabs([
    ("Python", code_box(hl_function(py_src, py_a, py_b, PythonLexer(), "Strategy simulation"), "Python", "letf.py")),
    ("R", code_box(hl_function(r_src, r_a, r_b, SLexer(), "Strategy simulation"), "R", "letf.R")),
])
tplsrc = (RS / "post_template.html").read_text(encoding="utf-8")
V["STYLE"] = tplsrc[tplsrc.index("<style>"):tplsrc.index("</style>") + 8]

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
