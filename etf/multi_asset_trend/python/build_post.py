"""Builds post.html (WordPress content) and preview.html for the 4.6 multi-asset trend-following article from post_template.html.
Every number comes from data/results_python.json. Run after trend.py, make_charts.py and verify.py."""
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
GH_TREE = "https://github.com/msmith01/quantstrat-strategies/tree/main/etf/multi_asset_trend"
O = json.load(open(ROOT / "data/results_python.json"))
G, B, C = O["grid"], O["benchmarks"], O["cap35"]
MINUS = "&minus;"


def sg(x, d=2):
    return (MINUS if round(x, d) < 0 else "") + f"{abs(x):.{d}f}"


def tbl(head, rows, cls="qs-table qs-num"):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    b = "".join("<tr>" + f"<th>{r[0]}</th>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows)
    return f"<div class='qs-tw'><table class='{cls}'><thead>{h}</thead><tbody>{b}</tbody></table></div>"


M = pd.read_csv(ROOT / "data/monthly_returns.csv", index_col=0, parse_dates=True)
g = lambda T, ma, wt: G[f"T{T}|ma{ma}|{wt}"]     # noqa: E731
m371, m372, m373 = g(12, 1, "371"), g(12, 1, "372"), g(12, 1, "373")
ew, spy, b6040 = B["Equal weight, all ten"], B["SPY buy and hold"], B["60/40 SPY and IEF, monthly"]
mains = (m371, m372, m373)
n12_worse = sum(g(12, ma, wt)["cagr"] < g(6, ma, wt)["cagr"] for ma in (0, 1) for wt in ("371", "372", "373"))
last = pd.Timestamp(spy["end"])
V = {
    "start_year": spy["start"][:4], "last_date": last.strftime("%d %B %Y").lstrip("0"), "start_month": pd.Timestamp(O["start"]).strftime("%B %Y"),
    "first_signal": (pd.Timestamp(O["start"]) - pd.offsets.MonthEnd(1)).strftime("%B %Y"), "n_months": str(m371["months"]),
    "r371": sg(m371["cagr"]), "r372": sg(m372["cagr"]), "r373": sg(m373["cagr"]), "spy": sg(spy["cagr"]), "ew": sg(ew["cagr"]),
    "dd_lo": sg(max(m["maxdd"] for m in mains), 1), "dd_hi": sg(min(m["maxdd"] for m in mains), 1), "dd_spy": sg(spy["maxdd"], 1), "dd_ew": sg(ew["maxdd"], 1),
    "sh371": sg(m371["sharpe"]), "sh373": sg(m373["sharpe"]), "sh_ew": sg(ew["sharpe"]), "sh6040": sg(b6040["sharpe"]),
    "sh_spread": sg(max(m["sharpe"] for m in mains) - min(m["sharpe"] for m in mains)),
    "v371": sg(m371["vol"], 1), "v373": sg(m373["vol"], 1),
    "gap_lo": sg(spy["cagr"] - max(m["cagr"] for m in mains), 1), "gap_hi": sg(spy["cagr"] - min(m["cagr"] for m in mains), 1),
    "beat371": sg(m371["cagr"] - ew["cagr"], 1), "turn371": f"{m371['turnover']:.0f}", "cost_gap": sg(m371["gross_cagr"] - m371["cagr"], 1),
    "held": f"{m371['avg_held']:.1f}", "cash_pct": f"{m371['cash_months_pct']:.0f}",
    "g_lo": sg(min(v["cagr"] for v in G.values()), 1), "g_hi": sg(max(v["cagr"] for v in G.values()), 1),
    "gdd_lo": sg(max(v["maxdd"] for v in G.values()), 1), "gdd_hi": sg(min(v["maxdd"] for v in G.values()), 1),
    "n_beat": str(sum(v["sharpe"] > ew["sharpe"] for v in G.values())), "n12_worse": str(n12_worse),
    "cap371": sg(C["371"]["cagr"], 1), "cap372": sg(C["372"]["cagr"], 1), "cap373": sg(C["373"]["cagr"], 1),
    "capsh371": sg(C["371"]["sharpe"]), "capsh373": sg(C["373"]["sharpe"]), "capdd371": sg(C["371"]["maxdd"], 1), "capdd373": sg(C["373"]["maxdd"], 1),
    "c0": sg(O["costs_372"]["0"], 1), "c10": sg(O["costs_372"]["10"], 1), "c25": sg(O["costs_372"]["25"], 1), "c50": sg(O["costs_372"]["50"], 1),
    "gh_tree": GH_TREE,
    "cap_cash": f"{C['371']['avg_cash_pct']:.0f}", "cap_cash_any": f"{C['371']['cash_any_pct']:.0f}", "unc_cash": f"{C['371']['uncapped_avg_cash_pct']:.0f}", "unc_cash_any": f"{C['371']['uncapped_cash_any_pct']:.0f}",
    "nd371": sg(O["next_day"]["371"]["cagr"], 1), "nd372": sg(O["next_day"]["372"]["cagr"], 1), "nd373": sg(O["next_day"]["373"]["cagr"], 1),
    "ndsh371": sg(O["next_day"]["371"]["sharpe"]), "ndsh373": sg(O["next_day"]["373"]["sharpe"]),
    "ndgap": sg(max(m["cagr"] - O["next_day"][k]["cagr"] for k, m in zip(("371", "372", "373"), mains)), 1),
    "shrf371": sg(m371["sharpe_rf"]), "shrf373": sg(m373["sharpe_rf"]), "shrf_ew": sg(ew["sharpe_rf"]), "shrf6040": sg(b6040["sharpe_rf"]), "shrf_spy": sg(spy["sharpe_rf"]),
}
assert float(V["capsh371"]) > 0 and "dd_lo" in V

rows = []
for lab, st, tn in (("Weights by return, Eq. (371)", m371, m371["turnover"]), ("Return / volatility, Eq. (372)", m372, m372["turnover"]), ("Return / variance, Eq. (373)", m373, m373["turnover"]),
                    ("Equal weight, all ten ETFs", ew, None), ("S&amp;P 500 ETF (SPY)", spy, None), ("60/40 SPY and IEF", b6040, None), ("Treasury bill ETF (SHV)", B["Treasury bill ETF"], None)):
    rows.append([lab, sg(st["cagr"], 1), sg(st["vol"], 1), sg(st["sharpe"]), "" if lab.startswith("Treasury") else sg(st["sharpe_rf"]), sg(st["maxdd"], 1), "" if tn is None else f"{tn:.0f}"])
V["T_MAIN"] = tbl(["Portfolio", "Return, % a year", "Volatility, %", "Sharpe ratio (zero rate)", "Sharpe ratio (over SHV)", "Max drawdown, %", "Monthly turnover, %"], rows)
rows = []
names = {"371": "Eq. (371)", "372": "Eq. (372)", "373": "Eq. (373)"}
for T in (6, 9, 12):
    for ma in (0, 1):
        for wt in ("371", "372", "373"):
            s = g(T, ma, wt)
            rows.append([f"{T} months, {'with' if ma else 'no'} 200-day filter, {names[wt]}", sg(s["cagr"], 1), sg(s["vol"], 1), sg(s["sharpe"]), sg(s["maxdd"], 1), f"{s['turnover']:.0f}", f"{s['avg_held']:.1f}", f"{s['cash_months_pct']:.0f}"])
V["T_GRID"] = tbl(["Setting", "Return, % a year", "Volatility, %", "Sharpe ratio", "Max drawdown, %", "Turnover, %", "Held", "Cash, % of months"], rows)

# equations: weights proportional to R, R/sigma, R/sigma^2 (the paper's Eq. 371-373)
Rr, Sg = sub("R", "i"), sub("σ", "i")
w_ = sub("w", "i")
def weq(n, rhs, label):
    return eq(n, row(w_, mo("∝"), rhs), label)
V["EQWEIGHTS"] = (weq("371", Rr, "weight proportional to the cumulative return")
                  + weq("372", frac(Rr, Sg), "weight proportional to the cumulative return divided by volatility")
                  + weq("373", frac(Rr, sup(Sg, mn("2"))), "weight proportional to the cumulative return divided by variance"))

py_src = (ROOT / "python/trend.py").read_text(encoding="utf-8")
r_src = (ROOT / "r/trend.R").read_text(encoding="utf-8")
defs = {nd.name: nd for nd in ast.parse(py_src).body if isinstance(nd, ast.FunctionDef)}
py_a, py_b = defs["weights"].lineno, defs["run"].end_lineno
m1 = re.search(r"^run <- function.*?^\}", r_src, re.S | re.M)
r_a, r_b = r_src[:m1.start()].count("\n") + 1, r_src[:m1.end()].count("\n") + 1
V["FULL_TABS"] = tabs([
    ("Python", code_box(hl_function(py_src, py_a, py_b, PythonLexer(), "Weights and monthly backtest"), "Python", "trend.py")),
    ("R", code_box(hl_function(r_src, r_a, r_b, SLexer(), "Weights and monthly backtest"), "R", "trend.R")),
])
tplsrc = (RS / "post_template.html").read_text(encoding="utf-8")
V["STYLE"] = tplsrc[tplsrc.index("<style>"):tplsrc.index("</style>") + 8]

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
for k in ("r371", "r372", "r373", "spy", "ew", "dd_lo", "dd_hi", "dd_spy", "sh371", "sh373", "sh_ew", "sh6040", "n_beat", "n12_worse", "gap_lo", "gap_hi", "beat371", "cost_gap"):
    print(k, V[k])
