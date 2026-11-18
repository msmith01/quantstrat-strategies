"""Builds post.html (WordPress content) and preview.html for the 4.4 IBS article from post_template.html.
Every number comes from data/results_python.json. Run after ibs.py, make_charts.py and verify.py."""
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
from postlib import code_box, eq, fn, frac, hl_function, mi, mn, mo, mop, row, sub, sup, sqrt, tabs   # noqa: E402

THEME_ASSETS = ROOT.parents[2] / "wp_theme_build" / "theme" / "genebook" / "assets"
WP_IMG = (ROOT / "wp_img_prefix.txt").read_text().strip() if (ROOT / "wp_img_prefix.txt").exists() else "charts/"
GH_TREE = "https://github.com/msmith01/quantstrat-strategies/tree/main/etf/mean_reversion"
O = json.load(open(ROOT / "data/results_python.json"))
P = O["params"]
PW, AF = "paper_window", "after"
MINUS = "&minus;"
PAPER_B = [0.350, 0.096, 0.025, -0.032, -0.126]
PAPER_UP = [56.76, 52.74, 51.34, 49.92, 46.22]


def sg(x, d=2):
    return (MINUS if round(x, d) < 0 else "") + f"{abs(x):.{d}f}"


def tbl(head, rows, cls="qs-table qs-num"):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    b = "".join("<tr>" + f"<th>{r[0]}</th>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows)
    return f"<div class='qs-tw'><table class='{cls}'><thead>{h}</thead><tbody>{b}</tbody></table></div>"


# ---------- derived ----------
t3a, t4a = O["table3_avg"], O["table4_avg"]
dec, cost, gross, rsi, cond = O["decile_rule"], O["ibs_strategy_costs"], O["ibs_strategy_gross"], O["rsi_filter"], O["conditional"]
last = pd.Timestamp(P["last_date"])
after_years = (last - pd.Timestamp("2013-12-06")).days / 365.25
mk = lambda per: gross[per]["avg_in_market_pct"]
C = lambda per, reg: cond[per]  # noqa: E731
d = lambda per, k, lvl: cond[per][k][lvl]["diff"]   # noqa: E731
be = dec[AF]["0"]["mean_bp_day"] / dec[AF]["avg_daily_turnover"]
V = {
    "n_etfs": str(P["n_tickers"]), "after_years": f"{after_years:.0f}", "last_date": last.strftime("%-d %B %Y") if sys.platform != "win32" else last.strftime("%d %B %Y").lstrip("0"),
    "n_noibs": f"{P['rows_without_ibs_high_eq_low'] + P['rows_ibs_outside_0_1_set_missing']:,}",
    "pw_days": f"{sum(v['n'] for v in O['table2'][PW].values()) / len(O['table2'][PW]):,.0f}",
    "pw_b1": sg(t3a[PW]["mean"][0], 3), "pw_b5": sg(t3a[PW]["mean"][4], 3), "pw_diff": f"{t3a[PW]['diff_1_5']:.2f}", "af_diff": f"{t3a[AF]['diff_1_5']:.2f}",
    "pw_cagr": f"{gross[PW]['avg_cagr']:.1f}", "af_cagr": f"{gross[AF]['avg_cagr']:.1f}", "af_cagr5": sg(cost[AF]["5"]["avg_cagr"], 1),
    "pw_pos": str(t3a[PW]["n_pos_diff"]), "pw_sig": str(t3a[PW]["n_sig_diff"]), "af_pos": str(t3a[AF]["n_pos_diff"]), "af_sig": str(t3a[AF]["n_sig_diff"]),
    "af_up5": f"{t4a[AF]['up'][4]:.1f}", "af_upgap": f"{t4a[AF]['diff_1_5']:.1f}", "pw_upgap": f"{t4a[PW]['diff_1_5']:.1f}",
    "pw_npos": str(gross[PW]["n_pos"]), "pw_mkt": f"{mk(PW):.0f}", "af_mkt": f"{mk(AF):.0f}",
    "vh_pw": sg(d(PW, "volatility", "high")), "vl_pw": sg(d(PW, "volatility", "low")), "tb_pw": sg(d(PW, "trend", "bear")), "tu_pw": sg(d(PW, "trend", "bull")),
    "vh_af": sg(d(AF, "volatility", "high")), "vl_af": sg(d(AF, "volatility", "low")), "tb_af": sg(d(AF, "trend", "bear")), "tu_af": sg(d(AF, "trend", "bull")),
    "mon_pw": sg(d(PW, "weekday", "Mon")), "mon_af": sg(d(AF, "weekday", "Mon")), "fri_pw": sg(d(PW, "weekday", "Fri")), "thu_af": sg(d(AF, "weekday", "Thu")),
    "sk1_pw": sg(O["skew_by_bucket"][PW][0]), "sk1_af": sg(O["skew_by_bucket"][AF][0]),
    "rsi_pw_s": f"{rsi[PW]['avg_sum_rsi']:.1f}", "rsi_pw_sf": f"{rsi[PW]['avg_sum_filtered']:.1f}", "rsi_af_s": f"{rsi[AF]['avg_sum_rsi']:.1f}", "rsi_af_sf": f"{rsi[AF]['avg_sum_filtered']:.1f}",
    "rsi_pw_bsum": str(rsi[PW]["n_better_sum"]), "rsi_af_bsum": str(rsi[AF]["n_better_sum"]), "dr_pw": f"{rsi[PW]['avg_days_rsi']:.0f}", "dr_pw_f": f"{rsi[PW]['avg_days_filtered']:.0f}",
    "dec_borrow": f"{dec[AF]['2_drift']['cagr'] - dec[AF]['2_drift_borrow']['cagr']:.1f}", "dec_af2b": f"{dec[AF]['2_drift_borrow']['cagr']:.1f}",
    "rsi_pw": f"{rsi[PW]['avg_total_rsi']:.0f}", "rsi_pw_f": f"{rsi[PW]['avg_total_filtered']:.0f}", "rsi_pw_better": str(rsi[PW]["n_better"]),
    "rsi_af": f"{rsi[AF]['avg_total_rsi']:.0f}", "rsi_af_f": f"{rsi[AF]['avg_total_filtered']:.0f}", "rsi_af_better": str(rsi[AF]["n_better"]),
    "dec_n": f"{dec[AF]['avg_etfs']:.0f}", "dec_k": f"{dec[AF]['avg_per_leg']:.0f}", "dec_turn": f"{dec[AF]['avg_daily_turnover']:.1f}",
    "dec_af0": f"{dec[AF]['0']['cagr']:.1f}", "dec_af2": f"{dec[AF]['2']['cagr']:.1f}", "dec_af5": sg(dec[AF]["5"]["cagr"], 1),
    "dec_sh_pw": f"{dec[PW]['0']['sharpe']:.1f}", "dec_sh_af": f"{dec[AF]['0']['sharpe']:.1f}", "dec_long_af": f"{dec[AF]['long_leg_cagr']:.1f}",
    "dec_short_af": sg(dec[AF]["short_leg_cagr"], 1), "dec_dd5": sg(dec[AF]["5"]["max_dd"], 0), "dec_be": f"{be:.1f}", "gh_tree": GH_TREE,
}

# ---------- tables ----------
bl = ["1st (0&ndash;0.2)", "2nd", "3rd", "4th", "5th (0.8&ndash;1)", "1st &minus; 5th"]
mkrow = lambda lab, v: [lab] + [sg(x, 3) for x in v] + [sg(v[0] - v[4], 3)]   # noqa: E731
V["T_BUCKETS"] = tbl([""] + bl, [mkrow("Published average (to 5 Dec 2013)", PAPER_B), mkrow("Ours, to 5 Dec 2013", t3a[PW]["mean"]), mkrow(f"Ours, after 5 Dec 2013", t3a[AF]["mean"])])
mku = lambda lab, v: [lab] + [f"{x:.1f}" for x in v] + [f"{v[0] - v[4]:.1f}"]   # noqa: E731
V["T_UP"] = tbl([""] + bl, [mku("Published average", PAPER_UP), mku("Ours, to 5 Dec 2013", t4a[PW]["up"]), mku("Ours, after 5 Dec 2013", t4a[AF]["up"])])
# per ETF
rows = []
for t in P["tickers"]:
    r = [t, P["first_dates"][t][:4]]
    for per in (PW, AF):
        x = O["table3"][per].get(t)
        r += [sg(x["mean"][0], 2), sg(x["mean"][4], 2)] if x else ["&ndash;", "&ndash;"]
        r += [sg(gross[per]["by_etf"][t], 1)] if t in gross[per]["by_etf"] else ["&ndash;"]
    rows.append(r)
V["T_ETF"] = tbl(["ETF", "Starts", "To 2013: 1st", "5th", "Rule %/yr", "After: 1st", "5th", "Rule %/yr"], rows)
# conditional
labs = [("volatility", "high", "High volatility"), ("volatility", "low", "Low volatility"), ("range", "high", "High range"), ("range", "low", "Low range"),
        ("trend", "bear", "Bear market (200-day return below 0)"), ("trend", "bull", "Bull market"), ("volume", "high", "High volume"), ("volume", "low", "Low volume"),
        ("weekday", "Mon", "After Monday"), ("weekday", "Tue", "After Tuesday"), ("weekday", "Wed", "After Wednesday"), ("weekday", "Thu", "After Thursday"), ("weekday", "Fri", "After Friday")]
V["T_COND"] = tbl(["Regime on day t", "To 5 Dec 2013", "After 5 Dec 2013"], [[lab, sg(cond[PW][k][l]["diff"], 2), sg(cond[AF][k][l]["diff"], 2)] for k, l, lab in labs])
# strategy
rows = [[f"{c} bps per unit traded" if c != "0" else "Before costs", sg(cost[PW][c]["avg_cagr"], 1), f"{cost[PW][c]['n_pos']} of {cost[PW][c]['n']}", sg(cost[AF][c]["avg_cagr"], 1), f"{cost[AF][c]['n_pos']} of {cost[AF][c]['n']}"] for c in ("0", "2", "5", "10")]
V["T_STRAT"] = tbl(["Average return per ETF, % a year", "To 5 Dec 2013", "ETFs positive", "After 5 Dec 2013", "ETFs positive"], rows)
rr = lambda p, nm: (f"{rsi[p]['avg_sum_' + nm]:.1f}", f"{rsi[p]['avg_total_' + nm]:.0f}", f"{rsi[p]['avg_days_' + nm]:.0f}")
V["T_RSI"] = tbl(["Cutler&rsquo;s RSI(3) rule", "Sum of daily returns, %", "Compounded, %", "Days invested", "Sum of daily returns, %", "Compounded, %", "Days invested"],
                 [["Without IBS filter", *rr(PW, "rsi"), *rr(AF, "rsi")], ["With IBS &le; 0.5 filter", *rr(PW, "filtered"), *rr(AF, "filtered")]])
V["T_RSI"] = V["T_RSI"].replace("<tr><th>Cutler", "<tr><th>Cutler", 1)
# decile
rows = []
for c in ("0", "2", "5", "10"):
    rows.append([("Before costs" if c == "0" else f"{c} bps"), sg(dec[PW][c]["cagr"], 1), f"{dec[PW][c]['sharpe']:.2f}", sg(dec[AF][c]["cagr"], 1), f"{dec[AF][c]['sharpe']:.2f}", sg(dec[AF][c]["max_dd"], 1)])
rows.append(["Equal-weight all ETFs (reference)", f"{dec[PW]['equal_weight_cagr']:.1f}", "", f"{dec[AF]['equal_weight_cagr']:.1f}", "", ""])
V["T_DEC"] = tbl(["Decile rule", "CAGR % to 5 Dec 2013", "Sharpe", "CAGR % after", "Sharpe", "Max drawdown % after"], rows)

# ---------- equations ----------
t_ = mi("t")
EQ370 = eq("370", row(mi("IBS"), mo("="), frac(row(sub("P", "C"), mo("−"), sub("P", "L")), row(sub("P", "H"), mo("−"), sub("P", "L")))),
           "IBS equals the previous close minus the previous low, over the previous high minus the previous low")
EQRET = eq("R", row(sub("r", row(t_, mo("+"), mn("1"))), mo("="), frac(sub("P", row(t_, mo("+"), mn("1"))), sub("P", t_)), mo("−"), mn("1")),
           "next-day return equals the next close over today's close minus one")
EQSTRAT = eq("S", row(sub("r", row(mi("s"), mo(","), t_, mo("+"), mn("1"))), mo("="), sub("w", t_), sub("r", row(t_, mo("+"), mn("1"))), mo("−"), mi("c"), mo("|"), sub("w", t_), mo("−"), sub("w", row(t_, mo("−"), mn("1"))), mo("|")),
              "strategy return equals the position times the next-day return minus the cost times the change in position")
EQDEC = eq("D", row(sub("r", row(mi("L"), mo("−"), mi("S"), mo(","), t_, mo("+"), mn("1"))), mo("="), frac(mn("1"), mi("k")), f"<munder>{mo('∑')}{row(mi('i'), mo('∈'), mi('Low'))}</munder>", sub("r", row(mi("i"), mo(","), t_, mo("+"), mn("1"))),
                    mo("−"), frac(mn("1"), mi("k")), f"<munder>{mo('∑')}{row(mi('i'), mo('∈'), mi('High'))}</munder>", sub("r", row(mi("i"), mo(","), t_, mo("+"), mn("1"))), mo("−"), mi("c"), sub(mi("TO"), t_)),
           "decile rule return equals the average next-day return of the k lowest IBS ETFs minus that of the k highest, minus cost times turnover")
V.update({"EQ370": EQ370, "EQRET": EQRET, "EQSTRAT": EQSTRAT, "EQDEC": EQDEC})

# ---------- code ----------
py_src = (ROOT / "python/ibs.py").read_text(encoding="utf-8")
r_src = (ROOT / "r/ibs.R").read_text(encoding="utf-8")
defs = {nd.name: nd for nd in ast.parse(py_src).body if isinstance(nd, ast.FunctionDef)}
py_a, py_b = defs["strat_returns"].lineno, defs["strat_returns"].end_lineno
m1 = re.search(r"^strat <- function.*?^\}", r_src, re.S | re.M)
r_a, r_b = r_src[:m1.start()].count("\n") + 1, r_src[:m1.end()].count("\n") + 1
V["FULL_TABS"] = tabs([
    ("Python", code_box(hl_function(py_src, py_a, py_b, PythonLexer(), "Single-ETF strategy return"), "Python", "ibs.py")),
    ("R", code_box(hl_function(r_src, r_a, r_b, SLexer(), "Single-ETF strategy return"), "R", "ibs.R")),
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
