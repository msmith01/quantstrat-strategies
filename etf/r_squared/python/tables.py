"""Formats the Amihud-Goyenko style tables (HTML for quantstr.at, Markdown for DEV.to) from data/results_ag_python.json.
Layout follows the paper: 5x5 grids with the average in the cell and the t-statistic in brackets underneath."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "data/results_ag_python.json"))
ROWS = {1: "Low", 2: "2", 3: "3", 4: "4", 5: "High"}
COLS = [1, 2, 3, 4, 5]


def f(x, d):
    return f"{x:.{d}f}"


def grid(key, d):
    """Rows: past-performance quintile (Low..High), All, High-Low. Columns: R2 quintile Low..High, All, Low-High."""
    t = RES[key]
    out = []
    for a in range(1, 6):
        row = [(ROWS[a], None)]
        for q in COLS:
            m, tt, _ = t["cell"][f"{a}_{q}"]
            row.append((f(m, d), f(tt, 2)))
        m, tt = t["margin_alpha"][str(a)]
        row.append((f(m, d), f(tt, 2)))
        m, tt = t["low_minus_high_r2"][str(a)]
        row.append((f(m, d), f(tt, 2)))
        out.append(row)
    row = [("All", None)]
    for q in COLS:
        m, tt = t["margin_r2"][str(q)]
        row.append((f(m, d), f(tt, 2)))
    m, tt = t["all"]
    row.append((f(m, d), f(tt, 2)))
    m, tt = t["low_minus_high_r2"]["All"]
    row.append((f(m, d), f(tt, 2)))
    out.append(row)
    row = [("High - Low", None)]
    for q in COLS:
        m, tt = t["high_minus_low_alpha"][str(q)]
        row.append((f(m, d), f(tt, 2)))
    m, tt = t["high_minus_low_alpha"]["All"]
    row.append((f(m, d), f(tt, 2)))
    row.append(("", None))
    out.append(row)
    return out


def sort_html(key, d, perf_label):
    head = ("<tr><th rowspan='2'>" + perf_label + "<sub>y-1</sub></th><th colspan='5'>R&sup2;<sub>y-1</sub> quintile</th><th rowspan='2'>All</th><th rowspan='2'>Low &minus; High</th></tr>"
            "<tr><th>Low</th><th>2</th><th>3</th><th>4</th><th>High</th></tr>")
    body = ""
    for row in grid(key, d):
        body += "<tr>" + f"<th>{row[0][0]}</th>"
        for v, t in row[1:]:
            body += f"<td>{v}" + (f"<br><span style='color:#52514e;font-size:.88em'>({t})</span>" if t else "") + "</td>"
        body += "</tr>"
    return f"<div class='qs-tw'><table class='qs-table'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def sort_md(key, d, perf_label):
    h = f"| {perf_label} (y-1) | R2 Low | R2 2 | R2 3 | R2 4 | R2 High | All | Low - High |\n|---|---|---|---|---|---|---|---|\n"
    for row in grid(key, d):
        h += "| " + row[0][0] + " | " + " | ".join((f"{v} ({t})" if t else v) for v, t in row[1:]) + " |\n"
    return h


def reg_rows(tname):
    """Paper-style regression table: columns = full / early / late periods; rows = variables with t in brackets."""
    T = RES[tname]
    spec = [("full", "tr2"), ("full", "comp"), ("full", "tr2_lag"), ("full", "comp_lag"), ("early", "tr2_lag"), ("early", "comp_lag"), ("late", "tr2_lag"), ("late", "comp_lag")]
    lagname = "alpha_lag" if tname == "table2" else "ir_lag"
    names = [("tr2", "TR&sup2;"), ("rmse_p", "RMSE (% a day)"), ("sdr_p", "SDR (% a day)"), (lagname, "Alpha" if tname == "table2" else "InfRatio")]
    d = 3 if tname == "table2" else 4
    rows = []
    for v, label in names:
        r = [label]
        for per, sp in spec:
            p = T[per][sp]
            r.append((f(p["coef"][v] / (100 if v in ("rmse_p", "sdr_p") else 1), d), f(p["t"][v], 2)) if v in p["coef"] else ("", None))
        rows.append(r)
    for lab, key, fm in (("N of funds", "n_funds", "{:,}"), ("Fund-years", "n_fund_years", "{:,}"), ("R&sup2;", "r2", "{:.2f}")):
        rows.append([lab] + [(fm.format(T[per][sp][key]), None) for per, sp in spec])
    return rows


def reg_html(tname):
    head = ("<tr><th rowspan='2'>Variables lagged one year</th><th colspan='4'>2012-2026</th><th colspan='2'>2012-2018</th><th colspan='2'>2019-2026</th></tr>"
            "<tr>" + "".join(f"<th>({i})</th>" for i in range(1, 9)) + "</tr>")
    body = ""
    for r in reg_rows(tname):
        body += f"<tr><th>{r[0]}</th>"
        for v, t in r[1:]:
            body += f"<td>{v}" + (f"<br><span style='color:#52514e;font-size:.88em'>({t})</span>" if t else "") + "</td>"
        body += "</tr>"
    return f"<div class='qs-tw'><table class='qs-table'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def reg_md(tname):
    h = "| Variable (lagged one year) | " + " | ".join(f"({i})" for i in range(1, 9)) + " |\n|---|" + "---|" * 8 + "\n"
    h += "| *Period* | 2012-26 | 2012-26 | 2012-26 | 2012-26 | 2012-18 | 2012-18 | 2019-26 | 2019-26 |\n"
    for r in reg_rows(tname):
        h += "| " + r[0].replace("&sup2;", "2") + " | " + " | ".join((f"{v} ({t})" if t else v) for v, t in r[1:]) + " |\n"
    return h


def fm_rows():
    T = RES["table5"]
    out = []
    for v, label in (("tr2", "TR&sup2;"), ("rmse_p", "RMSE"), ("sdr_p", "SDR"), ("alpha_lag", "Alpha (lag)"), ("ir_lag", "InfRatio (lag)")):
        r = [label]
        for col in ("alpha_tr2", "alpha_tr2_comp", "ir_tr2", "ir_tr2_comp"):
            c = T[col].get(v)
            r.append((f"{c['mean'] / (100 if v in ('rmse_p', 'sdr_p') else 1):.4g}", f"{c['t']:.2f}", f"{c['pos_share'] * 100:.0f}%") if c else ("", None, None))
        out.append(r)
    return out


def fm_html():
    head = "<tr><th rowspan='2'>Mean of annual coefficients</th><th colspan='2'>Alpha</th><th colspan='2'>InfRatio</th></tr><tr><th>(1)</th><th>(2)</th><th>(3)</th><th>(4)</th></tr>"
    body = ""
    for r in fm_rows():
        body += f"<tr><th>{r[0]}</th>"
        for v, t, p in r[1:]:
            body += f"<td>{v}" + (f"<br><span style='color:#52514e;font-size:.88em'>({t})</span><br><span style='color:#52514e;font-size:.88em'>[{p} positive]</span>" if t else "") + "</td>"
        body += "</tr>"
    return f"<div class='qs-tw'><table class='qs-table'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def fm_md():
    h = "| Mean of annual coefficients | Alpha (1) | Alpha (2) | InfRatio (3) | InfRatio (4) |\n|---|---|---|---|---|\n"
    for r in fm_rows():
        h += "| " + r[0].replace("&sup2;", "2") + " | " + " | ".join((f"{v} ({t}) [{p} pos.]" if t else v) for v, t, p in r[1:]) + " |\n"
    return h


def yearly_rows():
    return RES["yearly"]


def yearly_html():
    head = ("<tr><th>Year</th><th>Funds sorted</th><th>In best group</th><th>Alpha, best group (%)</th><th>Alpha, all funds (%)</th>"
            "<th>Return, best group (%)</th><th>Return, all funds (%)</th><th>Return, SPY (%)</th></tr>")
    body = ""
    for r in yearly_rows():
        y = f"{r['year']}*" if r["partial_year"] else str(r["year"])
        body += (f"<tr><th>{y}</th><td>{r['n_funds']}</td><td>{r['n_portfolio']}</td><td>{r['alpha_portfolio']:.2f}</td><td>{r['alpha_all']:.2f}</td>"
                 f"<td>{r['ret_portfolio']:.1f}</td><td>{r['ret_all_funds']:.1f}</td><td>{r['ret_spy']:.1f}</td></tr>")
    return f"<div class='qs-tw'><table class='qs-table'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def yearly_md():
    h = "| Year | Funds sorted | In best group | Alpha, best group (%) | Alpha, all funds (%) | Return, best group (%) | Return, all funds (%) | Return, SPY (%) |\n|---|---|---|---|---|---|---|---|\n"
    for r in yearly_rows():
        y = f"{r['year']}*" if r["partial_year"] else str(r["year"])
        h += f"| {y} | {r['n_funds']} | {r['n_portfolio']} | {r['alpha_portfolio']:.2f} | {r['alpha_all']:.2f} | {r['ret_portfolio']:.1f} | {r['ret_all_funds']:.1f} | {r['ret_spy']:.1f} |\n"
    return h


def table1_html():
    T = RES["table1"]
    lab = {"r2_p": "R&sup2; (prior year)", "alpha_p": "Alpha, prior year (% a year)", "rmse_p": "RMSE, daily (%)", "sdr_p": "SDR, daily (%)",
           "alpha": "Alpha, test year (% a year)", "ir": "InfRatio, test year"}
    scale = {"rmse_p": 100, "sdr_p": 100}
    head = "<tr><th>Panel A</th><th>Mean</th><th>Median</th><th>Std dev</th><th>Min</th><th>Max</th></tr>"
    body = ""
    for k, l in lab.items():
        s = scale.get(k, 1)
        d = 4 if k == "ir" else 3 if k == "r2_p" else 2
        body += f"<tr><th>{l}</th>" + "".join(f"<td>{T[k][c] * s:.{d}f}</td>" for c in ("mean", "median", "sd", "min", "max")) + "</tr>"
    body += f"<tr><th>Funds / fund-years</th><td colspan='5'>{T['n_funds']:,} / {T['n_fund_years']:,}</td></tr>"
    return f"<div class='qs-tw'><table class='qs-table'><thead>{head}</thead><tbody>{body}</tbody></table></div>"


if __name__ == "__main__":
    print(sort_md("table8_alpha", 3, "Alpha"))
    print(reg_md("table2"))
    print(fm_md())
