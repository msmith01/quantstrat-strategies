"""Charts for the alpha-rotation post (reads the Python outputs in ./data)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]; CH = ROOT / "charts"; CH.mkdir(exist_ok=True)
BLUE, PURPLE, GREEN, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
summ = json.load(open(ROOT / "data" / "results_python.json"))
SRC = "Source: Yahoo Finance adjusted closes, Kenneth French data library; author calculations. Strategies and equal-weight net of 10 bps per unit traded."


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=150, facecolor="white"); plt.close(fig)


ret = pd.read_csv(ROOT / "data" / "monthly_returns_python.csv", index_col=0, parse_dates=True)
SER = [("alpha3_net", "Alpha top 3", BLUE, "-"), ("alpha1_net", "Alpha top 1", PURPLE, "-"),
       ("mom3_net", "Momentum top 3 (4.1)", GREEN, "-"), ("ew_net", "Equal-weight 9", ORANGE, "-"), ("spy", "SPY", GREY, "--")]
eq = (1 + ret[[c for c, *_ in SER]]).cumprod() * 100
start = pd.Series(100.0, index=[eq.index[0] - pd.offsets.MonthEnd(1)])

# 1 equity curve (log)
fig, ax = plt.subplots(figsize=(9, 4.8))
for col, lab, c, ls in SER:
    s = pd.concat([start, eq[col]]); ax.plot(s.index, s, color=c, ls=ls, lw=2, label=f"{lab}  ${s.iloc[-1]:,.0f}")
ax.set_yscale("log"); ax.set_yticks([50, 100, 200, 400, 800]); ax.set_yticklabels(["$50", "$100", "$200", "$400", "$800"])
ax.legend(loc="upper left", frameon=False)
ax.set_title("Growth of $100 (log scale), Jan 2000 - Sep 2026"); ax.text(0, -0.13, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "01_equity_curve.png")

# 2 drawdowns
fig, ax = plt.subplots(figsize=(9, 4.0))
for col, lab, c, ls in SER:
    e = pd.concat([start, eq[col]]); dd = (e / e.cummax() - 1) * 100
    ax.plot(dd.index, dd, color=c, ls=ls, lw=1.7, label=f"{lab}  (worst {dd.min():.0f}%)")
ax.set_ylabel("Drawdown from peak (%)"); ax.legend(loc="lower left", frameon=False, ncol=2)
ax.set_title("Drawdowns: how far each portfolio sat below its previous peak"); ax.text(0, -0.17, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "02_drawdowns.png")

# 3 sensitivity heatmap (Sharpe, CAGR) on the common sample
Ws, Ns = [126, 252, 504], [1, 2, 3, 4]
G = np.array([[summ["sensitivity"][f"W{w}_N{n}"]["sharpe"] for n in Ns] for w in Ws])
C = np.array([[summ["sensitivity"][f"W{w}_N{n}"]["cagr_pct"] for n in Ns] for w in Ws])
SS = summ["sensitivity_sample"]; ewS, ewC = SS["equal_weight"]["sharpe"], SS["equal_weight"]["cagr_pct"]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.imshow(G, cmap=LinearSegmentedColormap.from_list("b", ["#cde2fb", "#184f95"]), aspect="auto"); ax.grid(False)
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{G[i,j]:.2f}\n({C[i,j]:.1f}%)", ha="center", va="center", fontsize=9,
                color="white" if G[i, j] > G.mean() else INK, fontweight="bold" if G[i, j] > ewS else "normal")
ax.set_xticks(range(4)); ax.set_xticklabels([f"top {n}" for n in Ns])
ax.set_yticks(range(3)); ax.set_yticklabels([f"{w} days" + (" (1 year)" if w == 252 else "") for w in Ws])
ax.set_xlabel("ETFs held"); ax.set_ylabel("Regression window")
ax.set_title("Sharpe ratio (CAGR) across settings")
ax.text(0, -0.24, f"Bold = beats equal-weight (Sharpe {ewS:.2f}, CAGR {ewC:.2f}%). The paper's window is one year (252 days).\nSame {SS['months']} months in every cell ({SS['start'][:7]} to {SS['end'][:7]}), fresh capital in each. Net of 10 bps.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "03_sensitivity.png")


def tab(name):
    return pd.read_csv(ROOT / "data" / f"regression_{name}_python.csv", index_col=0)


# 4 whole-sample regressions: alpha with 95% interval, and the three betas
full = tab("full"); S = list(full.index); x = np.arange(len(S))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
se = (full["alpha_pct_yr"] / full["alpha_t"]).abs()
ax[0].errorbar(x, full["alpha_pct_yr"], yerr=1.96 * se, fmt="o", color=BLUE, ecolor=BLUE, capsize=4, lw=1.6)
ax[0].axhline(0, color=INK, lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(S)
ax[0].set_ylabel("Alpha (% per year)"); ax[0].set_title("Alpha, Dec 1998 - Sep 2026, 95% interval")
wd = 0.26
for k, (col, lab, c) in enumerate((("beta_mkt", "MKT", BLUE), ("beta_smb", "SMB (size)", ORANGE), ("beta_hml", "HML (value)", PURPLE))):
    ax[1].bar(x + (k - 1) * wd, full[col], wd, color=c, label=lab)
ax[1].axhline(0, color=INK, lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(S)
ax[1].set_ylabel("Factor loading (beta)"); ax[1].set_title("Factor loadings"); ax[1].set_ylim(-0.6, 1.5); ax[1].legend(frameon=False, ncol=3, loc="upper center")
fig.text(0.01, 0.005, "Source: Eq. (364), daily excess returns, whole sample; Kenneth French data library, Yahoo Finance; author calculations.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(CH / "04_regression_full.png", dpi=150, facecolor="white"); plt.close(fig)

# 5 the latest one-year regression: what the rule actually ranks
lat = tab("latest").sort_values("alpha_pct_yr", ascending=False)
se = (lat["alpha_pct_yr"] / lat["alpha_t"]).abs()
fig, ax = plt.subplots(figsize=(9, 4.4))
y = np.arange(len(lat))[::-1]
top = set(lat.index[:3])
for yi, (s, row), e in zip(y, lat.iterrows(), se):
    c = BLUE if s in top else GREY
    ax.errorbar(row["alpha_pct_yr"], yi, xerr=1.96 * e, fmt="o", color=c, ecolor=c, capsize=4, lw=1.6)
ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(lat.index)
ax.set_xlabel("Alpha (% per year), 95% interval"); ax.set_title("What the rule ranks: one-year alphas to the last trading day of Sep 2026")
ax.text(0, -0.2, "Blue = the three ETFs the rule would buy. The intervals are wide compared with the gaps between ETFs, so the order is uncertain.\n" + "Source: Eq. (364), 252 daily excess returns; Kenneth French data library, Yahoo Finance; author calculations.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "05_latest_alpha.png")
print("charts:", sorted(p.name for p in CH.iterdir()))
