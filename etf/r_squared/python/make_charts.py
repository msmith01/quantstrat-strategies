"""Charts for the 4.3 R-squared article (reads data/results_ag_python.json, data/ag_yearly.csv, data/monthly_cum_funds.csv)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "charts"
CH.mkdir(exist_ok=True)
BLUE, PURPLE, GREEN, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
res = json.load(open(ROOT / "data/results_ag_python.json"))
yy = pd.read_csv(ROOT / "data/ag_yearly.csv")
SRC = "Source: Yahoo Finance fund prices, Kenneth French data library; author calculations. Test years 2012-2026 (2026 to 30 Sep)."


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CH / name, dpi=150, facecolor="white")
    plt.close(fig)


# 1. heat map of Table VIII, Panel A
c = res["table8_alpha"]["cell"]
M = np.array([[c[f"{a}_{q}"][0] for q in range(1, 6)] for a in range(5, 0, -1)])
T = np.array([[c[f"{a}_{q}"][1] for q in range(1, 6)] for a in range(5, 0, -1)])
lim = float(np.abs(M).max())
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.imshow(M, cmap=LinearSegmentedColormap.from_list("d", [ORANGE, "#ffffff", BLUE]), vmin=-lim, vmax=lim, aspect="auto")
ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{M[i, j]:.2f}\n({T[i, j]:.1f})", ha="center", va="center", fontsize=9, color=INK)
ax.set_xticks(range(5))
ax.set_xticklabels(["Low", "2", "3", "4", "High"])
ax.set_yticks(range(5))
ax.set_yticklabels(["High", "4", "3", "2", "Low"])
ax.set_xlabel("R-squared quintile, previous year")
ax.set_ylabel("Alpha quintile, previous year")
ax.set_title("Next-year alpha (% a year), by R-squared and past alpha")
fig.text(0.02, 0.01, "t-statistics in brackets; orange = negative alpha.\n" + SRC, fontsize=7, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(CH / "01_alpha_heatmap.png", dpi=150, facecolor="white")
plt.close(fig)

# 2. calendar-year returns of the three portfolios
x = np.arange(len(yy))
w = 0.27
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(x - w, yy.ret_portfolio, w, color=BLUE, label="Lowest R-squared, highest alpha funds")
ax.bar(x, yy.ret_all_funds, w, color=ORANGE, label="All funds (equal weight)")
ax.bar(x + w, yy.ret_spy, w, color=GREY, label="SPY")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(y) if y < 2026 else "2026*" for y in yy.year])
ax.set_ylabel("Calendar-year return (%)")
ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=8.5)
ax.set_title("Year by year: the paper's best group against all funds and SPY")
ax.text(0, -0.15, "* 2026 is January to September. " + SRC, transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "02_yearly_returns.png")

# 3. next-year alpha, year by year
fig, ax = plt.subplots(figsize=(10, 4.0))
ax.bar(x - 0.2, yy.alpha_portfolio, 0.4, color=BLUE, label="Lowest R-squared, highest alpha funds")
ax.bar(x + 0.2, yy.alpha_all, 0.4, color=ORANGE, label="All funds")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(y) if y < 2026 else "2026*" for y in yy.year])
ax.set_ylabel("Four-factor alpha (% a year)")
ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8.5)
ax.set_title("Alpha earned in each test year")
ax.text(0, -0.15, "* 2026 is January to September. " + SRC, transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "03_yearly_alpha.png")

# 4. growth of 100, annual rebalancing
cum = pd.read_csv(ROOT / "data/ag_cum_annual.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for col, lab, colr, ls in (("low_r2_high_alpha", "Lowest R-squared, highest alpha funds", BLUE, "-"), ("all_funds", "All funds", ORANGE, "-"), ("spy", "SPY", GREY, "--")):
    s = pd.concat([pd.Series([100.0], index=[cum.index[0] - 1]), cum[col]])
    ax.plot(s.index, s, color=colr, ls=ls, lw=2, label=f"{lab}  ${s.iloc[-1]:,.0f}")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Growth of $100, funds chosen once a year (no costs, equal weight)")
ax.text(0, -0.14, SRC, transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "04_growth_annual.png")

# 5. the book's monthly rule
m = pd.read_csv(ROOT / "data/monthly_cum_funds.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(8.8, 4.6))
for col, lab, colr, ls in (("r2_alpha", "R-squared x alpha long-short", BLUE, "-"), ("long_leg", "Long leg only", PURPLE, "-"),
                           ("equal_weight", "All funds", ORANGE, "-"), ("spy", "SPY", GREY, "--")):
    s = pd.concat([pd.Series([100.0], index=[m.index[0] - pd.offsets.MonthEnd(1)]), m[col]])
    ax.plot(s.index, s, color=colr, ls=ls, lw=2, label=f"{lab}  ${s.iloc[-1]:,.0f}")
ax.legend(frameon=False, loc="upper left")
ax.set_title("The book's monthly rule: growth of $100, net of 10 bps costs")
ax.text(0, -0.14, SRC, transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "05_monthly_rule.png")
print("charts:", sorted(p.name for p in CH.glob("*.png")))
