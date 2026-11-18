"""Charts for the 4.5 leveraged-ETF article (reads data/results_python.json, decay_etf_years.csv, portfolio_*daily.csv)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "charts"
CH.mkdir(exist_ok=True)
BLUE, PURPLE, GREEN, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
R = json.load(open(ROOT / "data/results_python.json"))
SRC = "Source: Yahoo Finance daily prices; author calculations."


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CH / name, dpi=150, facecolor="white")
    plt.close(fig)


# 1. realised log-return gap against the volatility-drag term, every ETF-year
D = pd.read_csv(ROOT / "data/decay_etf_years.csv")
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for lab, m, col in (("2x and -2x ETFs", D.L.abs() == 2, BLUE), ("3x and -3x ETFs", D.L.abs() == 3, ORANGE)):
    ax.scatter(D.x[m] * 100, D.y[m] * 100, s=14, alpha=0.65, color=col, label=lab)
xs = np.linspace(0, D.x.max() * 100, 50)
ax.plot(xs, -xs, color=INK, lw=1.2, ls="--", label="Prediction: gap equals minus the drag term")
b = R["decay_regression"]["all"]
ax.plot(xs, b["intercept"] * 100 + b["slope"] * xs, color=GREEN, lw=1.5, label=f"Fitted line, slope {b['slope']:.2f}")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("Drag term: L(L-1)/2 x sum of squared daily index returns (%)")
ax.set_ylabel("ETF log return minus L x index log return (%)")
ax.set_title("Leveraged ETFs lose what the volatility formula says they should")
ax.title.set_fontsize(11)
ax.legend(frameon=False, fontsize=8)
ax.text(0, -0.17, f"Each point is one ETF in one calendar year ({len(D)} ETF-years). Index return proxied by an unleveraged ETF. {SRC}", transform=ax.transAxes, fontsize=6.5, color=MUTED)
save(fig, "01_decay_scatter.png")

# 2. strategy CAGR by pair under the three reset schedules (no borrow cost), with cash
S = R["strategy"]
names = list(S)
fig, ax = plt.subplots(figsize=(8.4, 5.0))
y = np.arange(len(names))
for off, (fk, col, lab) in zip((-0.27, 0, 0.27), (("daily|0", BLUE, "Reset daily"), ("monthly|0", GREEN, "Reset monthly"), ("annual|0", ORANGE, "Reset annually"))):
    ax.barh(y + off, [S[n][fk]["cagr"] for n in names], 0.26, color=col, label=lab)
ax.scatter([S[n]["cash_only"]["cagr"] for n in names], y, color=INK, marker="|", s=90, zorder=5, label="Treasury bill ETF alone")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlim(-30, 20)
ax.set_xlabel("Annualised return on equity, % (bars cut at -30%; two annual resets lose everything)")
ax.set_title("Short a leveraged ETF and its inverse: how often you reset matters")
ax.title.set_fontsize(11)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.text(0, -0.13, f"No borrow fee. {SRC}", transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "02_pairs_by_reset.png")

# 3. growth of $100: monthly-reset portfolio at several borrow costs, daily-reset portfolio, cash
pm = pd.read_csv(ROOT / "data/portfolio_monthly_daily.csv", index_col=0, parse_dates=True)
pdl = pd.read_csv(ROOT / "data/portfolio_daily.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for c, col, lab in (("net_0", BLUE, "Monthly reset, no borrow fee"), ("net_3", ORANGE, "Monthly reset, 3% a year borrow fee")):
    ax.plot(pm.index, (1 + pm[c]).cumprod() * 100, color=col, lw=1.6, label=lab)
ax.plot(pdl.index, (1 + pdl["net_0"]).cumprod() * 100, color=GREEN, lw=1.4, label="Daily reset, no borrow fee")
ax.plot(pdl.index, (1 + pdl["cash"]).cumprod() * 100, color=GREY, lw=1.4, label="Treasury bill ETF alone")
ax.set_yscale("log")
ax.set_ylabel("Growth of $100 (log scale)")
ax.set_title("All pairs together: growth of $100")
ax.legend(frameon=False, fontsize=8.5)
ax.text(0, -0.15, f"Equal $ in each pair when it starts, no rebalancing between pairs. {SRC}", transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "03_portfolio_growth.png")
print("charts ok")
