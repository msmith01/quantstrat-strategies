"""Charts for the 4.6 multi-asset trend-following article (reads data/results_python.json, monthly_returns.csv)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "charts"; CH.mkdir(exist_ok=True)
BLUE, PURPLE, GREEN, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
R = json.load(open(ROOT / "data/results_python.json"))
M = pd.read_csv(ROOT / "data/monthly_returns.csv", index_col=0, parse_dates=True)
SRC = "Source: Yahoo Finance adjusted prices; author calculations. Net of 10 bps trading costs."


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=150, facecolor="white"); plt.close(fig)


SER = [("Eq.(371)", BLUE, "Weights by return, Eq. (371)"), ("Eq.(372)", PURPLE, "Return / volatility, Eq. (372)"), ("Eq.(373)", GREEN, "Return / variance, Eq. (373)"),
       ("Equal weight, all ten", ORANGE, "Equal weight, all ten ETFs"), ("SPY buy and hold", INK, "SPY"), ("60/40 SPY and IEF, monthly", GREY, "60/40 SPY and IEF")]
curve = lambda s: np.r_[100.0, 100 * (1 + s).cumprod().to_numpy()]
idx = lambda s: pd.DatetimeIndex([s.index[0] - pd.offsets.MonthEnd(1)]).append(s.index)

fig, ax = plt.subplots(figsize=(8.4, 4.6))
for k, c, lab in SER:
    ax.plot(idx(M[k]), curve(M[k]), color=c, lw=1.6 if k.startswith("Eq.(37") else 1.2, label=lab)
ax.set_yscale("log"); ax.set_ylabel("Growth of $100 (log scale)")
ax.set_title("Trend-following weights against simple alternatives, 2007 to 2026"); ax.legend(frameon=False, fontsize=8)
ax.text(0, -0.13, f"Trend rule: 12-month return, 200-day moving-average filter. {SRC}", transform=ax.transAxes, fontsize=6.5, color=MUTED)
save(fig, "01_growth.png")

fig, ax = plt.subplots(figsize=(8.4, 4.2))
for k, c, lab in SER:
    if k == "Eq.(371)" or k == "Eq.(373)":
        continue
    cv = curve(M[k]); dd = (cv / np.maximum.accumulate(cv) - 1) * 100
    ax.plot(idx(M[k]), dd, color=c, lw=1.3, label=lab)
ax.set_ylabel("Fall from previous peak, %"); ax.set_title("Drawdowns: trend following against holding everything")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.text(0, -0.13, f"{SRC}", transform=ax.transAxes, fontsize=6.5, color=MUTED)
save(fig, "02_drawdowns.png")

G = R["grid"]; fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
for ax, ma in zip(axs, (0, 1)):
    x = np.arange(3)
    for off, (wt, col) in zip((-0.27, 0, 0.27), (("371", BLUE), ("372", PURPLE), ("373", GREEN))):
        ax.bar(x + off, [G[f"T{T}|ma{ma}|{wt}"]["cagr"] for T in (6, 9, 12)], 0.26, color=col, label=f"Eq. ({wt})")
    ax.axhline(R["benchmarks"]["Equal weight, all ten"]["cagr"], color=ORANGE, lw=1.2, ls="--", label="Equal weight, all ten")
    ax.set_xticks(x); ax.set_xticklabels(["6 months", "9 months", "12 months"]); ax.set_xlabel("Return look-back")
    ax.set_title("With 200-day moving-average filter" if ma else "Without moving-average filter", fontsize=10)
axs[0].set_ylabel("Annual return, % (net)"); axs[1].legend(frameon=False, fontsize=8)
fig.suptitle("Return by look-back, weighting and filter", fontsize=11, fontweight="bold")
fig.text(0.01, 0.01, SRC, fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(CH / "03_sensitivity.png", dpi=150, facecolor="white"); plt.close(fig)
print("charts ok")
