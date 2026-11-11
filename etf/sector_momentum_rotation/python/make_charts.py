"""Charts for the sector-momentum-rotation post (reads the Python outputs in ./data)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]; CH = ROOT / "charts"; CH.mkdir(exist_ok=True)
BLUE, PURPLE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#7b4fc4", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
SRC = "Source: Yahoo Finance adjusted closes; author calculations. Strategies and equal-weight net of 10 bps per unit traded."


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=150, facecolor="white"); plt.close(fig)


ret = pd.read_csv(ROOT / "data" / "monthly_returns_python.csv", index_col=0, parse_dates=True)
summ = json.load(open(ROOT / "data" / "results_python.json"))
SER = [("top3_net", "Top 3 of 9", BLUE, "-"), ("top1_net", "Top 1 of 9", PURPLE, "-"),
       ("ew_net", "Equal-weight 9", ORANGE, "-"), ("spy", "SPY", GREY, "--")]
eq = (1 + ret[[c for c, *_ in SER]]).cumprod() * 100
start = pd.Series(100.0, index=[eq.index[0] - pd.offsets.MonthEnd(1)])

# 1 equity curve (log)
fig, ax = plt.subplots(figsize=(9, 4.8))
for col, lab, c, ls in SER:
    s = pd.concat([start, eq[col]]); ax.plot(s.index, s, color=c, ls=ls, lw=2, label=f"{lab}  ${s.iloc[-1]:,.0f}")
ax.set_yscale("log"); ax.set_yticks([50, 100, 200, 400, 800]); ax.set_yticklabels(["$50", "$100", "$200", "$400", "$800"])
ax.legend(loc="upper left", frameon=False)
ax.set_title("Growth of $100 (log scale), Jan 2000 - Oct 2026"); ax.text(0, -0.13, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
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
Ts, Ns = [3, 6, 9, 12], [1, 2, 3, 4]
G = np.array([[summ["sensitivity"][f"T{T}_N{n}"]["sharpe"] for n in Ns] for T in Ts])
C = np.array([[summ["sensitivity"][f"T{T}_N{n}"]["cagr_pct"] for n in Ns] for T in Ts])
ewS, ewC = summ["equal_weight"]["sharpe"], summ["equal_weight"]["cagr_pct"]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.imshow(G, cmap=LinearSegmentedColormap.from_list("b", ["#cde2fb", "#184f95"]), aspect="auto"); ax.grid(False)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{G[i,j]:.2f}\n({C[i,j]:.1f}%)", ha="center", va="center", fontsize=9,
                color="white" if G[i, j] > G.mean() else INK, fontweight="bold" if G[i, j] > ewS else "normal")
ax.set_xticks(range(4)); ax.set_xticklabels([f"top {n}" for n in Ns])
ax.set_yticks(range(4)); ax.set_yticklabels([f"{t}m" + ("*" if t == 3 else "") for t in Ts])
ax.set_xlabel("ETFs held"); ax.set_ylabel("Formation period T")
ax.set_title("Sharpe ratio (CAGR) across settings")
ax.text(0, -0.2, f"Bold = beats equal-weight (Sharpe {ewS:.2f}, CAGR {ewC:.2f}%). *3 months is outside the paper's 6-12 month range.\nSame {summ['top3']['months']} months in every cell. " + "Net of 10 bps.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
save(fig, "03_sensitivity.png")
print("charts:", sorted(p.name for p in CH.iterdir()))
