"""Charts for the 4.4 IBS article (reads data/results_python.json, decile_daily.csv, table5_avg_by_year.csv)."""
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
                     "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
R = json.load(open(ROOT / "data/results_python.json"))
SRC = "Source: Yahoo Finance daily prices; author calculations. Average over 31 ETFs."
PAPER = [0.350, 0.096, 0.025, -0.032, -0.126]            # Pagonidis (2014), Table 3, average row


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=150, facecolor="white"); plt.close(fig)


# 1. next-day return by IBS bucket
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(5); w = 0.36
a, b = R["table3_avg"]["paper_window"]["mean"], R["table3_avg"]["after"]["mean"]
ax.bar(x - w / 2, a, w, color=BLUE, label="To 5 Dec 2013 (the original window)")
ax.bar(x + w / 2, b, w, color=ORANGE, label="After 5 Dec 2013")
ax.scatter(x - w / 2, PAPER, color=INK, zorder=5, s=26, marker="D", label="Published averages (Pagonidis, 2014)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(["1st\n0-0.2", "2nd\n0.2-0.4", "3rd\n0.4-0.6", "4th\n0.6-0.8", "5th\n0.8-1"])
ax.set_xlabel("IBS bucket on day t"); ax.set_ylabel("Average close-to-close return on day t+1 (%)")
ax.set_title("Next-day return by IBS bucket"); ax.legend(frameon=False, fontsize=8.5)
ax.text(0, -0.24, SRC, transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "01_ibs_buckets.png")

# 2. average yearly return of the single-ETF IBS strategy
yr = pd.read_csv(ROOT / "data/table5_avg_by_year.csv", index_col=0).avg_return_pct.sort_index()
yr = yr[yr.index >= 1996]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
cols = [BLUE if y <= 2013 else ORANGE for y in yr.index]
ax.bar(range(len(yr)), yr.values, color=cols)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(len(yr))); ax.set_xticklabels([str(y) if y % 2 == 0 else "" for y in yr.index], rotation=0, fontsize=8)
ax.set_ylabel("Average return across ETFs (%)"); ax.set_title("Long if IBS 0.2 or below, short if 0.8 or above: calendar-year return, before costs")
ax.title.set_fontsize(11)
ax.text(0, -0.17, "Blue: to 2013 (the original window). Orange: after. Average of each ETF's compounded yearly return. Source: Yahoo Finance; author calculations.", transform=ax.transAxes, fontsize=6.5, color=MUTED)
save(fig, "02_yearly_strategy.png")

# 3. the book's decile rule: growth of $100 after costs, since the paper's end date
dd = pd.read_csv(ROOT / "data/decile_daily.csv", index_col=0, parse_dates=True)
dd = dd[dd.index > "2013-12-05"]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
for c, col, lab in (("net_0bp", BLUE, "Before costs"), ("net_2bp", GREEN, "2 bps per unit traded"), ("net_5bp", ORANGE, "5 bps per unit traded")):
    ax.plot(dd.index, (1 + dd[c]).cumprod() * 100, color=col, lw=1.6, label=lab)
ax.plot(dd.index, (1 + dd.spy.fillna(0)).cumprod() * 100, color=GREY, lw=1.4, label="SPY")
ax.set_yscale("log"); ax.set_ylabel("Growth of $100 (log scale)"); ax.set_title("Book's rule: bottom IBS decile long, top decile short, daily, after 5 Dec 2013")
ax.title.set_fontsize(11); ax.legend(frameon=False, fontsize=8.5)
ax.text(0, -0.15, "Equal weights within each leg, \$1 long and \$1 short. Source: Yahoo Finance; author calculations.", transform=ax.transAxes, fontsize=7, color=MUTED)
save(fig, "03_decile_rule.png")
print("charts ok")
