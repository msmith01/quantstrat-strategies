"""Charts for the sector-momentum-rotation post (reads the Python outputs)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sector_momentum as sm

ROOT = Path(__file__).resolve().parents[1]; CH = ROOT / "charts"; CH.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#7a7975", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 100})
SRC = "Source: Yahoo Finance adjusted closes; author calculations. Net of 10 bps/unit turnover."

def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=150, facecolor="white"); plt.close(fig)

px_m = sm.monthly_prices(); res, w, rcum = sm.run(px_m)
ret = pd.read_csv(ROOT / "data" / "monthly_returns_python.csv", index_col=0, parse_dates=True)
summ = json.load(open(ROOT / "data" / "results_python.json"))
eq = (1 + ret[["strategy_net", "equal_weight", "spy"]]).cumprod() * 100
start = pd.Series(100.0, index=[eq.index[0] - pd.offsets.MonthEnd(1)])

# 1 equity curve (log)
fig, ax = plt.subplots(figsize=(9, 4.8))
for col, c, ls, lab in [("strategy_net", BLUE, "-", "Sector momentum"), ("equal_weight", ORANGE, "-", "Equal-weight sectors"),
                        ("spy", GREY, "--", "SPY")]:
    s = pd.concat([start, eq[col]]); ax.plot(s.index, s, color=c, ls=ls, lw=2, label=lab)
    dy = {"strategy_net": -7, "equal_weight": 7, "spy": -20}[col]
    ax.annotate(f"{lab}  ${s.iloc[-1]:,.0f}", (s.index[-1], s.iloc[-1]), xytext=(6, dy),
                textcoords="offset points", va="center", fontsize=8.5, color=INK)
ax.set_yscale("log"); ax.set_ylim(40, 1600); ax.set_yticks([100, 200, 400, 800, 1600]); ax.set_yticklabels(["$100", "$200", "$400", "$800", "$1,600"])
ax.set_xlim(right=eq.index[-1] + pd.Timedelta(days=1500)); ax.legend(loc="upper left", frameon=False)
ax.set_title("Growth of $100 (log scale), Jan 2000 - Oct 2026"); ax.text(0, -0.13, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "01_equity_curve.png")

# 2 drawdowns
fig, ax = plt.subplots(figsize=(9, 3.8))
for col, c, ls, lab in [("equal_weight", ORANGE, "-", "Equal-weight sectors"), ("spy", GREY, "--", "SPY"),
                        ("strategy_net", BLUE, "-", "Sector momentum")]:
    dd = (eq[col] / eq[col].cummax() - 1) * 100; ax.plot(dd.index, dd, color=c, ls=ls, lw=1.8, label=lab)
ax.set_ylabel("Drawdown from peak (%)"); ax.legend(loc="lower left", frameon=False, ncol=3)
ax.set_title("Drawdowns: momentum cut the worst loss from -49% to -39%"); ax.text(0, -0.17, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "02_drawdowns.png")

# 3 holdings heatmap
fig, ax = plt.subplots(figsize=(9, 3.6))
H = w.fillna(0).T; ax.imshow(H.values, aspect="auto", cmap=LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE]), vmin=0, vmax=1/3,
    extent=[mdates_start := matplotlib.dates.date2num(H.columns[0]), matplotlib.dates.date2num(H.columns[-1]), len(H) - .5, -.5], interpolation="nearest")
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index); ax.xaxis_date(); ax.grid(False)
ax.set_title("Which sector ETFs were held (dark = in the top 3)"); ax.text(0, -0.15, SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "03_holdings.png")

# 4 sensitivity heatmap (Sharpe)
Ts, Ns = [3, 6, 9, 12], [1, 2, 3, 4]
G = np.array([[summ["sensitivity"][f"T{T}_N{n}"]["sharpe"] for n in Ns] for T in Ts])
C = np.array([[summ["sensitivity"][f"T{T}_N{n}"]["cagr_pct"] for n in Ns] for T in Ts])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.imshow(G, cmap=LinearSegmentedColormap.from_list("b", ["#cde2fb", "#184f95"]), aspect="auto"); ax.grid(False)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{G[i,j]:.2f}\n({C[i,j]:.1f}%)", ha="center", va="center", fontsize=9,
                color="white" if G[i, j] > G.mean() else INK)
ax.set_xticks(range(4)); ax.set_xticklabels([f"top {n}" for n in Ns]); ax.set_yticks(range(4)); ax.set_yticklabels([f"{t}m" for t in Ts])
ax.set_xlabel("ETFs held"); ax.set_ylabel("Formation period T")
ax.set_title("Sharpe ratio (CAGR) across parameters"); ax.text(0, -0.2, "Equal-weight benchmark: Sharpe 0.66, CAGR 8.95%", transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "04_sensitivity.png")

# 5 annual excess return vs equal-weight (polarity: blue up / orange down)
yr = ((1 + ret[["strategy_net", "equal_weight"]]).groupby(ret.index.year).prod() - 1) * 100
ex = yr["strategy_net"] - yr["equal_weight"]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(ex.index, ex.values, color=[BLUE if v >= 0 else ORANGE for v in ex.values], width=0.7)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel("Strategy minus equal-weight (pp)")
ax.set_title(f"Calendar-year excess return vs equal-weight sectors (beat in {int((ex>0).sum())} of {len(ex)} years)")
ax.text(0, -0.15, SRC + " 2026 covers Jan-Oct only.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "05_annual_excess.png")

# 6 formula in action: latest signal
last = rcum.dropna().index[-1]; s = (rcum.loc[last] * 100).sort_values()
top = set(rcum.loc[last].sort_values(ascending=False).index[:3])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(s.index, s.values, color=[BLUE if t in top else "#b7b6b0" for t in s.index], height=0.65)
for i, v in enumerate(s.values): ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
ax.set_xlabel("12-month cumulative return, $R_i(t)$"); ax.grid(axis="y", visible=False)
ax.set_title(f"The rule in action: ranking at {last:%d %b %Y}"); ax.text(0, -0.2, "Blue = bought for the next month. " + SRC, transform=ax.transAxes, fontsize=7.5, color=MUTED)
save(fig, "06_current_signal.png")
print("charts:", sorted(p.name for p in CH.iterdir()), "| last signal:", last.date(), sorted(top))
print("beat EW in", int((ex > 0).sum()), "of", len(ex), "years")
