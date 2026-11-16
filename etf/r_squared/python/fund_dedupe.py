"""One share class per fund. Classes of one fund hold the same portfolio and differ only by fees, so their daily returns
correlate at ~0.9999. Two funds are merged when the daily
return correlation over common days since 2012-01 is >= 0.999 (>= 250 common days), whatever the family (so also
cross-family funds that track the same index are one exposure). Merging is by connected components.
The class kept is the one with the longest price history (then the lowest ticker): chosen on data availability only,
never on performance. Also applies the bad-price screen (any one-day move beyond +-30%)."""
import json, numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data/funds"
pd.set_option("compute.use_numexpr", False)
meta = pd.read_csv(D / "yahoo_us_equity_funds_universe.csv").set_index("ticker")
px = pd.read_parquet(D / "fund_prices.parquet").loc["2010-06-01":]
v = px.to_numpy(); r = v[1:] / v[:-1] - 1; cols = list(px.columns)
bad = np.nanmax(np.abs(r), axis=0) > 0.30
keep = [i for i in range(len(cols)) if not bad[i]]
r = r[:, keep]; cols = [cols[i] for i in keep]
n = len(cols); fam = [str(meta.loc[c, "name"]).split()[0].lower() for c in cols]
rc = r[(px.index[1:] >= "2012-01-01")]                       # correlations over the test period (2012 onward)
m = np.isfinite(rc); z = np.where(m, rc, 0.0)
cnt = m.T.astype(float) @ m.astype(float)                       # common days
sx = z.T @ m.astype(float); sxx = (z ** 2).T @ m.astype(float); sxy = z.T @ z   # sums over common days
mean_x = sx / np.maximum(cnt, 1); mean_y = sx.T / np.maximum(cnt, 1)
cov = sxy - sx * mean_y - mean_x * sx.T + cnt * mean_x * mean_y
vx = sxx - 2 * sx * mean_x + cnt * mean_x ** 2; vy = vx.T
corr = cov / np.sqrt(np.maximum(vx * vy, 1e-30))
parent = list(range(n))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in range(n):
    for j in np.where((corr[i] >= 0.999) & (cnt[i] >= 250))[0]:
        if j > i: parent[find(j)] = find(i)
groups = {}
for i in range(n): groups.setdefault(find(i), []).append(cols[i])
rows, kept = [], []
for g in groups.values():
    g = sorted(g, key=lambda t: (-int(meta.loc[t, "n_days"]), t)); kept.append(g[0])
    rows.append({"kept": g[0], "name": meta.loc[g[0], "name"], "n_classes": len(g), "classes": " ".join(g)})
pd.DataFrame(rows).sort_values("kept").to_csv(D / "share_class_groups.csv", index=False)
u = meta.loc[sorted(kept)].reset_index(); u.to_csv(D / "yahoo_us_equity_funds_deduped.csv", index=False)
print("after bad-price screen:", n, " distinct funds:", len(kept), " multi-class groups:", sum(len(g) > 1 for g in groups.values()))
print(u.yahoo_category.value_counts().to_dict())
