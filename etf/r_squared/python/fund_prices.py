"""Downloads daily adjusted closes (dividends and capital gains reinvested) for the filtered fund universe from Yahoo."""
import json, sys
import pandas as pd, yfinance as yf
tk = sorted(json.load(open("data/funds/universe_usd_pre2011.json")))
parts = []
for i in range(0, len(tk), 200):
    b = tk[i:i + 200]
    px = yf.download(b, start="1988-01-01", end="2026-10-01", auto_adjust=True, progress=False, threads=True)["Close"]
    parts.append(px.dropna(how="all", axis=1)); print(i, px.shape, flush=True)
px = pd.concat(parts, axis=1).sort_index()
px.to_parquet("data/funds/fund_prices.parquet")
print("saved", px.shape, px.index.min(), px.index.max())
