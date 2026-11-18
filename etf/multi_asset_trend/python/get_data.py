"""Downloads daily adjusted closes for ten multi-asset ETFs and a Treasury bill ETF (cash) from Yahoo Finance -> data/prices.csv."""
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "DBC"]
CASH = "SHV"
cols = {}
for t in UNIVERSE + [CASH]:
    d = yf.download(t, start="2003-01-01", end="2026-11-01", auto_adjust=True, progress=False, multi_level_index=False)
    cols[t] = d["Close"].dropna()
    print(t, cols[t].index.min().date(), cols[t].index.max().date(), len(cols[t]), flush=True)
pd.DataFrame(cols).to_csv(ROOT / "data/prices.csv")
