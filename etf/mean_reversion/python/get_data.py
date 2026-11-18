"""Downloads daily OHLCV for the 33 ETFs of Pagonidis (2014), Table 1, from Yahoo Finance (adjusted for splits and dividends).
Writes data/ohlc_<ticker>.csv and data/ohlc_all.csv (long format). IBS is a ratio inside one day, so the adjustment factor cancels."""
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKERS = "ECH EEM EFA EPHE EPI EPP EWA EWC EWD EWG EWH EWI EWJ EWL EWM EWO EWP EWQ EWS EWT EWU EWY EWZ EZA FXI GXG IDX ILF IWM QQQ RSX SPY THD".split()
frames, missing = [], []
for t in TICKERS:
    d = yf.download(t, period="max", auto_adjust=True, progress=False, multi_level_index=False)
    if d is None or d.empty:
        missing.append(t)
        continue
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["High", "Low", "Close"]).copy()
    d["ticker"] = t
    frames.append(d.reset_index())
    print(t, d.index.min().date(), d.index.max().date(), len(d), flush=True)
out = pd.concat(frames)
out.to_csv(ROOT / "data/ohlc_all.csv", index=False)
print("missing:", missing)
