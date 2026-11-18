"""Downloads daily adjusted closes for leveraged / inverse ETF pairs, their unleveraged proxies and a Treasury bill ETF from Yahoo Finance.
Adjusted for splits (including the many reverse splits of leveraged ETFs) and distributions. Writes data/prices_all.csv (wide)."""
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
# name, proxy for the underlying index, leveraged long, leveraged inverse, leverage
PAIRS = [("S&P 500 2x", "SPY", "SSO", "SDS", 2), ("S&P 500 3x", "SPY", "UPRO", "SPXU", 3),
         ("Nasdaq-100 2x", "QQQ", "QLD", "QID", 2), ("Nasdaq-100 3x", "QQQ", "TQQQ", "SQQQ", 3),
         ("Russell 2000 2x", "IWM", "UWM", "TWM", 2), ("Russell 2000 3x", "IWM", "TNA", "TZA", 3),
         ("Dow 30 2x", "DIA", "DDM", "DXD", 2), ("Dow 30 3x", "DIA", "UDOW", "SDOW", 3),
         ("S&P MidCap 2x", "MDY", "MVV", "MZZ", 2), ("S&P MidCap 3x", "MDY", "UMDD", "SMDD", 3),
         ("Financials 2x", "XLF", "UYG", "SKF", 2), ("Financials 3x", "XLF", "FAS", "FAZ", 3),
         ("Energy 3x", "XLE", "ERX", "ERY", 3), ("Technology 3x", "XLK", "TECL", "TECS", 3),
         ("Semiconductors 3x", "SOXX", "SOXL", "SOXS", 3), ("20+yr Treasuries 3x", "TLT", "TMF", "TMV", 3)]
CASH = "SHV"
tick = sorted({t for p in PAIRS for t in p[1:4]} | {CASH, "IYF"})
cols = {}
for t in tick:
    d = yf.download(t, period="max", auto_adjust=True, progress=False, multi_level_index=False)
    if d is None or d.empty:
        print("MISSING", t); continue
    cols[t] = d["Close"].dropna()
    print(t, cols[t].index.min().date(), cols[t].index.max().date(), len(cols[t]), flush=True)
pd.DataFrame(cols).to_csv(ROOT / "data/prices_all.csv")
