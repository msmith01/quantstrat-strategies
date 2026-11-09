"""Download adjusted daily closes for the 9 original SPDR sector ETFs + SPY.

Both the Python and R implementations read the CSV written here, so any
difference between them is a code difference, not a data difference.
"""
from pathlib import Path
import yfinance as yf

TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"]
OUT = Path(__file__).resolve().parents[1] / "data" / "prices.csv"

px = yf.download(TICKERS, start="1998-12-01", end="2026-11-01", auto_adjust=True, progress=False)["Close"]
px = px[TICKERS].dropna(how="all")
px.index.name = "date"
px.to_csv(OUT)
print(f"saved {OUT}  {px.index[0].date()} -> {px.index[-1].date()}  rows={len(px)}")
