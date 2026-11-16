"""Downloads Yahoo's current fund snapshot (.info) for every fund in the pre-dedupe universe: total net assets, expense
ratio, holdings turnover, inception date, Morningstar rating. These are TODAY's values (Yahoo has no history), so they are
proxies for the paper's end-of-year characteristics. Resumable: skips tickers already saved."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

D = Path(__file__).resolve().parents[1] / "data/funds"
out_f = D / "fund_info.json"
have = json.load(open(out_f)) if out_f.exists() else {}
tick = pd.read_csv(D / "yahoo_us_equity_funds_universe.csv").ticker.tolist()
todo = [t for t in tick if t not in have]
KEYS = ["totalAssets", "netAssets", "netExpenseRatio", "annualReportExpenseRatio", "annualHoldingsTurnover",
        "fundInceptionDate", "morningStarOverallRating", "fundFamily", "category"]


def get(t):
    for k in range(3):
        try:
            i = yf.Ticker(t).info
            return t, {key: i.get(key) for key in KEYS}
        except Exception as e:
            time.sleep(2 * (k + 1))
    return t, None


n = 0
with ThreadPoolExecutor(4) as ex:
    for t, r in ex.map(get, todo):
        have[t] = r
        n += 1
        if n % 100 == 0:
            json.dump(have, open(out_f, "w"))
            print(n, len(todo), flush=True)
json.dump(have, open(out_f, "w"))
print("done", len(have), "with data:", sum(1 for v in have.values() if v))
