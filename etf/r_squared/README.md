# 4.3 R-squared: Amihud and Goyenko (2013) on US equity mutual funds, 2012-2026

Article: https://quantstr.at/quant-strategies/r-squared/

- `python/ag_replication.py` and `r/ag_replication.R`: the annual design (independent implementations, checked against each other by `python/verify_ag.py`).
- `python/fund_study.py`: monthly extension of the book's rule (Python only).
- `data/funds/yahoo_us_equity_funds_universe.csv`: the fund tickers used. Prices are not included (Yahoo terms); `python/fund_prices.py` downloads them with yfinance.
- `data/MANIFEST.json`: data vintages and hashes. Ken French factor files are in `data/raw/`.

Educational code, not investment advice.
