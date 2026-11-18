# 4.4 Mean-reversion: the IBS effect in equity ETFs (Pagonidis, 2014), Yahoo data to 2026

Article: https://quantstr.at/quant-strategies/mean-reversion/

- `python/get_data.py`: downloads daily OHLC for the ETFs from Yahoo Finance (yfinance) into `data/ohlc_all.csv`. The price file is not committed (Yahoo terms). GXG and RSX are no longer available.
- `python/ibs.py`: bucket tables, single-ETF IBS rule, RSI(3) with the IBS filter, and the book's decile rule. Writes `data/results_python.json`.
- `r/ibs.R`: independent R implementation of the bucket tables, the single-ETF rule and the decile rule. `python/verify.py` compares the two.
- `python/make_charts.py`, `python/build_post.py`: charts and the article.

This is an attempt to replicate the original study with free data; differences are listed in the article. Educational code, not investment advice.
