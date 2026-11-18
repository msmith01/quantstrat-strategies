# 4.5 Leveraged ETFs: volatility drag and the short-both-legs strategy, Yahoo data to 2026

Article: https://quantstr.at/quant-strategies/leveraged-etfs/

- `python/get_data.py`: downloads daily prices for the 16 ETF pairs, their unleveraged proxies, IYF and SHV from Yahoo Finance into `data/prices_all.csv` (not committed; Yahoo terms).
- `python/letf.py`: the volatility-drag regression and the strategy simulation (reset schedules, borrow fees, margin calls, trading costs, no-interest scenario). Writes `data/results_python.json`.
- `r/letf.R`: independent R implementation of the regressions and of the strategy for every pair and scenario; `python/verify.py` compares the two.
- `python/make_charts.py`, `python/build_post.py`: charts and the article.

Simulated results with simplified financing and margin rules; educational code, not investment advice.
