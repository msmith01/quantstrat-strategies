# quantstrat-strategies

Code for the Quant Strategies series on [quantstr.at](https://quantstr.at): academic strategies re-implemented in **Python and R** on real data, with plots and verification.

Source paper: Kakushadze, Z. and Serur, J. A. (2018). *151 Trading Strategies*. SSRN 3247865. See `ROADMAP.md` for the full strategy list and status.

## Layout

```
strategy_replications/
  ROADMAP.md
  <chapter>/                       e.g. etf/, stocks/, fixed_income/
    <strategy>/                    e.g. sector_momentum_rotation/
      data/                        prices.csv (shared by Python and R), results_*.json, monthly returns
      python/                      download_data.py, <strategy>.py, verify.py, make_charts.py
      r/                           <strategy>.R
      charts/                      PNGs used in the post
```

## Conventions

- Python and R read the **same** `data/prices.csv`, so any difference is a code difference.
- `python/verify.py` must print `ALL CHECKS PASSED` (Python vs R, plus a hand calculation) before a post is drafted.
- The code shown in each article is extracted from the tested files in this repo, so what is shown is what was run.
- Every post ends with an academic reference naming the paper, section and equation replicated.
- `data/prices.csv` is not committed (it is Yahoo Finance data). Run `python/download_data.py` first; the end date is pinned, but Yahoo can revise adjusted prices, so results may differ slightly from the post.
- The source paper is copyrighted and is not included. Get it from SSRN: https://ssrn.com/abstract=3247865

## Reproduce a strategy

```
cd etf/sector_momentum_rotation/python
python download_data.py      # pins end date, rewrites data/prices.csv
python sector_momentum.py
Rscript ../r/sector_momentum.R
python verify.py
python make_charts.py
```

Requires: pandas, numpy, matplotlib, yfinance; R with xts and jsonlite.

## Disclaimer

Educational code, not investment advice. Backtested results are not a forecast of future returns.
