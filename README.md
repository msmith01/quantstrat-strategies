# quantstrat-strategies

Code for the Quant Strategies series on [quantstr.at](https://quantstr.at): academic strategies re-implemented in **Python and R** on real data, with plots and verification.

Source paper: Kakushadze, Z. and Serur, J. A. (2018). *151 Trading Strategies*. SSRN 3247865. See `ROADMAP.md` for the full strategy list and status.

## Layout

```
strategy_replications/
  ROADMAP.md
  <chapter>/                       e.g. etf/, stocks/, fixed_income/
    <strategy>/                    e.g. sector_momentum_rotation/
      data/                        written by the scripts: prices_*.csv, results_*.json, monthly returns
      python/                      <strategy>.py, verify.py, make_charts.py
      r/                           <strategy>.R
      charts/                      PNGs used in the post
```

## Conventions

- Each script is self-contained: it downloads its own prices from Yahoo Finance (yfinance in Python, quantmod in R), runs the strategy and writes its results to `./data`. `verify.py` checks that the two agree.
- `python/verify.py` must print `ALL CHECKS PASSED` (Python vs R, plus a hand calculation) before a post is drafted.
- The code shown in each article is extracted from the tested files in this repo, so what is shown is what was run.
- Every post ends with an academic reference naming the paper, section and equation replicated.
- Price data is not committed (it is Yahoo Finance data). The end date is pinned, but Yahoo can revise adjusted prices, so results may differ slightly from the post.
- The source paper is copyrighted and is not included. Get it from SSRN: https://ssrn.com/abstract=3247865

## Reproduce a strategy

```
cd etf/sector_momentum_rotation
python python/sector_momentum.py   # downloads data, prints the table, draws the plots
Rscript r/sector_momentum.R        # same strategy in R
python python/regen_results.py     # re-runs both scripts with their optional SAVE block on (writes ./data)
python python/verify.py            # Python vs R + hand check
python python/make_charts.py
```

Requires: pandas, numpy, matplotlib, yfinance; R with quantmod and xts.

## Disclaimer

Educational code, not investment advice. Backtested results are not a forecast of future returns.
