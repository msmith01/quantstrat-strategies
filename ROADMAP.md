# Strategy replication roadmap

Source: Kakushadze, Z. and Serur, J. A. (2018). *151 Trading Strategies*. SSRN 3247865.
The paper is copyrighted; get it from SSRN (link above).

Series: **Quant Strategies** (website category) with one sub-category per chapter. Options (ch. 2) is skipped for now.
Each post = short re-worded strategy + maths + code (Python and R) + real-data plots + verification + academic reference to the exact section.
Where the paper includes Python code, copy it and translate to R.

Feasibility key: **A** = free price data (Yahoo), straightforward. **B** = needs extra data (fundamentals, futures curves, rates) that may be free but takes work. **C** = needs paid or unavailable data. **X** = not a backtestable trading strategy; not planned.

## Status

| # | Strategy | Section | Status |
|---|---|---|---|
| 1 | Sector momentum rotation | 4.1 | **Drafted** (`etf/sector_momentum_rotation/`), awaiting review |

## 4 ETFs (start here)
| Section | Strategy | Feasibility | Notes |
|---|---|---|---|
| 4.1 | Sector momentum rotation | A | Done (draft) |
| 4.1.1 | Sector momentum rotation with MA filter | A | Natural next post; reuses the 4.1 code. |
| 4.1.2 | Dual-momentum sector rotation | A | Reuses 4.1; add a bond/gold ETF as the fallback asset. |
| 4.2 | Alpha rotation | A | Same as 4.1 with alpha instead of cumulative return. |
| 4.3 | R-squared | A | |
| 4.4 | Mean-reversion | A | |
| 4.5 | Leveraged ETFs (LETFs) | A | Volatility decay is a good visual. |
| 4.6 | Multi-asset trend following | A | |

## 3 Stocks
3.1 Price-momentum (A, needs a stock universe), 3.2 Earnings-momentum (B/C), 3.3 Value (B), 3.4 Low-volatility anomaly (A), 3.5 Implied volatility (C), 3.6 Multifactor portfolio (B), 3.7 Residual momentum (A), 3.8 Pairs trading (A), 3.9 / 3.9.1 Mean-reversion single / multiple clusters (A), 3.10 Mean-reversion weighted regression (A), 3.11 Single moving average (A), 3.12 Two moving averages (A), 3.13 Three moving averages (A), 3.14 Support and resistance (A), 3.15 Channel (A), 3.16 Event-driven M&A (C), 3.17 ML single-stock KNN (A), 3.18 / 3.18.1 Statistical arbitrage optimisation (A), 3.19 Market-making (C, needs order-book data), 3.20 Alpha combos (A).

## 5 Fixed income
5.2 Bullets, 5.3 Barbells, 5.4 Ladders, 5.5 Bond immunization (all B; mostly illustrative maths with bond-ETF or Treasury-yield data), 5.6 Dollar-duration-neutral butterfly, 5.7 Fifty-fifty butterfly, 5.8 Regression-weighted butterfly, 5.8.1 Maturity-weighted butterfly (B), 5.9 Low-risk factor (A with bond ETFs), 5.10 Value factor (B), 5.11 Carry factor (B), 5.12 Rolling down the yield curve (B), 5.13 Yield curve spread, flatteners and steepeners (B, FRED yields), 5.14 CDS basis arbitrage (C), 5.15 Swap-spread arbitrage (B/C).

## 6 Indexes
6.2 Cash-and-carry arbitrage (B), 6.3 / 6.3.1 Dispersion trading (C, needs options), 6.4 Intraday arbitrage between index ETFs (C, needs intraday quotes), 6.5 Index volatility targeting with risk-free asset (A; good early candidate).

## 7 Volatility
7.2 VIX futures basis trading (B), 7.3 / 7.3.1 Volatility carry with two ETNs, hedging short VXX with VIX futures (B; ETN history is awkward), 7.4 / 7.4.1 Volatility risk premium (B/C), 7.5 Volatility skew, long risk reversal (C), 7.6 Volatility trading with variance swaps (C).

## 8 Foreign exchange
8.1 Moving averages with HP filter (A), 8.2 / 8.2.1 Carry trade, high-minus-low carry (B, rates), 8.3 Dollar carry trade (B), 8.4 Momentum and carry combo (B), 8.5 FX triangular arbitrage (C, needs tick quotes).

## 9 Commodities
9.1 Roll yields (B), 9.2 Trading based on hedging pressure (B, CFTC COT data), 9.3 Portfolio diversification with commodities (A), 9.4 Value (B), 9.5 Skewness premium (B), 9.6 Trading with pricing models (B).

## 10 Futures
10.1 / 10.1.1 / 10.1.2 Hedging risk with futures, cross-hedging, interest-rate hedging (B), 10.2 Calendar spread (B), 10.3 / 10.3.1 Contrarian trading (A/B), 10.4 Trend following (A/B).

## 11 Structured assets
11.1-11.7 CDO and MBS strategies (C; mostly not replicable with public data).

## 12 Convertibles
12.1 Convertible arbitrage, 12.2 Convertible option-adjusted spread (C).

## 13 Tax arbitrage
13.1 Municipal bond tax arbitrage (B), 13.2 / 13.2.1 Cross-border tax arbitrage (C).

## 14 Miscellaneous assets
14.1 Inflation hedging with inflation swaps (B), 14.2 TIPS-Treasury arbitrage (B, FRED), 14.3 Weather risk (C), 14.4 Energy spark spread (B).

## 15 Distressed assets
15.1-15.3.1 (C; mostly narrative, little backtestable).

## 16 Real estate
16.2 Mixed-asset diversification with real estate (A with REIT ETFs), 16.3-16.3.3 Intra-asset diversification (B), 16.4 Real estate momentum, regional approach (B), 16.5 Inflation hedging (B), 16.6 Fix-and-flip (X).

## 17 Cash
17.3 Liquidity management (B), 17.4 Repo (C). 17.2 Money laundering, 17.5 Pawnbroking, 17.6 Loan sharking are not trading strategies (X).

## 18 Cryptocurrencies
18.2 Artificial neural network (A, crypto prices), 18.3 Sentiment analysis, naive Bayes Bernoulli (C, needs text data; the paper includes code).

## 19 Global macro
19.2 Fundamental macro momentum (B), 19.3 Global macro inflation hedge (B), 19.4 Global fixed-income strategy (B), 19.5 Trading on economic announcements (B/C).

## Suggested order after the pilot
4.1.1, 4.1.2, 4.6, 4.2, 4.3, 4.4, 4.5 (finish ETFs), then 6.5 (index volatility targeting), then Stocks (3.11-3.15 moving-average strategies share code), then Fixed Income and Volatility.
