# 4.3 Strategy: R-squared -- implementation of Kakushadze & Serur (2018),
# "151 Trading Strategies", Section 4.3, Eq. (365)-(369):
#     R_i(t) = alpha_i + b1_i MKT(t) + b2_i SMB(t) + b3_i HML(t) + b4_i MOM(t) + eps_i(t)     (365)
#     R2_i   = 1 - SSres_i / SStot_i                                                           (366)
# R_i(t) is the ETF's DAILY return in excess of the risk-free rate; MKT, SMB, HML and Carhart's MOM are Kenneth
# French's daily factors. The regression uses the last year of daily returns (252 trading days).
# At each month-end the ETFs are sorted into quintiles by R2 and, inside each R2 quintile, into sub-quintiles by alpha
# (25 groups). The strategy buys the group with the lowest R2 quintile and the highest alpha sub-quintile and sells
# the group with the highest R2 quintile and the lowest alpha sub-quintile, equal weights, held for one month.
# Variants: long leg only; alpha-only quintiles (the 4.2 rule on the same universe); R2-only quintiles.
suppressPackageStartupMessages({ library(quantmod); library(xts) })
# install.packages(c("quantmod", "xts"))   # once

# 36 US-listed equity ETFs, all with Yahoo data from 2006-06-22 or earlier; list fixed before any result was seen
UNIVERSE <- c("XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY",
              "IYR","IYZ","IYT","IGV","SOXX","IBB","OIH","SMH","KBE","KRE","KIE",
              "XBI","XHB","XME","XOP","XRT","ITB","GDX",
              "IWM","IWD","IWF","IWN","IWO","IWS","IWP","IJH","IJR","IJK","IJJ","IJS","IJT",
              "MDY","QQQ","DIA")
FACTORS <- c("MKT", "SMB", "HML", "MOM")
WINDOW <- 252; NQ <- 5; COST_BPS <- 10
START <- "2005-06-01"; END <- "2026-10-01"     # fixed end date (exclusive, as in the Python script)
FF_BASE <- "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# Daily adjusted closes for the ETFs and SPY, straight from Yahoo Finance
TICKERS <- c(UNIVERSE, "SPY")
px <- do.call(merge, lapply(TICKERS, function(s)
  Ad(getSymbols(s, src = "yahoo", from = START, to = END, auto.assign = FALSE))))
colnames(px) <- TICKERS

# One daily CSV from Ken French's library (percent -> decimal); $QS_RAW_DIR points at an archived copy if set
french_daily <- function(zip_name, header_pattern, cols) {
  raw <- Sys.getenv("QS_RAW_DIR"); zf <- file.path(raw, zip_name)
  if (!file.exists(zf)) zf <- file.path(raw, sub("_CSV.zip", "_TXT.zip", zip_name))   # archived copy may be the TXT download
  if (!nzchar(raw) || !file.exists(zf)) { zf <- tempfile(fileext = ".zip"); download.file(paste0(FF_BASE, zip_name), zf, mode = "wb", quiet = TRUE) }
  lines <- readLines(unz(zf, unzip(zf, list = TRUE)$Name[1]), warn = FALSE)
  body <- gsub(",", " ", lines[grepl("^[0-9]{8}[, ]", lines)])    # data rows only, CSV or TXT layout
  f <- read.table(text = body, header = FALSE, col.names = c("date", cols))
  xts(f[, -1, drop = FALSE] / 100, as.Date(as.character(f$date), "%Y%m%d"))
}
ff <- merge(french_daily("F-F_Research_Data_Factors_daily_CSV.zip", "^,Mkt-RF", c("MKT", "SMB", "HML", "RF")),
            french_daily("F-F_Momentum_Factor_daily_CSV.zip", "^,.*Mom", "MOM"), join = "inner")[, c(FACTORS, "RF")]
last_day <- min(index(px)[nrow(px)], index(ff)[nrow(ff)])             # both series must cover the same last day
px <- px[paste0("/", last_day)]; ff <- ff[paste0("/", last_day)]

pm <- apply.monthly(px, function(x) tail(x, 1))        # month-end prices
index(pm) <- as.Date(format(index(pm), "%Y-%m-01"))    # label by month
months <- format(index(pm), "%Y-%m")
NU <- length(UNIVERSE)

# Daily returns of the ETFs and the factors on common dates
rets <- px[, UNIVERSE] / lag(px[, UNIVERSE], 1) - 1
d <- na.omit(merge(rets, ff, join = "inner"))
Y <- coredata(d[, UNIVERSE]) - as.numeric(d$RF)        # excess returns (paper's footnote 50)
X <- coredata(d[, FACTORS])
month_end_rows <- tapply(seq_len(nrow(d)), format(index(d), "%Y-%m"), max)

# Eq. (365)-(366) at each month-end: alpha, t-statistic and R2 of every ETF over the last `window` trading days
fit_window <- function(s) {
  A <- cbind(1, X[s, ]); fit <- lm.fit(A, Y[s, ]); res <- fit$residuals
  ss_res <- colSums(res^2)                                            # Eq. (367)
  ss_tot <- colSums(sweep(Y[s, ], 2, colMeans(Y[s, ]))^2)             # Eq. (368)-(369)
  list(alpha = fit$coefficients[1, ], r2 = 1 - ss_res / ss_tot)
}
rolling_fit <- function(window = WINDOW) {
  Aa <- Rr <- matrix(NA_real_, length(months), NU, dimnames = list(months, UNIVERSE))
  for (m in names(month_end_rows)) {
    i <- month_end_rows[[m]]
    if (i < window) next
    f <- fit_window((i - window + 1):i)
    Aa[m, ] <- f$alpha; Rr[m, ] <- f$r2
  }
  list(alpha = Aa, r2 = Rr)
}

# Bin 1..NQ of a rank 1..n (1 = lowest): ceiling(rank * NQ / n). Same rule as the Python script
bin_of <- function(rank, n) ceiling(rank * NQ / n)

# Long and short membership (logical matrices, months x ETFs) for one sorting variant
legs <- function(alpha, r2, variant) {
  lg <- sh <- matrix(FALSE, nrow(alpha), NU, dimnames = dimnames(alpha))
  for (m in seq_len(nrow(alpha))) {
    ok <- !is.na(alpha[m, ]) & !is.na(r2[m, ])
    if (sum(ok) < NQ) next
    a <- alpha[m, ok]; r <- r2[m, ok]
    qa <- bin_of(rank(a, ties.method = "first"), length(a)); qr <- bin_of(rank(r, ties.method = "first"), length(r))
    if (variant %in% c("r2_alpha", "long_leg")) {
      sub <- rep(0, length(a))
      for (q in c(1, NQ)) {                       # alpha sub-quintiles inside the lowest / highest R2 quintile
        mem <- which(qr == q); sub[mem] <- bin_of(rank(a[mem], ties.method = "first"), length(mem))
      }
      lg[m, ok] <- qr == 1 & sub == NQ            # lowest R2, highest alpha
      if (variant == "r2_alpha") sh[m, ok] <- qr == NQ & sub == 1     # highest R2, lowest alpha
    } else if (variant == "alpha_only") { lg[m, ok] <- qa == NQ; sh[m, ok] <- qa == 1
    } else if (variant == "r2_only") { lg[m, ok] <- qr == 1; sh[m, ok] <- qr == NQ
    } else stop(variant)
  }
  list(long = lg, short = sh)
}

# Same engine as 4.1/4.2 for a long leg and an optional short leg. Each leg is equal-weighted, held one month and
# rebalanced to 100% of its own notional; costs on both legs, drift-adjusted. Return = long return - short return
run_legs <- function(pm, L, cost_bps = COST_BPS, start = 1) {
  P <- coredata(pm[, UNIVERSE]); N <- nrow(P)
  r <- rbind(NA, P[-1, ] / P[-N, ] - 1)                       # monthly returns
  lg <- L$long; sh <- L$short; lg[seq_len(N) < start, ] <- FALSE; sh[seq_len(N) < start, ] <- FALSE
  has_short <- any(sh)
  first <- which(rowSums(lg) > 0)[1]
  wl_prev <- ws_prev <- rep(0, NU)
  gross <- net <- turn <- rep(NA_real_, N)
  for (m in first:(N - 1)) {                                  # signal at end of month m
    wl <- lg[m, ] / max(sum(lg[m, ]), 1); ws <- sh[m, ] / max(sum(sh[m, ]), 1)
    leg_turn <- function(w, w_prev, init) {
      dr <- w_prev * (1 + r[m, ]); dr <- dr / sum(dr)
      if (init) 1 else sum(abs(w - dr))
    }
    init <- m == first
    g <- sum(wl * r[m + 1, ]); tn <- leg_turn(wl, wl_prev, init)
    if (has_short) { g <- g - sum(ws * r[m + 1, ]); tn <- tn + leg_turn(ws, ws_prev, init) }
    gross[m + 1] <- g; net[m + 1] <- g - tn * cost_bps / 1e4; turn[m + 1] <- tn
    wl_prev <- wl; ws_prev <- ws
  }
  data.frame(month = months, gross = gross, net = net, turnover = turn)
}

stats <- function(ret, label) {
  ret <- ret[!is.na(ret)]; yrs <- length(ret) / 12
  eq <- c(1, cumprod(1 + ret)); dd <- eq / cummax(eq) - 1   # starts at 1 so an initial loss counts
  list(label = label, months = length(ret),
       total_return_pct = round((tail(eq, 1) - 1) * 100, 2),
       cagr_pct = round((tail(eq, 1)^(1 / yrs) - 1) * 100, 2),
       ann_vol_pct = round(sd(ret) * sqrt(12) * 100, 2),
       sharpe = round(mean(ret) / sd(ret) * sqrt(12), 3),
       max_drawdown_pct = round(min(dd) * 100, 2),
       hit_rate_pct = round(mean(ret > 0) * 100, 1))
}

fit <- rolling_fit()                                          # Eq. (365)-(366), one-year daily window
variants <- c("r2_alpha", "long_leg", "alpha_only", "r2_only")
res <- lapply(setNames(variants, variants), function(v) run_legs(pm, legs(fit$alpha, fit$r2, v)))
ew_L <- list(long = !is.na(fit$alpha), short = matrix(FALSE, nrow(fit$alpha), NU))
ew <- run_legs(pm, ew_L)                                      # equal-weight universe, same engine and costs
ok <- !is.na(res$r2_alpha$net)                                # one common sample for everything
spy <- c(NA, as.numeric(pm[-1, "SPY"]) / as.numeric(pm[-nrow(pm), "SPY"]) - 1)

# ---- Results table (net of costs) ----
rows <- c("total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct", "hit_rate_pct")
col <- function(x, turnover) { s <- stats(x[ok], ""); c(100 * (1 + s$total_return_pct / 100), unlist(s[rows[-1]]), turnover) }
results <- data.frame(
  Metric = c("$100 became", "CAGR (%)", "Volatility (%)", "Sharpe", "Max drawdown (%)", "Months with a gain (%)", "Monthly turnover (%)"),
  "R2 x alpha long-short" = col(res$r2_alpha$net, mean(res$r2_alpha$turnover[ok]) * 100),
  "Long leg only" = col(res$long_leg$net, mean(res$long_leg$turnover[ok]) * 100),
  "Alpha-only long-short" = col(res$alpha_only$net, mean(res$alpha_only$turnover[ok]) * 100),
  "R2-only long-short" = col(res$r2_only$net, mean(res$r2_only$turnover[ok]) * 100),
  "Equal-weight (net)" = col(ew$net, mean(ew$turnover[ok]) * 100), SPY = col(spy, 0),
  check.names = FALSE, row.names = NULL)
results[, -1] <- round(results[, -1], 2)
cat("\n"); print(results, row.names = FALSE)
