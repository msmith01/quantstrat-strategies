# 4.2 Strategy: Alpha rotation -- implementation of Kakushadze & Serur (2018),
# "151 Trading Strategies", Section 4.2, Eq. (364):
#     R_i(t) = alpha_i + b1_i * MKT(t) + b2_i * SMB(t) + b3_i * HML(t) + eps_i(t)
# R_i(t) is the ETF's DAILY return in excess of the risk-free rate, and MKT, SMB, HML are the three
# Fama-French factors (Kenneth French's data library). The regression uses the last year of daily
# returns (252 trading days; the paper's footnote 77: "typically 1 year, daily or weekly returns").
# At each month-end: rank the ETFs by alpha_i, buy the top N equal-weighted, hold for one month.
# This is the sector momentum rotation of Section 4.1 with the cumulative return replaced by alpha.
suppressPackageStartupMessages({ library(quantmod); library(xts) })
# install.packages(c("quantmod", "xts"))   # once

SECTORS <- c("XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY")
WINDOW <- 252; TOP_N <- 3; COST_BPS <- 10; T_MOM <- 12
START <- "1998-12-01"; END <- "2026-10-01"     # fixed end date (exclusive, as in the Python script) so the sample is reproducible
FF_URL <- "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

# Daily adjusted closes for the 9 sector ETFs + SPY, straight from Yahoo Finance
TICKERS <- c(SECTORS, "SPY")
px <- do.call(merge, lapply(TICKERS, function(s)
  Ad(getSymbols(s, src = "yahoo", from = START, to = END, auto.assign = FALSE))))
colnames(px) <- TICKERS

# Daily Fama-French 3 factors and the risk-free rate (percent) from Ken French's library
zf <- tempfile(fileext = ".zip"); download.file(FF_URL, zf, mode = "wb", quiet = TRUE)
lines <- readLines(unz(zf, "F-F_Research_Data_Factors_daily.csv"), warn = FALSE)
head_i <- grep("^,Mkt-RF", lines)[1]
body <- lines[(head_i + 1):length(lines)]
body <- body[grepl("^[0-9]{8},", body)]                       # data rows only (drops the copyright line)
ff <- read.csv(text = body, header = FALSE, col.names = c("date", "MKT", "SMB", "HML", "RF"))
ff <- xts(ff[, -1] / 100, as.Date(as.character(ff$date), "%Y%m%d"))   # percent -> decimal
last_day <- min(index(px)[nrow(px)], index(ff)[nrow(ff)])             # both series must cover the same last day
px <- px[paste0("/", last_day)]; ff <- ff[paste0("/", last_day)]

pm <- apply.monthly(px, function(x) tail(x, 1))        # month-end prices
index(pm) <- as.Date(format(index(pm), "%Y-%m-01"))    # label by month
months <- format(index(pm), "%Y-%m")

# Daily returns of the ETFs and the factors on common dates
rets <- px[, SECTORS] / lag(px[, SECTORS], 1) - 1
d <- na.omit(merge(rets, ff, join = "inner"))
Y <- coredata(d[, SECTORS]) - as.numeric(d$RF)         # excess returns (paper's footnote 50)
X <- coredata(d[, c("MKT", "SMB", "HML")])
month_end_rows <- tapply(seq_len(nrow(d)), format(index(d), "%Y-%m"), max)

# Eq. (364) at each month-end: alpha of every ETF over the last `window` trading days.
# lag > 0 is the factor-availability test: the window ends `lag` trading days before month-end, as if the
# factor data were published with that delay (lag = 0 is the main, idealised strategy).
rolling_alpha <- function(window = WINDOW, lag = 0) {
  A <- matrix(NA_real_, length(months), length(SECTORS), dimnames = list(months, SECTORS))
  for (m in names(month_end_rows)) {
    i <- month_end_rows[[m]] - lag
    if (i < window) next
    s <- (i - window + 1):i
    A[m, ] <- lm.fit(cbind(1, X[s, ]), Y[s, ])$coefficients[1, ]
  }
  A
}

# Eq. (364) for each ETF over one period, with lm(): annualised alpha (%), betas, t-statistics, R-squared
regression_table <- function(rows) {
  do.call(rbind, lapply(SECTORS, function(s) {
    f <- lm(Y[rows, s] ~ X[rows, ]); cf <- coef(summary(f))
    data.frame(alpha_pct_yr = cf[1, 1] * 252 * 100, alpha_t = cf[1, 3], beta_mkt = cf[2, 1], beta_smb = cf[3, 1],
               beta_hml = cf[4, 1], t_mkt = cf[2, 3], t_smb = cf[3, 3], t_hml = cf[4, 3],
               r2 = summary(f)$r.squared, days = length(rows), row.names = s)
  }))
}

# signal: matrix of month-end scores (months x ETFs); higher = better. Same engine as the 4.1 article.
run_strategy <- function(pm, signal, n = TOP_N, cost_bps = COST_BPS) {
  P <- coredata(pm[, SECTORS]); N <- nrow(P)
  r <- rbind(NA, P[-1, ] / P[-N, ] - 1)                # monthly returns
  first <- which(rowSums(!is.na(signal)) == length(SECTORS))[1]
  w_prev <- rep(0, length(SECTORS))                    # weights held during month m
  gross <- net <- turn <- rep(NA_real_, N)
  for (m in first:(N - 1)) {                           # signal at end of month m
    top <- order(signal[m, ], decreasing = TRUE)[1:n]
    w <- rep(0, length(SECTORS)); w[top] <- 1 / n
    g <- sum(w * r[m + 1, ])                           # earned over NEXT month
    drifted <- w_prev * (1 + r[m, ])                   # last month's weights after the month's price moves
    drifted <- drifted / sum(drifted)
    tn <- if (m == first) 1 else sum(abs(w - drifted)) # trades needed; first month buys everything
    gross[m + 1] <- g
    net[m + 1] <- g - tn * cost_bps / 1e4
    turn[m + 1] <- tn
    w_prev <- w
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

alpha <- rolling_alpha()                                                  # Eq. (364), one-year daily window
mom <- coredata(pm[, SECTORS]); mom <- rbind(matrix(NA, T_MOM, length(SECTORS)),
         mom[-(1:T_MOM), ] / mom[1:(nrow(mom) - T_MOM), ] - 1)             # Section 4.1 signal, Eq. (361)
a3  <- run_strategy(pm, alpha)                 # alpha rotation, top 3 of 9
a1  <- run_strategy(pm, alpha, n = 1)          # top 1 of 9: closest to the paper's top decile
m3  <- run_strategy(pm, mom)                   # 4.1 momentum top 3
ew  <- run_strategy(pm, mom, n = 9)            # all 9 equally, rebalanced monthly, same costs
ok <- !is.na(a3$net)                           # one common sample for everything
spy <- c(NA, as.numeric(pm[-1, "SPY"]) / as.numeric(pm[-nrow(pm), "SPY"]) - 1)

# ---- Regressions (Eq. 364): whole sample and the latest one-year window ----
full <- regression_table(seq_len(nrow(d)))
latest <- regression_table((nrow(d) - WINDOW + 1):nrow(d))
cat("Eq. (364) over the whole sample, daily excess returns (alpha in % per year):\n"); print(round(full, 2))
cat("\nEq. (364) over the latest one-year window:\n"); print(round(latest, 2))

# ---- Results table (net of costs) ----
rows <- c("total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct", "hit_rate_pct")
col <- function(x, turnover) { s <- stats(x[ok], ""); c(100 * (1 + s$total_return_pct / 100), unlist(s[rows[-1]]), turnover) }
results <- data.frame(
  Metric = c("$100 became", "CAGR (%)", "Volatility (%)", "Sharpe", "Max drawdown (%)", "Months with a gain (%)",
             "Monthly turnover (%)"),
  "Alpha top 3" = col(a3$net, mean(a3$turnover[ok]) * 100), "Alpha top 1" = col(a1$net, mean(a1$turnover[ok]) * 100),
  "Momentum top 3 (4.1)" = col(m3$net, mean(m3$turnover[ok]) * 100),
  "Equal-weight 9 (net)" = col(ew$net, mean(ew$turnover[ok]) * 100), SPY = col(spy, 0),
  check.names = FALSE, row.names = NULL)
results[, -1] <- round(results[, -1], 2)
cat("\n"); print(results, row.names = FALSE)

# ---- Sensitivity: every combination of window (trading days) and number of ETFs held n ----
# Every cell, and the equal-weight benchmark, starts with fresh capital in the same month: the first month in
# which the longest window (504 days) has an alpha. So every cell covers exactly the same months.
from_month <- function(signal, start) { signal[seq_len(nrow(signal)) < start, ] <- NA; signal }
first_full <- function(A) which(rowSums(is.na(A)) == 0)[1]
alphas <- lapply(c(126, 252, 504), rolling_alpha)
g0 <- max(sapply(alphas, first_full))
gok <- ok & seq_along(ok) > g0
sens <- expand.grid(window = c(126, 252, 504), n = 1:4)
sens[c("cagr_pct", "sharpe", "months")] <- t(mapply(function(w, n) {
  s <- stats(run_strategy(pm, from_month(alphas[[match(w, c(126, 252, 504))]], g0), n)$net[gok], "")
  c(s$cagr_pct, s$sharpe, s$months) }, sens$window, sens$n))
ew_g <- stats(run_strategy(pm, from_month(mom, g0), n = 9)$net[gok], "")
print(sens, row.names = FALSE)
cat(sprintf("Sensitivity sample: %s to %s, %d months; equal-weight CAGR %.2f, Sharpe %.3f\n",
            months[which(gok)[1]], months[tail(which(gok), 1)], sum(gok), ew_g$cagr_pct, ew_g$sharpe))

# ---- Factor-availability test: alpha from a window that ends `lag` trading days before month-end, and the trade
# at the same close or at the next trading day's close. Same months and fresh capital for every variant. ----
ends <- tapply(seq_len(nrow(px)), format(index(px), "%Y-%m"), max)
nx <- coredata(px)[pmin(ends + 1, nrow(px)), ]; nx[ends + 1 > nrow(px), ] <- NA   # last month: no next day yet
pm_next <- pm; coredata(pm_next) <- nx
variants <- expand.grid(lag = c(0, 5, 21), ex = c("same", "next"), stringsAsFactors = FALSE)
v_alpha <- lapply(c(0, 5, 21), function(l) rolling_alpha(WINDOW, l))
t0 <- max(sapply(v_alpha, first_full))
v_net <- lapply(seq_len(nrow(variants)), function(k)
  run_strategy(if (variants$ex[k] == "same") pm else pm_next, from_month(v_alpha[[match(variants$lag[k], c(0, 5, 21))]], t0))$net)
tok <- ok & seq_along(ok) > t0
for (v in v_net) tok <- tok & !is.na(v)
timing <- do.call(rbind, lapply(seq_len(nrow(variants)), function(k) {
  s <- stats(v_net[[k]][tok], ""); data.frame(variant = paste0("lag", variants$lag[k], "_", variants$ex[k]),
    cagr_pct = s$cagr_pct, sharpe = s$sharpe, max_drawdown_pct = s$max_drawdown_pct, months = s$months) }))
cat(sprintf("\nFactor-availability test, %s to %s (%d months):\n", months[which(tok)[1]], months[tail(which(tok), 1)], sum(tok)))
print(timing, row.names = FALSE)
# benchmarks on the same months, traded at the month-end close ("same") and at the next trading day's close ("next")
bench <- lapply(list(equal_weight = run_strategy(pm, from_month(mom, t0), n = 9), mom3 = run_strategy(pm, from_month(mom, t0)),
                     equal_weight_next = run_strategy(pm_next, from_month(mom, t0), n = 9), mom3_next = run_strategy(pm_next, from_month(mom, t0))),
                function(x) stats(x$net[tok], ""))
cat(sprintf("Same months, month-end close: equal-weight CAGR %.2f Sharpe %.3f; momentum top 3 CAGR %.2f Sharpe %.3f
",
            bench$equal_weight$cagr_pct, bench$equal_weight$sharpe, bench$mom3$cagr_pct, bench$mom3$sharpe))
cat(sprintf("Same months, next-day close:  equal-weight CAGR %.2f Sharpe %.3f; momentum top 3 CAGR %.2f Sharpe %.3f
",
            bench$equal_weight_next$cagr_pct, bench$equal_weight_next$sharpe, bench$mom3_next$cagr_pct, bench$mom3_next$sharpe))

# ---- Plots ----
yr <- as.numeric(format(index(pm)[ok], "%Y")) + (as.numeric(format(index(pm)[ok], "%m")) - 1) / 12
yr <- c(yr[1] - 1 / 12, yr)                  # the portfolios start at $100 one month before the first return
eq <- 100 * cbind("Alpha top 3" = cumprod(1 + a3$net[ok]), "Alpha top 1" = cumprod(1 + a1$net[ok]),
                  "Momentum top 3 (4.1)" = cumprod(1 + m3$net[ok]), "Equal-weight 9 (net)" = cumprod(1 + ew$net[ok]),
                  SPY = cumprod(1 + spy[ok]))
eq <- rbind(100, eq)                         # start at $100 so a first-month loss counts as a drawdown
cols <- c("#2a78d6", "#7b4fc4", "#1a9e75", "#eb6834", "grey40"); ltys <- c(1, 1, 1, 1, 2)
par(mfrow = c(3, 1), mar = c(3, 4, 3, 1))
matplot(yr, eq, type = "l", col = cols, lty = ltys, lwd = 2, log = "y", xlab = "", ylab = "$",
        main = "Growth of $100 (log scale)")
legend("topleft", colnames(eq), col = cols, lty = ltys, lwd = 2, bty = "n")
matplot(yr, 100 * (eq / apply(eq, 2, cummax) - 1), type = "l", col = cols, lty = ltys, lwd = 2, xlab = "",
        ylab = "%", main = "Drawdown from previous peak")
M <- matrix(sens$sharpe, nrow = 3)           # rows = window (126, 252, 504), columns = n (1 to 4)
image(1:4, 1:3, t(M)[, 3:1], col = hcl.colors(20, "Blues", rev = TRUE), axes = FALSE,
      xlab = "ETFs held (n)", ylab = "Regression window (trading days)", main = "Sharpe ratio by setting (same sample)")
axis(1, 1:4, 1:4); axis(2, 1:3, rev(c(126, 252, 504)), las = 1)
for (i in 1:3) for (j in 1:4) text(j, 4 - i, sprintf("%.2f", M[i, j]))

# ---- SAVE-BEGIN: optional, uncomment to write the results to ./data ----
# root <- getwd(); dir.create(file.path(root, "data"), showWarnings = FALSE)
# write.csv(data.frame(date = index(px), coredata(px)), file.path(root, "data", "prices_r.csv"), row.names = FALSE)
# write.csv(data.frame(date = index(ff), coredata(ff)), file.path(root, "data", "factors_r.csv"), row.names = FALSE)
# write.csv(results, file.path(root, "data", "results_table_r.csv"), row.names = FALSE)
# write.csv(sens, file.path(root, "data", "sensitivity_r.csv"), row.names = FALSE)
# write.csv(timing, file.path(root, "data", "timing_test_r.csv"), row.names = FALSE)
# write.csv(data.frame(name = names(bench), cagr_pct = sapply(bench, `[[`, "cagr_pct"), sharpe = sapply(bench, `[[`, "sharpe")),
#           file.path(root, "data", "timing_benchmarks_r.csv"), row.names = FALSE)
# write.csv(data.frame(start = months[which(gok)[1]], end = months[tail(which(gok), 1)], months = sum(gok),
#           ew_cagr_pct = ew_g$cagr_pct, ew_sharpe = ew_g$sharpe), file.path(root, "data", "sensitivity_sample_r.csv"), row.names = FALSE)
# write.csv(full, file.path(root, "data", "regression_full_r.csv"))
# write.csv(latest, file.path(root, "data", "regression_latest_r.csv"))
# write.csv(data.frame(month = months, alpha), file.path(root, "data", "alphas_r.csv"), row.names = FALSE)
# write.csv(data.frame(month = a3$month[ok], alpha3_net = a3$net[ok], alpha3_gross = a3$gross[ok],
#           alpha1_net = a1$net[ok], mom3_net = m3$net[ok], ew_net = ew$net[ok], spy = spy[ok],
#           turnover_alpha3 = a3$turnover[ok]),
#           file.path(root, "data", "monthly_returns_r.csv"), row.names = FALSE)
# ---- SAVE-END ----
