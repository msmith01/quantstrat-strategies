# Sector momentum rotation -- replication of Kakushadze & Serur (2018),
# "151 Trading Strategies", Section 4.1, Eq. (361):
#     R_cum_i(t) = P_i(t) / P_i(t - T) - 1
# Each month-end: rank sector ETFs by R_cum, buy top N equal-weighted, hold 1 month.
suppressPackageStartupMessages({ library(xts); library(jsonlite) })

this_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) {
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(sub("^--file=", "", a)) else getwd() })
root <- normalizePath(file.path(this_dir, ".."))

SECTORS <- c("XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY")
T_MONTHS <- 12; TOP_N <- 3; COST_BPS <- 10

px <- read.csv(file.path(root, "data", "prices.csv"), stringsAsFactors = FALSE)
px <- xts(px[, -1], order.by = as.Date(px$date))
pm <- apply.monthly(px, function(x) tail(x, 1))        # month-end prices
index(pm) <- as.Date(format(index(pm), "%Y-%m-01")) # label by month
months <- format(index(pm), "%Y-%m")

run_strategy <- function(pm, T = T_MONTHS, n = TOP_N, cost_bps = COST_BPS) {
  P <- coredata(pm[, SECTORS]); N <- nrow(P)
  r <- rbind(NA, P[-1, ] / P[-N, ] - 1)                # monthly returns
  w_prev <- rep(0, length(SECTORS))
  gross <- net <- turn <- rep(NA_real_, N)
  for (m in (T + 1):(N - 1)) {                         # signal at end of month m
    rcum <- P[m, ] / P[m - T, ] - 1                    # Eq. (361)
    top <- order(rcum, decreasing = TRUE)[1:n]
    w <- rep(0, length(SECTORS)); w[top] <- 1 / n
    g <- sum(w * r[m + 1, ])                           # earned over NEXT month
    tn <- sum(abs(w - w_prev))                         # one-way turnover
    gross[m + 1] <- g
    net[m + 1] <- g - tn * cost_bps / 1e4
    turn[m + 1] <- tn
    w_prev <- w
  }
  data.frame(month = months, gross = gross, net = net, turnover = turn)
}

stats <- function(ret, label) {
  ret <- ret[!is.na(ret)]; yrs <- length(ret) / 12
  eq <- cumprod(1 + ret); dd <- eq / cummax(eq) - 1
  list(label = label, months = length(ret),
       total_return_pct = round((tail(eq, 1) - 1) * 100, 2),
       cagr_pct = round((tail(eq, 1)^(1 / yrs) - 1) * 100, 2),
       ann_vol_pct = round(sd(ret) * sqrt(12) * 100, 2),
       sharpe = round(mean(ret) / sd(ret) * sqrt(12), 3),
       max_drawdown_pct = round(min(dd) * 100, 2),
       hit_rate_pct = round(mean(ret > 0) * 100, 1))
}

res <- run_strategy(pm)
ok <- !is.na(res$net)
spy <- c(NA, as.numeric(pm[-1, "SPY"]) / as.numeric(pm[-nrow(pm), "SPY"]) - 1)
Psec <- coredata(pm[, SECTORS])
ew <- c(NA, rowMeans(Psec[-1, ] / Psec[-nrow(Psec), ] - 1))

summary <- list(
  strategy_net = stats(res$net[ok], "Sector momentum (net of costs)"),
  strategy_gross = stats(res$gross[ok], "Sector momentum (gross)"),
  equal_weight_sectors = stats(ew[ok], "Equal-weight 9 sectors"),
  spy = stats(spy[ok], "SPY buy & hold"),
  avg_monthly_turnover_pct = round(mean(res$turnover[ok]) * 100, 1))
sens <- list()
for (T in c(3, 6, 9, 12)) for (n in 1:4) {
  s <- run_strategy(pm, T, n); s <- stats(s$net[!is.na(s$net)], "")
  sens[[sprintf("T%d_N%d", T, n)]] <- list(cagr_pct = s$cagr_pct, sharpe = s$sharpe) }
summary$sensitivity <- sens

write_json(summary, file.path(root, "data", "results_r.json"), auto_unbox = TRUE, pretty = TRUE)
write.csv(data.frame(month = res$month[ok], strategy_net = res$net[ok],
          strategy_gross = res$gross[ok], equal_weight = ew[ok], spy = spy[ok],
          turnover = res$turnover[ok]),
          file.path(root, "data", "monthly_returns_r.csv"), row.names = FALSE)
str(summary[c("strategy_net", "equal_weight_sectors", "spy")])
