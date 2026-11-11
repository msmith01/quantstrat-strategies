# Sector momentum rotation -- replication of Kakushadze & Serur (2018),
# "151 Trading Strategies", Section 4.1, Eq. (361):
#     R_cum_i(t) = P_i(t) / P_i(t - T) - 1
# Each month-end: rank sector ETFs by R_cum, buy top N equal-weighted, hold 1 month.
suppressPackageStartupMessages({ library(quantmod); library(xts) })
# install.packages(c("quantmod", "xts"))   # once

SECTORS <- c("XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY")
T_MONTHS <- 12; TOP_N <- 3; COST_BPS <- 10
START <- "1998-12-01"; END <- "2026-10-31"     # fixed end date so results are reproducible

# Daily adjusted closes for the 9 sector ETFs + SPY, straight from Yahoo Finance
TICKERS <- c(SECTORS, "SPY")
px <- do.call(merge, lapply(TICKERS, function(s)
  Ad(getSymbols(s, src = "yahoo", from = START, to = END, auto.assign = FALSE))))
colnames(px) <- TICKERS
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

# ---- Results table (net of costs) ----
rows <- c("total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct", "hit_rate_pct")
col <- function(x) { s <- stats(x[ok], ""); c(100 * (1 + s$total_return_pct / 100), unlist(s[rows[-1]])) }
results <- data.frame(
  Metric = c("$100 became", "CAGR (%)", "Volatility (%)", "Sharpe", "Max drawdown (%)", "Months with a gain (%)"),
  "Sector momentum (net)" = col(res$net), "Equal-weight 9 sectors" = col(ew), SPY = col(spy),
  check.names = FALSE, row.names = NULL)
results[, -1] <- round(results[, -1], 2)
print(results, row.names = FALSE)
cat(sprintf("Average monthly turnover: %.1f%%\n", mean(res$turnover[ok]) * 100))

# ---- Sensitivity: every combination of look-back T and number of ETFs held n ----
sens <- expand.grid(T_months = c(3, 6, 9, 12), n = 1:4)
sens[c("cagr_pct", "sharpe")] <- t(mapply(function(T, n) {
  s <- stats(run_strategy(pm, T, n)$net, ""); c(s$cagr_pct, s$sharpe) }, sens$T_months, sens$n))
print(sens, row.names = FALSE)

# ---- Plots ----
yr <- as.numeric(format(index(pm)[ok], "%Y")) + (as.numeric(format(index(pm)[ok], "%m")) - 1) / 12
eq <- 100 * cbind("Sector momentum" = cumprod(1 + res$net[ok]), "Equal-weight 9 sectors" = cumprod(1 + ew[ok]),
                  SPY = cumprod(1 + spy[ok]))
cols <- c("#2a78d6", "#eb6834", "grey40"); ltys <- c(1, 1, 2)
par(mfrow = c(3, 1), mar = c(3, 4, 3, 1))
matplot(yr, eq, type = "l", col = cols, lty = ltys, lwd = 2, log = "y", xlab = "", ylab = "$",
        main = "Growth of $100 (log scale)")
legend("topleft", colnames(eq), col = cols, lty = ltys, lwd = 2, bty = "n")
matplot(yr, 100 * (eq / apply(eq, 2, cummax) - 1), type = "l", col = cols, lty = ltys, lwd = 2, xlab = "",
        ylab = "%", main = "Drawdown from previous peak")
M <- matrix(sens$sharpe, nrow = 4)           # rows = T (3, 6, 9, 12), columns = n (1 to 4)
image(1:4, 1:4, t(M)[, 4:1], col = hcl.colors(20, "Blues", rev = TRUE), axes = FALSE,
      xlab = "ETFs held (n)", ylab = "Look-back T (months)", main = "Sharpe ratio by setting")
axis(1, 1:4, 1:4); axis(2, 1:4, rev(c(3, 6, 9, 12)), las = 1)
for (i in 1:4) for (j in 1:4) text(j, 5 - i, sprintf("%.2f", M[i, j]))

# ---- SAVE-BEGIN: optional, uncomment to write the results to ./data ----
# root <- getwd(); dir.create(file.path(root, "data"), showWarnings = FALSE)
# write.csv(data.frame(date = index(px), coredata(px)), file.path(root, "data", "prices_r.csv"), row.names = FALSE)
# write.csv(results, file.path(root, "data", "results_table_r.csv"), row.names = FALSE)
# write.csv(sens, file.path(root, "data", "sensitivity_r.csv"), row.names = FALSE)
# write.csv(data.frame(month = res$month[ok], strategy_net = res$net[ok],
#           strategy_gross = res$gross[ok], equal_weight = ew[ok], spy = spy[ok],
#           turnover = res$turnover[ok]),
#           file.path(root, "data", "monthly_returns_r.csv"), row.names = FALSE)
# ---- SAVE-END ----
