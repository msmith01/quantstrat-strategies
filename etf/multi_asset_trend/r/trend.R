# Independent R implementation of the 4.6 multi-asset trend-following study (see python/trend.py). Writes data/trend_r_returns.csv: net monthly returns of the 18 grid cells.
suppressPackageStartupMessages(library(data.table))
UNIVERSE <- c("SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "DBC"); CASH <- "SHV"; COST <- 10
px <- fread("data/prices.csv"); setnames(px, 1, "Date"); px[, Date := as.Date(Date)]; setorder(px, Date)
mkey <- format(px$Date, "%Y-%m"); mlab <- unique(mkey)
lastvalid <- function(x) tapply(x, factor(mkey, levels = mlab), function(v) { v <- v[!is.na(v)]; if (length(v)) v[length(v)] else NA_real_ })
cols <- c(UNIVERSE, CASH)
pm <- sapply(cols, function(cc) as.numeric(lastvalid(px[[cc]]))); rownames(pm) <- mlab
n <- nrow(pm)
d <- sapply(UNIVERSE, function(cc) { x <- px[[cc]]; c(NA, x[-1] / x[-length(x)] - 1) })
fsd <- function(x, k) { m1 <- frollmean(x, k); m2 <- frollmean(x^2, k); sqrt((m2 - m1^2) * k / (k - 1)) }
sig <- sapply(UNIVERSE, function(cc) as.numeric(lastvalid(fsd(d[, cc], 252) * sqrt(252))))
ma_ok <- sapply(UNIVERSE, function(cc) { x <- px[[cc]]; m <- frollmean(x, 200); as.numeric(lastvalid(ifelse(is.na(m), NA, as.numeric(x > m)))) })
lag <- function(m, k) rbind(matrix(NA, k, ncol(m)), m[seq_len(nrow(m) - k), , drop = FALSE])
rcum_for <- function(T) pm[, UNIVERSE] / lag(pm[, UNIVERSE], T) - 1
start_i <- which(apply(rcum_for(12), 1, function(r) all(!is.na(r))))[1]
r <- pm[, UNIVERSE] / lag(pm[, UNIVERSE], 1) - 1; rc <- pm[, CASH] / c(NA, pm[-n, CASH]) - 1
run <- function(T, ma, wt) {
  rcum <- rcum_for(T); ok <- !is.na(rcum) & rcum > 0 & !is.na(sig)
  if (ma) ok <- ok & !is.na(ma_ok) & ma_ok == 1
  raw <- switch(wt, "371" = rcum, "372" = rcum / sig, "373" = rcum / sig^2)
  raw[!ok | is.na(raw)] <- 0; raw[raw < 0] <- 0
  s <- rowSums(raw); w <- raw / ifelse(s > 0, s, NA); w[is.na(w)] <- 0
  wl <- rbind(NA, w[-n, ]); cw <- pmax(1 - rowSums(wl), 0)
  gross <- rowSums(wl * r) + cw * rc
  drift <- wl * (1 + r); tot <- rowSums(drift) + cw * (1 + rc); drift <- drift / tot
  turn <- rowSums(abs(wl - rbind(NA, drift[-n, ])))
  f <- start_i + 1; turn[f] <- sum(wl[f, ])
  (gross - turn * COST / 1e4)[f:n]
}
out <- list(); idx <- (start_i + 1):n
for (T in c(6, 9, 12)) for (ma in c(0, 1)) for (wt in c("371", "372", "373")) out[[sprintf("T%d|ma%d|%s", T, ma, wt)]] <- run(T, ma == 1, wt)
o <- as.data.table(out); o[, month := rownames(pm)[idx]]; fwrite(o, "data/trend_r_returns.csv"); cat("done", nrow(o), "months\n")
