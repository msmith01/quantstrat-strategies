# Independent R implementation of the IBS analysis (see python/ibs.py): next-day return by IBS bucket (Tables 3, 4), the single-ETF
# IBS long/short (long if IBS <= 0.2, short if >= 0.8) and the book's cross-sectional decile rule. Writes data/ibs_r_*.csv for verify.
suppressPackageStartupMessages(library(data.table))
CUT <- as.Date("2013-12-05")
d <- fread("data/ohlc_all.csv"); d[, Date := as.Date(Date)]
d[, ibs := ifelse(High > Low, (Close - Low) / (High - Low), NA_real_)]; d[!is.na(ibs) & (ibs < -1e-6 | ibs > 1 + 1e-6), ibs := NA_real_]; d[, ibs := pmin(pmax(ibs, 0), 1)]
setorder(d, ticker, Date)
d[, ret := Close / shift(Close) - 1, by = ticker]; d[, nxt := shift(ret, -1), by = ticker]; d[, ndate := shift(Date, -1), by = ticker]
d[, bucket := pmin(floor(ibs * 5), 4) + 1]
per <- list(paper_window = function(x) x <= CUT, after = function(x) x > CUT, all = function(x) rep(TRUE, length(x)))
rows3 <- list(); rows4 <- list()
for (p in names(per)) {
  g <- d[!is.na(nxt) & !is.na(ibs) & per[[p]](Date) & per[[p]](ndate)]   # signal date and return date both in the window
  g <- g[, if (.N >= 250) .SD, by = ticker]
  m <- g[, .(mean = mean(nxt) * 100, up = mean(nxt > 0) * 100, n = .N), by = .(ticker, bucket)]
  m[, period := p]; rows3[[p]] <- m
  rows4[[p]] <- m[, .(ticker, bucket, period, up)]
}
fwrite(rbindlist(rows3), "data/ibs_r_buckets.csv")
fwrite(rbindlist(rows4), "data/ibs_r_up_probabilities.csv")
# single-ETF strategy: position decided at close t-1 earns return t; cost on position change
strat <- function(g, cost) {
  pos <- ifelse(is.na(g$ibs), 0, ifelse(g$ibs <= 0.2, 1, ifelse(g$ibs >= 0.8, -1, 0)))
  p <- shift(pos, 1, fill = 0); turn <- abs(shift(pos, 1, fill = 0) - shift(pos, 2, fill = 0))
  r <- p * g$ret; r[is.na(r)] <- 0; r - turn * cost / 1e4
}
res <- list()
for (p in names(per)) for (cost in c(0, 2, 5, 10)) {
  v <- sapply(split(d, d$ticker), function(g) { m <- per[[p]](g$Date); r <- strat(g, cost)[m]; if (length(r) >= 250) (prod(1 + r)^(252 / length(r)) - 1) * 100 else NA_real_ })
  res[[paste(p, cost)]] <- data.table(period = p, cost = cost, avg_cagr = mean(v, na.rm = TRUE), n = sum(!is.na(v)))
}
fwrite(rbindlist(res), "data/ibs_r_strategy.csv")
# decile rule
ib <- dcast(d, Date ~ ticker, value.var = "ibs"); nx <- dcast(d, Date ~ ticker, value.var = "nxt")
dates <- ib$Date; I <- as.matrix(ib[, -1]); N <- as.matrix(nx[match(dates, nx$Date), -1])
nT <- ncol(I); cnt <- rowSums(!is.na(I)); k <- pmax(round(cnt * 0.1), 1)
# eligibility = valid IBS at the signal date; ties broken by a ticker order that rotates every day (same rule as the Python code)
rot <- (matrix(0:(nT - 1), nrow(I), nT, byrow = TRUE) + (0:(nrow(I) - 1))) %% nT
It <- I + rot / nT * 1e-9
rk <- t(apply(It, 1, function(x) rank(x, ties.method = "first", na.last = "keep")))
Lm <- (rk <= k) & !is.na(rk); Sm <- (rk > cnt - k) & !is.na(rk)
Lw <- Lm / pmax(rowSums(Lm), 1); Sw <- Sm / pmax(rowSums(Sm), 1)
have <- !is.na(N)
legret <- function(W) { Wa <- W * have; s <- rowSums(Wa); ifelse(s > 0, rowSums(Wa * ifelse(have, N, 0)) / s, NA_real_) }
rl <- legret(Lw); rsh <- legret(Sw)
lag1 <- function(M) rbind(0, M[-nrow(M), ])
turn <- rowSums(abs(Lw - lag1(Lw))) + rowSums(abs(Sw - lag1(Sw)))
drift <- function(W, r) { Nl <- ifelse(is.na(N), 0, N); dr <- lag1(W) * (1 + lag1(Nl)) / ifelse(is.na(c(0, r[-length(r)])), 1, 1 + c(0, r[-length(r)])); rowSums(abs(W - dr)) }
turn_d <- drift(Lw, rl) + drift(Sw, rsh)
ok <- cnt >= 10 & rowSums(Lm) > 0 & !is.na(rl) & !is.na(rsh)
out <- list()
for (p in names(per)) for (cost in c(0, 2, 5, 10)) {
  m <- ok & per[[p]](dates); ls <- (rl - rsh)[m] - turn[m] * cost / 1e4; lsd <- (rl - rsh)[m] - turn_d[m] * cost / 1e4
  out[[paste(p, cost)]] <- data.table(period = p, cost = cost, cagr = (prod(1 + ls)^(252 / length(ls)) - 1) * 100,
    cagr_drift = (prod(1 + lsd)^(252 / length(ls)) - 1) * 100, cagr_drift_borrow = (prod(1 + lsd - 0.01 / 252)^(252 / length(ls)) - 1) * 100, days = length(ls))
}
fwrite(rbindlist(out), "data/ibs_r_decile.csv")
cat("done
")
