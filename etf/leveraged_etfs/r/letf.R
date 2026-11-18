# Independent R implementation of the 4.5 leveraged-ETF study (see python/letf.py): the volatility-decay regression (Avellaneda-Zhang) and the
# short-both-legs strategy with a Treasury ETF, simulated day by day from share values, including trading costs, no interest on the proceeds and a
# margin-call rule. Writes data/letf_r_*.csv for verification.
suppressPackageStartupMessages(library(data.table))
px <- fread("data/prices_all.csv"); setnames(px, 1, "Date"); px[, Date := as.Date(Date)]; setorder(px, Date)
ERX_CHANGE <- as.Date("2020-04-01"); UYG_CHANGE <- as.Date("2023-03-20")
pairs <- data.table(name = c("S&P 500 2x","S&P 500 3x","Nasdaq-100 2x","Nasdaq-100 3x","Russell 2000 2x","Russell 2000 3x","Dow 30 2x","Dow 30 3x","S&P MidCap 2x","S&P MidCap 3x","Financials 2x","Financials 3x","Energy 3x to 2x","Technology 3x","Semiconductors 3x","20+yr Treasuries 3x"),
  u = c("SPY","SPY","QQQ","QQQ","IWM","IWM","DIA","DIA","MDY","MDY","XLF","XLF","XLE","XLK","SOXX","TLT"),
  p = c("SSO","UPRO","QLD","TQQQ","UWM","TNA","DDM","UDOW","MVV","UMDD","UYG","FAS","ERX","TECL","SOXL","TMF"),
  m = c("SDS","SPXU","QID","SQQQ","TWM","TZA","DXD","SDOW","MZZ","SMDD","SKF","FAZ","ERY","TECS","SOXS","TMV"),
  L = c(2,3,2,3,2,3,2,3,2,3,2,3,3,3,3,3))
get <- function(i) {
  cols <- c("Date", pairs$u[i], pairs$p[i], pairs$m[i], "SHV"); if (pairs$name[i] == "Financials 2x") cols <- c(cols, "IYF")
  f <- na.omit(px[, ..cols]); n <- nrow(f)
  for (v in setdiff(cols, "Date")) set(f, j = v, value = c(NA, f[[v]][-1] / f[[v]][-n] - 1))
  f <- f[-1]
  u <- f[[pairs$u[i]]]; if (pairs$name[i] == "Financials 2x") u <- ifelse(f$Date < UYG_CHANGE, f$IYF, u)
  g <- data.table(Date = f$Date, u = u, p = f[[pairs$p[i]]], m = f[[pairs$m[i]]], c = f$SHV, L = pairs$L[i])
  if (startsWith(pairs$name[i], "Energy")) g[Date >= ERX_CHANGE, L := 2]
  g
}
F <- lapply(seq_len(nrow(pairs)), get); names(F) <- pairs$name
# decay regression: all calendar years with >=120 days, and full years (>=240 days); years with a change of leverage are left out
rows <- list()
for (i in seq_len(nrow(pairs))) for (leg in c("p", "m")) {
  f <- F[[i]]
  for (y in unique(year(f$Date))) { g <- f[year(Date) == y]; if (nrow(g) < 120 || length(unique(g$L)) > 1) next
    lev <- (if (leg == "p") 1 else -1) * g$L[1]
    rows[[length(rows) + 1]] <- data.table(n = nrow(g), year = y, y = sum(log1p(g[[leg]])) - lev * sum(log1p(g$u)), x = lev * (lev - 1) / 2 * sum(g$u^2)) } }
D <- rbindlist(rows); res <- list()
for (nm in c("all", "full")) { d <- if (nm == "all") D else D[n >= 240]; fit <- lm(y ~ x, d)
  res[[nm]] <- data.table(sample = nm, n = nrow(d), intercept = coef(fit)[1], slope = coef(fit)[2], r2 = summary(fit)$r.squared) }
fwrite(rbindlist(res), "data/letf_r_decay.csv")
# strategy
sim <- function(f, freq, borrow = 0, trade_bps = 0, margin = FALSE, share = 1) {
  per <- switch(freq, daily = seq_len(nrow(f)), monthly = format(f$Date, "%Y-%m"), annual = format(f$Date, "%Y"))
  fee <- borrow / 100 / 252; n <- nrow(f); r <- numeric(n); E <- 1; prevper <- NULL; dead <- FALSE
  a <- 0; b <- 0; cc <- 0; col <- 1; closed <- FALSE
  for (t in seq_len(n)) {
    if (dead) next
    E_pre <- E
    if (is.null(prevper) || per[t] != prevper) {
      traded <- abs(0.5 * E - a) + abs(0.5 * E - b); E <- E - trade_bps / 1e4 * traded
      a <- 0.5 * E; b <- 0.5 * E; cc <- E; col <- E; closed <- FALSE; prevper <- per[t]
    }
    if (closed) { r[t] <- E / E_pre - 1; next }
    fee_t <- fee * (a + b)
    a <- a * (1 + f$p[t]); b <- b * (1 + f$m[t]); cc <- cc * (1 + share * f$c[t]); col <- col - fee_t
    En <- col + cc - a - b
    if (En <= 0) { r[t] <- -1; dead <- TRUE; next }
    if (margin && En < (if (f$L[t] >= 3) 0.9 else 0.6) * (a + b)) { col <- En; a <- 0; b <- 0; cc <- 0; closed <- TRUE }
    r[t] <- En / E_pre - 1; E <- En
  }
  r
}
cagr <- function(r) { fin <- prod(1 + r); if (fin > 0) (fin^(252 / length(r)) - 1) * 100 else -100 }
out <- list()
for (i in seq_len(nrow(pairs))) {
  for (freq in c("daily", "monthly", "annual")) for (bp in c(0, 1, 3, 5))
    out[[length(out) + 1]] <- data.table(pair = pairs$name[i], scen = paste0(freq, "|", bp), cagr = cagr(sim(F[[i]], freq, bp)))
  for (sc in c("base", "no_interest", "trade_cost", "margin", "all")) for (bp in c(0, 3)) {
    a <- list(share = if (sc %in% c("no_interest", "all")) 0 else 1, trade_bps = if (sc %in% c("trade_cost", "all")) 5 else 0, margin = sc %in% c("margin", "all"))
    out[[length(out) + 1]] <- data.table(pair = pairs$name[i], scen = paste0("scen:", sc, "|", bp), cagr = cagr(do.call(sim, c(list(F[[i]], "monthly", bp), a))))
  }
}
fwrite(rbindlist(out), "data/letf_r_strategy.csv"); cat("done\n")
