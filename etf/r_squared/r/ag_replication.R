# Independent R implementation of the Amihud and Goyenko (2013) design on Yahoo US equity funds (see python/ag_replication.py).
# For every fund and calendar year: regress daily excess returns on MKT, SMB, HML, MOM and their one-day lags; keep alpha
# (annualised %), R2, RMSE, InfRatio. Year y-1 R2 and alpha predict year y alpha. Writes data/ag_r_*.csv for verify.py.
# Run from the r_squared folder with QS_RAW_DIR=data/raw. Inputs: data/funds/ag_prices.csv (adjusted closes of the active funds).
suppressPackageStartupMessages({ library(data.table); library(sandwich) })
D <- "data/funds"; MIN_PRIOR <- 125; MIN_TEST <- 50

french_daily <- function(zip_name, cols) {
  raw <- Sys.getenv("QS_RAW_DIR"); zf <- file.path(raw, zip_name)
  if (!file.exists(zf)) zf <- file.path(raw, sub("_CSV.zip", "_TXT.zip", zip_name))
  lines <- readLines(unz(zf, unzip(zf, list = TRUE)$Name[1]), warn = FALSE)
  body <- gsub(",", " ", lines[grepl("^[0-9]{8}[, ]", lines)])
  f <- read.table(text = body, header = FALSE, col.names = c("date", cols))
  data.frame(date = as.Date(as.character(f$date), "%Y%m%d"), f[, -1, drop = FALSE] / 100)
}
f3 <- french_daily("F-F_Research_Data_Factors_daily_CSV.zip", c("MKT", "SMB", "HML", "RF"))
fm <- french_daily("F-F_Momentum_Factor_daily_CSV.zip", "MOM")
ff <- merge(f3, fm, by = "date")                      # days both files cover
px <- fread(file.path(D, "ag_prices.csv")); dates <- as.Date(px$Date); P <- as.matrix(px[, -1]); tick <- colnames(P)
last <- min(tail(dates, 1), tail(ff$date, 1)); keep <- dates <= last; dates <- dates[keep]; P <- P[keep, ]; ff <- ff[ff$date <= last, ]
FAC <- c("MKT", "SMB", "HML", "MOM")
Fm <- as.matrix(ff[, FAC]); Z <- cbind(Fm, rbind(NA, Fm[-nrow(Fm), ])); colnames(Z) <- c(FAC, paste0(FAC, "_L"))
ok_z <- complete.cases(Z) & !is.na(ff$RF)
ret <- rbind(NA, P[-1, ] / P[-nrow(P), ] - 1)         # simple daily returns; NA across price gaps
common <- intersect(as.character(dates), as.character(ff$date[ok_z]))
ri <- match(common, as.character(dates)); fi <- match(common, as.character(ff$date))
RET <- ret[ri, ]; ZZ <- Z[fi, ]; RF <- ff$RF[fi]; yrs <- as.integer(format(as.Date(common), "%Y"))

year_fit <- function(y) {
  m <- which(yrs == y); X <- cbind(1, ZZ[m, ]); Y <- RET[m, ] - RF[m]
  res <- matrix(NA_real_, length(tick), 6, dimnames = list(tick, c("alpha", "r2", "rmse", "sdr", "ir", "n")))
  for (j in seq_along(tick)) {
    ok <- is.finite(Y[, j]); n <- sum(ok); if (n < MIN_TEST) next
    A <- X[ok, ]; yy <- Y[ok, j]; b <- qr.solve(A, yy); e <- yy - A %*% b
    ssr <- sum(e^2); sst <- sum((yy - mean(yy))^2); rmse <- sqrt(ssr / (n - ncol(A)))
    res[j, ] <- c(b[1] * 252 * 100, 1 - ssr / sst, rmse, sd(yy), b[1] / rmse, n)
  }
  res
}
fits <- setNames(lapply(2011:2026, year_fit), 2011:2026)
rows <- lapply(2012:2026, function(y) {
  a <- fits[[as.character(y - 1)]]; b <- fits[[as.character(y)]]
  keep <- !is.na(a[, "alpha"]) & a[, "n"] >= MIN_PRIOR & !is.na(b[, "alpha"])
  data.frame(year = y, fund = tick[keep], alpha_p = a[keep, "alpha"], r2_p = a[keep, "r2"], rmse_p = a[keep, "rmse"],
             sdr_p = a[keep, "sdr"], ir_p = a[keep, "ir"], alpha = b[keep, "alpha"], ir = b[keep, "ir"])
})
fy <- do.call(rbind, rows)
lo <- quantile(fy$r2_p, 0.01); hi <- quantile(fy$r2_p, 0.99)       # symmetric 1% trim (type-7 quantiles, as pandas)
fy <- fy[fy$r2_p >= lo & fy$r2_p <= hi, ]; rownames(fy) <- NULL
fy$tr2 <- log(sqrt(fy$r2_p) / (1 - sqrt(fy$r2_p)))
cat_map <- fread(file.path(D, "yahoo_us_equity_funds_deduped.csv"))
fy$style <- cat_map$yahoo_category[match(fy$fund, cat_map$ticker)]
fwrite(fy, "data/ag_r_fund_years.csv")

bin_of <- function(rank, n) ceiling(rank * 5 / n)
sorted <- function(df, dep, lag, independent) {
  df$qr <- 0L; df$qa <- 0L
  for (y in unique(df$year)) {
    i <- which(df$year == y); qr <- bin_of(rank(df$r2_p[i], ties.method = "first"), length(i)); df$qr[i] <- qr
    if (independent) df$qa[i] <- bin_of(rank(df[[lag]][i], ties.method = "first"), length(i)) else
      for (q in 1:5) { m <- i[qr == q]; df$qa[m] <- bin_of(rank(df[[lag]][m], ties.method = "first"), length(m)) }
  }
  out <- expand.grid(alpha_q = 1:5, r2_q = 1:5)
  s <- function(x) c(mean(x), mean(x) / (sd(x) / sqrt(length(x))), length(x))
  v <- t(mapply(function(a, q) s(df[[dep]][df$qa == a & df$qr == q]), out$alpha_q, out$r2_q))
  out$mean <- v[, 1]; out$t <- v[, 2]; out$n <- v[, 3]; out
}
for (nm in c("dep", "ind")) for (dv in c("alpha", "ir")) {
  lag <- if (dv == "alpha") "alpha_p" else "ir_p"
  fwrite(sorted(fy, dv, lag, nm == "ind"), sprintf("data/ag_r_table%s_%s.csv", if (nm == "dep") "8" else "9", dv))
}
# Table II/III style panel regressions: year and style dummies, standard errors clustered by fund
panel <- function(dep, cols, df) {
  f <- as.formula(paste(dep, "~", paste(c(cols, "factor(year)", "factor(style)"), collapse = " + ")))
  m <- lm(f, df); V <- vcovCL(m, cluster = df$fund, type = "HC1")     # CR1 small-sample factor, as in the Python code
  data.frame(var = cols, coef = coef(m)[cols], t = coef(m)[cols] / sqrt(diag(V))[cols], n = nrow(df))
}
tabs <- rbind(cbind(spec = "alpha_tr2", panel("alpha", "tr2", fy)), cbind(spec = "alpha_comp_lag", panel("alpha", c("rmse_p", "sdr_p", "alpha_p"), fy)),
              cbind(spec = "ir_tr2_lag", panel("ir", c("tr2", "ir_p"), fy)))
fwrite(tabs, "data/ag_r_regressions.csv"); print(tabs)
cat("fund-years", nrow(fy), "funds", length(unique(fy$fund)), "\n")
