"""Builds the candidate US mutual-fund list from Yahoo's screener (live funds only: survivorship bias, disclosed)."""
import json, sys, time
import yfinance as yf
CATS = ["Large Blend", "Large Growth", "Large Value", "Mid-Cap Blend", "Mid-Cap Growth", "Mid-Cap Value",
        "Small Blend", "Small Growth", "Small Value"]
out = {}
for c in CATS:
    off, n = 0, 0
    while True:
        q = yf.FundQuery("eq", ["categoryname", c])
        try:
            r = yf.screen(q, size=250, offset=off, sortField="eodprice", sortAsc=False)
        except Exception as e:
            print(c, off, "ERR", e); break
        qs = r.get("quotes", [])
        for x in qs:
            out[x["symbol"]] = {"name": x.get("longName"), "cat": c, "cur": x.get("currency"), "ex": x.get("exchange"),
                                "first": x.get("firstTradeDateMilliseconds")}
        n += len(qs); off += 250
        if len(qs) < 250 or off >= r.get("total", 0): break
        time.sleep(0.3)
    print(c, n, "total", r.get("total"), flush=True)
json.dump(out, open("data/funds/universe_raw.json", "w"))
print(len(out))
