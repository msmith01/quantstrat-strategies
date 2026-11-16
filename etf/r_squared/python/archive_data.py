"""Writes data/MANIFEST.json (checksums of the raw inputs) and data/versions.txt (R18: every input is archived and identified)."""
import hashlib
import json
import platform
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import yfinance

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = {"retrieved": "16 November 2026", "yfinance": yfinance.__version__, "files": {}}
for z in sorted((ROOT / "data/raw").glob("*.zip")):
    name = zipfile.ZipFile(z).namelist()[0]
    first = zipfile.ZipFile(z).read(name).decode("latin1").splitlines()[0]
    man["files"][str(z.relative_to(ROOT))] = {"sha256": sha(z), "bytes": z.stat().st_size, "header": first}
    if "Factors_daily" in z.name:
        man["factor_file_header"] = first
for f in ("data/funds/yahoo_us_equity_funds_universe.csv", "data/funds/yahoo_us_equity_funds_deduped.csv", "data/funds/share_class_groups.csv",
          "data/funds/fund_prices.parquet", "data/funds/fund_info.json", "data/funds/excluded_bad_prices.csv"):
    if (ROOT / f).exists():
        man["files"][f] = {"sha256": sha(ROOT / f), "bytes": (ROOT / f).stat().st_size}
json.dump(man, open(ROOT / "data/MANIFEST.json", "w"), indent=1)
(ROOT / "data/versions.txt").write_text(f"Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, statsmodels {statsmodels.__version__}, yfinance {yfinance.__version__}; R 4.5, data.table, sandwich")
print(len(man["files"]), "files recorded")
