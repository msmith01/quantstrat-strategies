"""Record exactly which data produced the article: retrieval date, SHA-256 of every input file, the factor
file's header (which states the CRSP vintage), and a copy of the raw Fama-French zip as downloaded.
Run after regen_results.py:  python python/archive_data.py   -> data/MANIFEST.json, data/raw/"""
import datetime, hashlib, io, json, urllib.request, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

raw = urllib.request.urlopen(urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})).read()
(D / "raw").mkdir(exist_ok=True); (D / "raw" / "F-F_Research_Data_Factors_daily_CSV.zip").write_bytes(raw)
text = zipfile.ZipFile(io.BytesIO(raw)).read("F-F_Research_Data_Factors_daily.csv").decode("latin-1")
header = " ".join(l.strip() for l in text.splitlines()[:4] if l.strip())
# does the raw file still equal the factors the strategy used? (the library revises its history)
saved = pd.read_csv(D / "factors_python.csv", index_col=0, parse_dates=True)
rows = [l.split(",") for l in text.splitlines() if l[:8].isdigit()]
now = pd.DataFrame(rows, columns=["date", "MKT", "SMB", "HML", "RF"]).astype(float)
now.index = pd.to_datetime(now.pop("date").astype(int).astype(str)); now = now / 100
same = bool(len(now) == len(saved) and (now.reindex(saved.index) - saved).abs().max().max() < 1e-12)
(D / "MANIFEST.json").write_text(json.dumps({
    "retrieved": datetime.date.today().isoformat(),
    "factor_source": FF_URL, "factor_file_header": header, "factor_zip_sha256": hashlib.sha256(raw).hexdigest(),
    "raw_factors_equal_used_factors": same,
    "files_sha256": {f: sha(D / f) for f in ("prices_python.csv", "prices_r.csv", "factors_python.csv", "factors_r.csv")},
    "prices_source": "Yahoo Finance adjusted closes via yfinance (Python) and quantmod (R); adjusted prices can be revised"}, indent=2))
print("manifest written; raw factor file identical to the one used:", same)
