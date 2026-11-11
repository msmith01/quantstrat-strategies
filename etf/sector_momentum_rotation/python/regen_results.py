"""Pipeline helper: run the published scripts with their optional SAVE block enabled,
so data/ gets the files that verify.py, make_charts.py and build_post.py read.
The scripts themselves are not modified.
Run from the strategy folder:  python python/regen_results.py"""
import re
import subprocess
import sys
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def enable_save(text):
    out, on = [], False
    for line in text.splitlines():
        if "SAVE-END" in line:
            on = False
        if on:
            line = re.sub(r"^(\s*)# ?", r"\1", line)
        if "SAVE-BEGIN" in line:
            on = True
        out.append(line)
    return "\n".join(out) + "\n"


for rel, cmd in (("python/sector_momentum.py", [sys.executable]), ("r/sector_momentum.R", ["Rscript"])):
    src = ROOT / rel
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / src.name
        tmp.write_text(enable_save(src.read_text(encoding="utf-8")), encoding="utf-8")
        subprocess.run(cmd + [str(tmp)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, env={**os.environ, "MPLBACKEND": "Agg"})
        (ROOT / "Rplots.pdf").unlink(missing_ok=True)
    print("ran", rel)
