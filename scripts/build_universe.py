"""Build the scan universe and persist it to the SQLite DB.

Run once before launching the dashboard so the ticker picker has something to
show. Re-run any time the index constituents change (quarterly-ish).

Usage:
    python scripts/build_universe.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow ``python scripts/build_universe.py`` from the repo root without installs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_droppers import db, universe


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    db.init_schema()
    df = universe.build_universe()
    by_index = {c: int(df[c].sum()) for c in db.INDEX_COLUMNS if c in df.columns}
    with_cik = int(df["cik"].notna().sum()) if "cik" in df.columns else 0
    print()
    print(f"Universe rows: {len(df)}")
    print(f"With SEC CIK : {with_cik}")
    for idx, n in by_index.items():
        print(f"  {idx:<12} {n:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
