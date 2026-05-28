"""Build per-index ticker lists as parquet files.

Writes one file per index to ``data/universe/{index}.parquet`` with columns
``ticker, name, cik``. These parquets are committed to the repo so the
dashboard's ticker dropdown loads instantly with zero network calls on cold
start (e.g. on Streamlit Community Cloud). Re-run periodically to keep the
lists in sync with index reconstitutions.

Usage:
    python scripts/build_universe_parquets.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow ``python scripts/build_universe_parquets.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading_droppers import config, universe


# Per-index fetchers in dashboard display order.
SOURCES = [
    ("sp400", universe.fetch_sp400),
    ("sp500", universe.fetch_sp500),
    ("sp600", universe.fetch_sp600),
    ("nasdaq100", universe.fetch_nasdaq100),
    ("russell2000", universe.fetch_russell2000),
]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("build_universe_parquets")
    out_dir = config.DATA_DIR / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching SEC ticker -> CIK map...")
    try:
        cik_map = universe.fetch_sec_ticker_cik_map()
    except Exception as e:
        log.warning("SEC ticker map fetch failed (%s); parquets will lack CIKs", e)
        cik_map = {}

    summary: list[tuple[str, int, int, int]] = []
    for name, fetcher in SOURCES:
        try:
            df = fetcher()
        except Exception as e:
            log.warning("%-12s FAIL  %s", name, e)
            continue
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset="ticker")
        df["cik"] = df["ticker"].map(cik_map)
        df = df[["ticker", "name", "cik"]].sort_values("ticker").reset_index(drop=True)
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        with_cik = int(df["cik"].notna().sum())
        size_kb = path.stat().st_size / 1024
        log.info("%-12s %4d tickers (%4d w/ CIK)  -> %s  [%.1f KB]",
                 name, len(df), with_cik, path.name, size_kb)
        summary.append((name, len(df), with_cik, int(size_kb)))

    print()
    print(f"{'index':<12} {'tickers':>8} {'with_cik':>10} {'size_kb':>9}")
    for name, n, with_cik, kb in summary:
        print(f"{name:<12} {n:>8} {with_cik:>10} {kb:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
