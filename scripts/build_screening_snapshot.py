"""Build the de-rating screening snapshot.

For each ticker (across the selected index parquets) with a SEC CIK, downloads
prices and fundamentals (cached in SQLite from prior runs) and derives one row
of screening metrics. Writes the result to
``data/screening/snapshot.parquet`` for the Streamlit Screener page to read.

The snapshot is what makes the Screener page *instant* on Streamlit Cloud -
re-running this script is the only thing that touches yfinance / SEC EDGAR
across many tickers at once.

Usage:
    python scripts/build_screening_snapshot.py                          # default
    python scripts/build_screening_snapshot.py --indices sp500 sp400    # subset
    python scripts/build_screening_snapshot.py --limit 50               # smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow ``python scripts/...`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading_droppers import config, db
from trading_droppers import fundamentals as fund_mod
from trading_droppers import prices as price_mod


UNIVERSE_DIR = config.DATA_DIR / "universe"
OUTPUT_PATH = config.DATA_DIR / "screening" / "snapshot.parquet"

DEFAULT_INDICES = ("sp500", "nasdaq100")


def _load_universe(indices: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx in indices:
        path = UNIVERSE_DIR / f"{idx}.parquet"
        if not path.exists():
            logging.warning("Missing universe parquet %s", path)
            continue
        df = pd.read_parquet(path)
        df["index"] = idx
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["cik"]).drop_duplicates(subset="ticker", keep="first")
    return out.reset_index(drop=True)


def _metrics_for(
    ticker: str,
    name: str,
    cik: str,
    index: str,
    prices: pd.DataFrame,
    ttm: pd.DataFrame | None,
) -> dict | None:
    """Compute one snapshot row. Returns ``None`` if there isn't enough data."""
    if prices is None or prices.empty:
        return None

    px = prices.sort_values("date").reset_index(drop=True)
    last_close = float(px["close"].iloc[-1])
    peak = float(px["close"].max())
    drawdown = (last_close / peak - 1.0) if peak > 0 else 0.0

    row: dict = {
        "ticker": ticker,
        "name": name,
        "cik": cik,
        "index": index,
        "last_close": last_close,
        "peak_5y": peak,
        "drawdown": drawdown,
        "revenue_ttm": None,
        "revenue_yoy": None,
        "net_income_ttm": None,
        "is_profitable": None,
        "eps_ttm": None,
        "pe_ttm": None,
        "ps_now": None,
        "ps_1y_ago": None,
        "ps_compression": None,
    }

    if ttm is None or ttm.empty:
        return row

    # Revenue YoY: latest TTM vs four quarters ago.
    rev = ttm["revenue_ttm"].dropna()
    if len(rev) >= 5:
        rev_now = float(rev.iloc[-1])
        rev_prior = float(rev.iloc[-5])
        if rev_prior > 0:
            row["revenue_ttm"] = rev_now
            row["revenue_yoy"] = rev_now / rev_prior - 1.0

    # Profitability and EPS.
    ni = ttm["net_income_ttm"].dropna()
    if len(ni) > 0:
        ni_now = float(ni.iloc[-1])
        row["net_income_ttm"] = ni_now
        row["is_profitable"] = ni_now > 0
    eps = ttm["eps_ttm"].dropna()
    if len(eps) > 0:
        row["eps_ttm"] = float(eps.iloc[-1])
        if row["eps_ttm"] and row["eps_ttm"] > 0:
            row["pe_ttm"] = last_close / row["eps_ttm"]

    # P/S now vs ~1 year ago. Compute rev_per_share series, do an as-of join
    # against weekly prices, then read off the value at the latest tick and
    # the tick closest to 52 weeks before.
    if "shares_ttm" in ttm.columns and "revenue_ttm" in ttm.columns:
        f = ttm.dropna(subset=["revenue_ttm", "shares_ttm"]).copy()
        f = f[f["shares_ttm"] > 0]
        if not f.empty:
            f["rev_per_share"] = f["revenue_ttm"] / f["shares_ttm"]
            f = f.sort_values("period_end")[["period_end", "rev_per_share"]]
            merged = pd.merge_asof(
                px.sort_values("date"),
                f,
                left_on="date",
                right_on="period_end",
                direction="backward",
            )
            merged = merged.dropna(subset=["rev_per_share"])
            if not merged.empty:
                merged["ps"] = merged["close"] / merged["rev_per_share"]
                ps_now = float(merged["ps"].iloc[-1])
                row["ps_now"] = ps_now
                target = merged["date"].iloc[-1] - pd.Timedelta(weeks=52)
                older = merged[merged["date"] <= target]
                if not older.empty:
                    ps_then = float(older["ps"].iloc[-1])
                    row["ps_1y_ago"] = ps_then
                    if ps_then > 0:
                        row["ps_compression"] = 1.0 - ps_now / ps_then

    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--indices",
        nargs="+",
        default=list(DEFAULT_INDICES),
        help=f"Index parquets to include (default: {' '.join(DEFAULT_INDICES)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N tickers (smoke testing).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("snapshot")
    db.init_schema()

    universe = _load_universe(tuple(args.indices))
    if universe.empty:
        log.error("Universe is empty. Did you run scripts/build_universe_parquets.py?")
        return 1
    if args.limit:
        universe = universe.head(args.limit)
    log.info("Screening %d tickers from %s", len(universe), ", ".join(args.indices))

    # Warm the price cache in batches before the per-ticker loop. yfinance
    # handles 100 tickers per call gracefully; one-at-a-time is ~100x slower.
    all_tickers = universe["ticker"].tolist()
    missing_prices = [t for t in all_tickers if db.load_prices(t).empty]
    if missing_prices:
        log.info(
            "Warming price cache: %d new tickers (batched yfinance fetch)...",
            len(missing_prices),
        )
        price_mod.bulk_fetch_and_store(missing_prices)
        log.info("Price cache warmed.")

    rows: list[dict] = []
    started = time.monotonic()
    for i, t in enumerate(universe.itertuples(index=False), 1):
        ticker, name, cik, index_name = t.ticker, t.name, t.cik, t.index
        try:
            px = price_mod.get_prices(ticker)
            if px is None or px.empty:
                continue
            fund = fund_mod.get_fundamentals(ticker, cik=cik)
            ttm = fund_mod.compute_ttm(fund) if fund is not None and not fund.empty else None
            metrics = _metrics_for(ticker, name, cik, index_name, px, ttm)
            if metrics:
                rows.append(metrics)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", ticker, e)
        if i % 25 == 0 or i == len(universe):
            rate = i / (time.monotonic() - started)
            log.info("%4d / %4d (%.1f tickers/s)", i, len(universe), rate)

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")
    if not out.empty and "drawdown" in out and "revenue_yoy" in out:
        candidates = out[
            (out["drawdown"] <= -0.30) & (out["revenue_yoy"] >= 0.08)
        ].copy()
        candidates["score"] = candidates["drawdown"].abs() * candidates["revenue_yoy"]
        candidates = candidates.sort_values("score", ascending=False).head(10)
        if not candidates.empty:
            print(f"\nTop 10 de-rating candidates (default thresholds):")
            for _, r in candidates.iterrows():
                print(
                    f"  {r['ticker']:<6} {r['name'][:30]:<30}"
                    f"  drawdown {r['drawdown']:>6.0%}"
                    f"  rev YoY {r['revenue_yoy']:>+6.0%}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
