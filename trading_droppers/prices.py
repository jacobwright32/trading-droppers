"""Price fetching via yfinance, cached into the ``prices`` table.

Two entry points:

* :func:`get_prices(ticker)` - lazy fetch with cache. Used by the dashboard.
* :func:`bulk_fetch_and_store(tickers)` - batched download for the screening
  pipeline. yfinance is happy with ~100-200 tickers per call.

Prices are split/dividend-adjusted (``auto_adjust=True``) so the weekly line
stays continuous across corporate actions.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf

from . import config, db

log = logging.getLogger(__name__)


_KEEP_COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def _flatten_one(sub: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten a single-ticker wide frame to long form with a stable 'date' column."""
    sub = sub.copy()
    sub.columns = [str(c).lower() for c in sub.columns]
    # yfinance leaves the DatetimeIndex unnamed in 0.2.40+, so reset_index()
    # produces a column called 'index'. Older versions name it 'Date'.
    sub = sub.reset_index().rename(columns={"index": "date", "Date": "date"})
    sub.columns = [str(c).lower() for c in sub.columns]
    sub["ticker"] = ticker
    return sub


def _wide_to_long(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Convert yfinance output (single-ticker wide OR multi-ticker MultiIndex) to long form."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_KEEP_COLS)

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        # With group_by="ticker", columns are (ticker, field). yfinance has
        # historically reversed this; be defensive.
        if any(t in level0 for t in tickers):
            for t in tickers:
                if t not in level0:
                    continue
                frames.append(_flatten_one(raw[t], t))
        else:
            level1 = raw.columns.get_level_values(1).unique().tolist()
            for t in tickers:
                if t not in level1:
                    continue
                frames.append(_flatten_one(raw.xs(t, axis=1, level=1), t))
    else:
        frames.append(_flatten_one(raw, tickers[0]))

    if not frames:
        return pd.DataFrame(columns=_KEEP_COLS)

    out = pd.concat(frames, ignore_index=True)
    for c in _KEEP_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[_KEEP_COLS].dropna(subset=["date", "close"])
    return out


def fetch_prices(
    tickers: Iterable[str],
    period: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV from yfinance and return a long-form DataFrame."""
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame(columns=_KEEP_COLS)

    raw = yf.download(
        tickers=" ".join(tickers),
        period=period or config.PRICE_PERIOD,
        interval=interval or config.PRICE_INTERVAL,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return _wide_to_long(raw, tickers)


def get_prices(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Return weekly prices for ``ticker``, downloading on miss (or when ``refresh``)."""
    if not refresh:
        cached = db.load_prices(ticker)
        if not cached.empty:
            return cached
    fresh = fetch_prices([ticker])
    if not fresh.empty:
        db.upsert_prices(fresh)
    return db.load_prices(ticker)


def bulk_fetch_and_store(
    tickers: Iterable[str],
    batch_size: int | None = None,
    period: str | None = None,
    interval: str | None = None,
) -> int:
    """Download prices for ``tickers`` in batches, persist, return total rows written."""
    tickers = list(tickers)
    batch_size = batch_size or config.YF_BATCH_SIZE
    total = 0
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        try:
            df = fetch_prices(chunk, period=period, interval=interval)
        except Exception as e:
            log.warning("Price batch %d failed: %s", i // batch_size, e)
            continue
        if not df.empty:
            total += db.upsert_prices(df)
        log.info(
            "prices batch %d (%d tickers): %d rows",
            i // batch_size,
            len(chunk),
            len(df),
        )
    return total
