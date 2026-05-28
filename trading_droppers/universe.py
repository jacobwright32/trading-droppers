"""Build the scan universe.

Pulls index constituents from each canonical source, normalises tickers to the
yfinance dash convention (``BRK-B`` not ``BRK.B``), attaches each name's SEC
CIK, and persists the merged result to the ``universe`` table.

Sources:
    * S&P 400 / 500 / 600   - Wikipedia "List of ..." tables
    * Nasdaq-100            - Wikipedia "Nasdaq-100" components table
    * Russell 2000          - iShares IWM ETF holdings CSV
    * ticker -> CIK         - SEC ``company_tickers.json``
"""
from __future__ import annotations

import io
import logging
import re
from typing import Callable

import pandas as pd
import requests

from . import config, db

log = logging.getLogger(__name__)


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
IWM_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
# iShares now firewalls direct CSV downloads (serves their HTML splash page).
# Fallback: a maintained ticker+name mirror on GitHub. The list lags iShares by
# whatever the maintainer's update cadence is, but it's a static endpoint we
# can rely on for the small-cap universe.
RUSSELL2000_FALLBACK_URL = (
    "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/"
    "russell_2000_components.csv"
)

_WIKI_UA = "Mozilla/5.0 (compatible; trading-droppers/0.1)"
_VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9\-]{0,9}$")


def _norm(t: object) -> str:
    """Normalise a ticker symbol: upper-case, strip, dots->dashes (yfinance style)."""
    if t is None:
        return ""
    s = str(t).strip().upper().replace(".", "-")
    # Wikipedia sometimes leaves trailing footnote markers like 'AAPL[a]'
    s = re.sub(r"\[.*?\]$", "", s).strip()
    return s


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers or {"User-Agent": _WIKI_UA}, timeout=timeout)
    r.raise_for_status()
    return r


def _pick_table(tables: list[pd.DataFrame], candidates: tuple[str, ...]) -> pd.DataFrame:
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(c in cols for c in candidates):
            return t
    raise RuntimeError(f"No table found with any of columns {candidates}")


# --- per-index fetchers ----------------------------------------------------

def fetch_sp500() -> pd.DataFrame:
    r = _get(SP500_URL)
    tables = pd.read_html(io.StringIO(r.text))
    df = _pick_table(tables, ("Symbol",))
    return pd.DataFrame(
        {
            "ticker": df["Symbol"].map(_norm),
            "name": df.get("Security", df["Symbol"]).astype(str),
        }
    )


def _sp_midsmall(url: str) -> pd.DataFrame:
    r = _get(url)
    tables = pd.read_html(io.StringIO(r.text))
    df = _pick_table(tables, ("Symbol", "Ticker symbol"))
    sym_col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
    name_col = next(
        (c for c in ("Security", "Company", "Name") if c in df.columns), sym_col
    )
    return pd.DataFrame(
        {"ticker": df[sym_col].map(_norm), "name": df[name_col].astype(str)}
    )


def fetch_sp400() -> pd.DataFrame:
    return _sp_midsmall(SP400_URL)


def fetch_sp600() -> pd.DataFrame:
    return _sp_midsmall(SP600_URL)


def fetch_nasdaq100() -> pd.DataFrame:
    r = _get(NASDAQ100_URL)
    tables = pd.read_html(io.StringIO(r.text))
    df = _pick_table(tables, ("Ticker", "Symbol"))
    sym_col = "Ticker" if "Ticker" in df.columns else "Symbol"
    name_col = next(
        (c for c in ("Company", "Security", "Name") if c in df.columns), sym_col
    )
    return pd.DataFrame(
        {"ticker": df[sym_col].map(_norm), "name": df[name_col].astype(str)}
    )


def _parse_ishares_csv(text: str) -> pd.DataFrame | None:
    """Parse an iShares holdings CSV. Returns None if the response wasn't a real CSV."""
    text = text.lstrip("﻿")  # strip BOM if present
    if text.lstrip().lower().startswith("<!doctype") or "<html" in text[:200].lower():
        return None  # iShares served the HTML splash page instead of the CSV
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.lower().lstrip().startswith("ticker,")),
        None,
    )
    if header_idx is None:
        return None
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body))
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().str.lower() == "equity"]
    df = df.dropna(subset=["Ticker"])
    return pd.DataFrame(
        {
            "ticker": df["Ticker"].map(_norm),
            "name": df["Name"].astype(str) if "Name" in df.columns else df["Ticker"],
        }
    )


def fetch_russell2000() -> pd.DataFrame:
    """Russell 2000 constituents.

    Primary: the iShares IWM holdings CSV (canonical, refreshed daily).
    Fallback: a maintained GitHub mirror (ticker, name) when iShares firewalls
    us with their HTML splash page - which is the common case at time of writing.
    """
    try:
        text = _get(IWM_URL).text
        parsed = _parse_ishares_csv(text)
        if parsed is not None and not parsed.empty:
            return parsed
        log.info("iShares IWM CSV looked like HTML; using GitHub fallback")
    except Exception as e:
        log.info("iShares IWM CSV fetch failed (%s); using GitHub fallback", e)

    r = _get(RUSSELL2000_FALLBACK_URL)
    df = pd.read_csv(io.StringIO(r.text))
    tick_col = next((c for c in df.columns if c.lower().strip() == "ticker"), df.columns[0])
    name_col = next(
        (c for c in df.columns if c.lower().strip() in ("name", "company", "security")),
        tick_col,
    )
    return pd.DataFrame(
        {"ticker": df[tick_col].map(_norm), "name": df[name_col].astype(str)}
    )


# --- SEC ticker -> CIK -----------------------------------------------------

def fetch_sec_ticker_cik_map() -> dict[str, str]:
    """Return ``{ticker -> 10-digit zero-padded CIK}``.

    SEC requires a descriptive User-Agent (set via ``SEC_USER_AGENT`` in .env).
    """
    r = requests.get(
        config.SEC_TICKERS_URL,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    out: dict[str, str] = {}
    # company_tickers.json is shaped as {"0": {...}, "1": {...}, ...}
    for entry in data.values():
        ticker = _norm(entry.get("ticker"))
        cik = int(entry.get("cik_str") or 0)
        if ticker and cik:
            out[ticker] = f"{cik:010d}"
    return out


# --- orchestrator ----------------------------------------------------------

_SOURCES: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("sp500", fetch_sp500),
    ("sp400", fetch_sp400),
    ("sp600", fetch_sp600),
    ("nasdaq100", fetch_nasdaq100),
    ("russell2000", fetch_russell2000),
]


def build_universe() -> pd.DataFrame:
    """Fetch every index, merge, attach CIKs, persist. Returns the merged frame."""
    by_ticker: dict[str, dict] = {}

    for index_name, fetcher in _SOURCES:
        try:
            df = fetcher()
        except Exception as e:  # one bad source shouldn't kill the whole build
            log.warning("Failed to fetch %s: %s", index_name, e)
            continue
        log.info("%-12s %4d tickers", index_name, len(df))
        for _, row in df.iterrows():
            t = row["ticker"]
            if not t or not _VALID_TICKER.match(t):
                continue
            cur = by_ticker.setdefault(
                t,
                {
                    "ticker": t,
                    "name": None,
                    "cik": None,
                    **{c: 0 for c in db.INDEX_COLUMNS},
                },
            )
            cur[index_name] = 1
            if not cur["name"] and isinstance(row.get("name"), str):
                cur["name"] = row["name"]

    # Best-effort CIK enrichment.
    try:
        cik_map = fetch_sec_ticker_cik_map()
        for t, row in by_ticker.items():
            row["cik"] = cik_map.get(t)
        matched = sum(1 for r in by_ticker.values() if r["cik"])
        log.info("Matched %d / %d tickers to a SEC CIK", matched, len(by_ticker))
    except Exception as e:
        log.warning("SEC ticker->CIK lookup failed: %s", e)

    db.init_schema()
    rows = list(by_ticker.values())
    db.upsert_universe(rows)
    log.info("Universe persisted: %d unique tickers", len(rows))
    return pd.DataFrame(rows)
