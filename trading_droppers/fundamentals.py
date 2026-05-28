"""SEC EDGAR XBRL fundamentals.

For a given CIK, fetch ``companyfacts.json``, extract per-quarter values for the
income-statement, balance-sheet, and share-count concepts we care about, and
persist into the ``fundamentals`` table. The dashboard then computes TTM
aggregates on read via :func:`compute_ttm`.

Quirks we handle:

* **ASC 606 transition.** Some companies report under ``Revenues``, others under
  ``RevenueFromContractWithCustomerExcludingAssessedTax``, and the *same*
  company often switches from one to the other in 2018. We walk a priority list
  of concepts and stitch them on ``period_end``.
* **Missing Q4.** Most 10-Ks report the *full fiscal year* but not the standalone
  Q4 quarter. We derive ``Q4 = FY - (Q1 + Q2 + Q3)`` where possible.
* **Restatements.** When a 10-K/A or later 10-Q restates an earlier period we
  prefer the latest filing.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Iterable

import pandas as pd
import requests

from . import config, db

log = logging.getLogger(__name__)


# Walked in order; first concept yielding a value for a given (period_end) wins.
REVENUE_CONCEPTS: tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
NET_INCOME_CONCEPTS: tuple[str, ...] = ("NetIncomeLoss",)
ASSETS_CONCEPTS: tuple[str, ...] = ("Assets",)
LIAB_CONCEPTS: tuple[str, ...] = ("Liabilities",)
EQUITY_CONCEPTS: tuple[str, ...] = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
DIL_SHARES_CONCEPTS: tuple[str, ...] = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)


# Period-length windows in days.
_QTR_MIN, _QTR_MAX = 80, 100      # 3-month period
_FY_MIN, _FY_MAX = 350, 380       # full fiscal year


_session = requests.Session()
_session.headers.update({"User-Agent": config.SEC_USER_AGENT})

# Single-threaded rate limit. SEC allows 10 req/s; we stay under.
_last_request_at = 0.0
_min_interval = 1.0 / max(1, config.SEC_RATE_LIMIT_PER_SEC)


def _throttled_get(url: str) -> requests.Response:
    global _last_request_at
    wait = _min_interval - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    r = _session.get(url, timeout=30)
    _last_request_at = time.monotonic()
    r.raise_for_status()
    return r


def fetch_companyfacts(cik: str) -> dict | None:
    """Fetch the SEC ``companyfacts`` payload for a CIK. Returns ``None`` on 404."""
    if not cik:
        return None
    cik_int = int(str(cik).lstrip("0") or "0")
    if cik_int == 0:
        return None
    url = config.SEC_COMPANYFACTS_URL.format(cik=cik_int)
    try:
        r = _throttled_get(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            log.warning("SEC companyfacts 404 for CIK %s", cik)
            return None
        raise
    return r.json()


# --- low-level fact extraction --------------------------------------------

def _entries(facts_root: dict, concept: str, unit: str) -> list[dict]:
    node = ((facts_root.get("us-gaap") or {}).get(concept)) or {}
    return (node.get("units") or {}).get(unit) or []


def _period_days(entry: dict) -> int | None:
    s, e = entry.get("start"), entry.get("end")
    if not s or not e:
        return None
    try:
        return (pd.Timestamp(e) - pd.Timestamp(s)).days
    except Exception:
        return None


def _is_quarterly(entry: dict) -> bool:
    d = _period_days(entry)
    return d is not None and _QTR_MIN <= d <= _QTR_MAX


def _is_fy(entry: dict) -> bool:
    d = _period_days(entry)
    return d is not None and _FY_MIN <= d <= _FY_MAX


def _filed_lag_days(entry: dict) -> int:
    """Days between period end and the filing date. Smaller = original report."""
    try:
        return (pd.Timestamp(entry.get("filed")) - pd.Timestamp(entry.get("end"))).days
    except Exception:
        return 10**6


def _series_by_period_end(
    facts_root: dict,
    concepts: tuple[str, ...],
    predicate,
    unit: str,
) -> dict[str, dict]:
    """Return ``{period_end -> {val, fp, fy, form, period_start, concept}}``.

    Walks ``concepts`` in priority order; once a period_end is filled, lower-
    priority concepts don't overwrite it. For multiple entries of the same
    (concept, period_end), prefer the one filed soonest after the period end -
    i.e. the original 10-Q/10-K rather than a comparative quote in a later
    filing. This is important because comparative quotes occasionally carry
    XBRL filer errors that change the value (we saw NOW's 2026 filings re-tag
    prior-period diluted shares at 5x scale).
    """
    out: dict[str, dict] = {}
    for concept in concepts:
        entries = [e for e in _entries(facts_root, concept, unit) if predicate(e)]
        entries.sort(key=lambda e: (e.get("end", ""), _filed_lag_days(e)))
        seen: set[str] = set()
        for e in entries:
            end = e.get("end")
            if not end or end in seen:
                continue
            seen.add(end)
            if end in out:
                continue
            out[end] = {
                "val": e.get("val"),
                "fp": e.get("fp"),
                "fy": e.get("fy"),
                "form": e.get("form"),
                "period_start": e.get("start"),
                "concept": concept,
            }
    return out


def _instant_by_period_end(
    facts_root: dict,
    concepts: tuple[str, ...],
    unit: str = "USD",
) -> dict[str, float]:
    """Point-in-time values keyed by ``period_end`` (no period_start).

    Selection rule mirrors :func:`_series_by_period_end`: prefer the entry
    filed soonest after period_end (the original 10-Q/10-K).
    """
    out: dict[str, float] = {}
    for concept in concepts:
        entries = [
            e for e in _entries(facts_root, concept, unit)
            if not e.get("start") or e.get("start") == e.get("end")
        ]
        entries.sort(key=lambda e: (e.get("end", ""), _filed_lag_days(e)))
        seen: set[str] = set()
        for e in entries:
            end = e.get("end")
            if not end or end in seen:
                continue
            seen.add(end)
            if end not in out:
                out[end] = e.get("val")
    return out


# --- Q4 derivation ---------------------------------------------------------

def _derive_q4_quarters(
    quarterly: dict[str, dict],
    annual: dict[str, dict],
) -> dict[str, dict]:
    """For each annual FY, if Q4 (the quarter ending at the same date) is
    absent from ``quarterly``, derive it as FY - sum(other 3 quarters in the
    fiscal year).
    """
    out = dict(quarterly)
    for fy_end, fy_entry in annual.items():
        if fy_end in out:
            continue  # we already have a real Q4 with that end date
        try:
            fy_end_ts = pd.Timestamp(fy_end)
        except Exception:
            continue
        # Quarters within (fy_end - ~365d, fy_end]
        window_start = fy_end_ts - pd.Timedelta(days=370)
        in_window = [
            (e, q) for e, q in quarterly.items()
            if window_start < pd.Timestamp(e) < fy_end_ts
            and q.get("val") is not None
        ]
        if len(in_window) != 3:
            continue
        fy_val = fy_entry.get("val")
        if fy_val is None:
            continue
        q_sum = sum(q["val"] for _, q in in_window)
        out[fy_end] = {
            "val": fy_val - q_sum,
            "fp": "Q4",
            "fy": fy_entry.get("fy"),
            "form": fy_entry.get("form"),
            "period_start": (fy_end_ts - pd.Timedelta(days=92)).strftime("%Y-%m-%d"),
            "concept": fy_entry.get("concept") + "::derived",
        }
    return out


# --- top-level: companyfacts -> fundamentals rows -------------------------

def companyfacts_to_rows(ticker: str, cf: dict) -> list[dict]:
    """Turn a ``companyfacts`` payload into one row per fiscal quarter."""
    facts = cf.get("facts") or {}

    rev_q = _series_by_period_end(facts, REVENUE_CONCEPTS, _is_quarterly, "USD")
    rev_fy = _series_by_period_end(facts, REVENUE_CONCEPTS, _is_fy, "USD")
    rev_q = _derive_q4_quarters(rev_q, rev_fy)

    ni_q = _series_by_period_end(facts, NET_INCOME_CONCEPTS, _is_quarterly, "USD")
    ni_fy = _series_by_period_end(facts, NET_INCOME_CONCEPTS, _is_fy, "USD")
    ni_q = _derive_q4_quarters(ni_q, ni_fy)

    sh_q = _series_by_period_end(facts, DIL_SHARES_CONCEPTS, _is_quarterly, "shares")

    assets = _instant_by_period_end(facts, ASSETS_CONCEPTS)
    liab = _instant_by_period_end(facts, LIAB_CONCEPTS)
    eq = _instant_by_period_end(facts, EQUITY_CONCEPTS)

    # Union of period_ends across the duration series
    all_ends = sorted(set(rev_q) | set(ni_q) | set(sh_q))
    rows: list[dict] = []
    for end in all_ends:
        rv = rev_q.get(end, {})
        ni = ni_q.get(end, {})
        sh = sh_q.get(end, {})
        # Best metadata source: prefer revenue's, then net income, then shares.
        meta = rv or ni or sh
        rows.append(
            {
                "ticker": ticker,
                "period_end": end,
                "period_start": meta.get("period_start"),
                "fp": meta.get("fp") or "??",
                "fy": meta.get("fy"),
                "form": meta.get("form"),
                "revenue": rv.get("val"),
                "net_income": ni.get("val"),
                "assets": assets.get(end),
                "liabilities": liab.get(end),
                "equity": eq.get(end),
                "shares_diluted": sh.get("val"),
            }
        )
    return rows


def get_fundamentals(
    ticker: str,
    cik: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Lazy fetch + cache, then return the per-quarter fundamentals for ``ticker``."""
    if not refresh:
        cached = db.load_fundamentals(ticker)
        if not cached.empty:
            return cached
    cik = cik or db.get_cik(ticker)
    if not cik:
        log.info("%s: no CIK on file, skipping SEC fetch", ticker)
        return pd.DataFrame()
    cf = fetch_companyfacts(cik)
    if not cf:
        return pd.DataFrame()
    rows = companyfacts_to_rows(ticker, cf)
    if rows:
        db.upsert_fundamentals(rows)
    return db.load_fundamentals(ticker)


# --- TTM derivation --------------------------------------------------------

def _sanitize_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Mask out implausible share-count spikes from XBRL filer scaling errors.

    Heuristic: for each row, compare ``shares_diluted`` to the median of the
    prior eight non-null values. A 3x deviation is suspicious. We only mask
    when we're confident it's a one-off spike, not a sustained corporate
    action:

    * If the NEXT non-null value reverts most of the way back to the prior
      baseline, the spike is a single-point error -> mask it.
    * If there is no next value (we're at the latest row) and the deviation
      is even larger (>4x), mask it too.
    * Otherwise leave it - this preserves legitimate jumps like IPOs, large
      secondary offerings, and stock splits, where the elevated share count
      persists into subsequent quarters.

    NOW's 2026 filings re-tag prior-period diluted shares at 5x scale; without
    this safeguard the chart shows a fake P/E spike in the most recent quarter.
    """
    df = df.copy()
    if "shares_diluted" not in df.columns or df["shares_diluted"].notna().sum() < 3:
        return df
    sh = df["shares_diluted"].astype("float64").copy()
    n = len(sh)
    for i in range(n):
        v = sh.iloc[i]
        if pd.isna(v):
            continue
        prior = sh.iloc[max(0, i - 8): i].dropna()
        if len(prior) < 2:
            continue
        prior_med = float(prior.median())
        if prior_med <= 0:
            continue
        high = v > prior_med * 3
        low = v < prior_med / 3
        if not (high or low):
            continue
        forward = sh.iloc[i + 1:].dropna()
        if forward.empty:
            # Latest row: mask only if the deviation is large (>4x).
            if v > prior_med * 4 or v < prior_med / 4:
                sh.iloc[i] = pd.NA
            continue
        nxt = float(forward.iloc[0])
        if high and nxt < v / 2:
            sh.iloc[i] = pd.NA
        elif low and nxt > v * 2:
            sh.iloc[i] = pd.NA
    df["shares_diluted"] = sh
    return df


def compute_ttm(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-twelve-month columns to a per-quarter fundamentals frame.

    New columns: ``revenue_ttm``, ``net_income_ttm``, ``shares_ttm``, ``eps_ttm``.
    Balance-sheet columns are passed through as point-in-time values.
    """
    if quarterly is None or quarterly.empty:
        return quarterly
    df = quarterly.copy()
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.sort_values("period_end").reset_index(drop=True)
    df = _sanitize_shares(df)
    df["revenue_ttm"] = df["revenue"].rolling(4, min_periods=4).sum()
    df["net_income_ttm"] = df["net_income"].rolling(4, min_periods=4).sum()
    df["shares_ttm"] = df["shares_diluted"].ffill()
    df["eps_ttm"] = df["net_income_ttm"] / df["shares_ttm"]
    df["eps_ttm"] = df["eps_ttm"].replace([float("inf"), float("-inf")], pd.NA)
    return df


def bulk_fetch_and_store(tickers: Iterable[str]) -> int:
    """Fetch and persist fundamentals for many tickers. Returns rows written."""
    total = 0
    for t in tickers:
        try:
            df = get_fundamentals(t, refresh=True)
            total += len(df)
            log.info("%s: %d quarters", t, len(df))
        except Exception as e:
            log.warning("%s: %s", t, e)
    return total
