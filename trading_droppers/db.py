"""SQLite persistence layer.

One shared DB at ``config.DB_PATH``. Three tables:

* ``universe``     - one row per ticker, with boolean columns marking which
  index(es) it belongs to. The dashboard filters off these.
* ``prices``       - weekly OHLCV from yfinance.
* ``fundamentals`` - per-period quarterly facts from SEC EDGAR XBRL. TTM
  aggregates are derived on read (see ``fundamentals.compute_ttm``).

Every writer accepts an optional ``conn`` so callers can batch many upserts in
a single transaction; when omitted, a short-lived connection is opened.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from . import config


# Columns on ``universe`` that mark index membership. Whitelisted so we can
# safely splice them into SQL when filtering (no user input ever reaches here).
INDEX_COLUMNS: tuple[str, ...] = (
    "sp400",
    "sp500",
    "sp600",
    "nasdaq100",
    "russell2000",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    cik           TEXT,
    sp400         INTEGER NOT NULL DEFAULT 0,
    sp500         INTEGER NOT NULL DEFAULT 0,
    sp600         INTEGER NOT NULL DEFAULT 0,
    nasdaq100     INTEGER NOT NULL DEFAULT 0,
    russell2000   INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS ix_universe_cik ON universe(cik);

CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_prices_ticker_date ON prices(ticker, date);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker         TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    period_start   TEXT,
    fp             TEXT NOT NULL,    -- Q1 / Q2 / Q3 / Q4 / FY
    fy             INTEGER,
    form           TEXT,             -- 10-K, 10-Q, ...
    revenue        REAL,             -- period revenue (quarterly or annual per fp)
    net_income     REAL,             -- period net income
    assets         REAL,             -- balance sheet, point-in-time at period_end
    liabilities    REAL,
    equity         REAL,
    shares_diluted REAL,             -- weighted-avg diluted shares for the period
    PRIMARY KEY (ticker, period_end, fp)
);
CREATE INDEX IF NOT EXISTS ix_fundamentals_ticker ON fundamentals(ticker);
"""


@contextmanager
def connect(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with sensible pragmas. Commits on clean exit."""
    path = Path(path) if path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    if conn is None:
        with connect() as c:
            c.executescript(SCHEMA)
    else:
        conn.executescript(SCHEMA)


# --- writers ---------------------------------------------------------------

def upsert_universe(
    rows: Iterable[dict],
    conn: sqlite3.Connection | None = None,
) -> int:
    """Upsert one or more universe rows.

    Index membership flags are OR-merged so re-running scrapers can't *unset*
    a flag set by a prior run that pulled a different index.
    """
    rows = list(rows)
    if not rows:
        return 0
    sql = """
    INSERT INTO universe (ticker, name, cik, sp400, sp500, sp600, nasdaq100,
                          russell2000, updated_at)
    VALUES (:ticker, :name, :cik, :sp400, :sp500, :sp600, :nasdaq100,
            :russell2000, datetime('now'))
    ON CONFLICT(ticker) DO UPDATE SET
        name        = COALESCE(excluded.name, universe.name),
        cik         = COALESCE(excluded.cik, universe.cik),
        sp400       = MAX(universe.sp400,       excluded.sp400),
        sp500       = MAX(universe.sp500,       excluded.sp500),
        sp600       = MAX(universe.sp600,       excluded.sp600),
        nasdaq100   = MAX(universe.nasdaq100,   excluded.nasdaq100),
        russell2000 = MAX(universe.russell2000, excluded.russell2000),
        updated_at  = datetime('now')
    """
    if conn is None:
        with connect() as c:
            return c.executemany(sql, rows).rowcount
    return conn.executemany(sql, rows).rowcount


def upsert_prices(
    df: pd.DataFrame,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Upsert weekly OHLCV. ``df`` columns: ticker, date, open, high, low, close, volume."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]].to_dict(
        "records"
    )
    sql = """
    INSERT INTO prices (ticker, date, open, high, low, close, volume)
    VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
    ON CONFLICT(ticker, date) DO UPDATE SET
        open=excluded.open, high=excluded.high, low=excluded.low,
        close=excluded.close, volume=excluded.volume
    """
    if conn is None:
        with connect() as c:
            return c.executemany(sql, rows).rowcount
    return conn.executemany(sql, rows).rowcount


def upsert_fundamentals(
    rows: Iterable[dict],
    conn: sqlite3.Connection | None = None,
) -> int:
    rows = list(rows)
    if not rows:
        return 0
    sql = """
    INSERT INTO fundamentals (ticker, period_end, period_start, fp, fy, form,
                              revenue, net_income, assets, liabilities, equity,
                              shares_diluted)
    VALUES (:ticker, :period_end, :period_start, :fp, :fy, :form,
            :revenue, :net_income, :assets, :liabilities, :equity, :shares_diluted)
    ON CONFLICT(ticker, period_end, fp) DO UPDATE SET
        period_start   = COALESCE(excluded.period_start,   fundamentals.period_start),
        fy             = COALESCE(excluded.fy,             fundamentals.fy),
        form           = COALESCE(excluded.form,           fundamentals.form),
        revenue        = COALESCE(excluded.revenue,        fundamentals.revenue),
        net_income     = COALESCE(excluded.net_income,     fundamentals.net_income),
        assets         = COALESCE(excluded.assets,         fundamentals.assets),
        liabilities    = COALESCE(excluded.liabilities,    fundamentals.liabilities),
        equity         = COALESCE(excluded.equity,         fundamentals.equity),
        shares_diluted = COALESCE(excluded.shares_diluted, fundamentals.shares_diluted)
    """
    if conn is None:
        with connect() as c:
            return c.executemany(sql, rows).rowcount
    return conn.executemany(sql, rows).rowcount


# --- readers ---------------------------------------------------------------

def load_universe(indices: Sequence[str] | None = None) -> pd.DataFrame:
    """Return the universe, optionally filtered to rows in *any* of ``indices``.

    ``indices`` values must be drawn from ``INDEX_COLUMNS``; anything else is
    silently dropped to keep the SQL splice safe.
    """
    sql = "SELECT * FROM universe"
    if indices:
        valid = [c for c in indices if c in INDEX_COLUMNS]
        if valid:
            sql += " WHERE " + " OR ".join(f"{c} = 1" for c in valid)
    sql += " ORDER BY ticker"
    with connect() as conn:
        return pd.read_sql_query(sql, conn)


def load_prices(ticker: str) -> pd.DataFrame:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date",
            conn,
            params=(ticker,),
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_fundamentals(ticker: str) -> pd.DataFrame:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM fundamentals WHERE ticker = ? ORDER BY period_end",
            conn,
            params=(ticker,),
        )
    for col in ("period_end", "period_start"):
        if col in df.columns and not df.empty:
            df[col] = pd.to_datetime(df[col])
    return df


def get_cik(ticker: str) -> str | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT cik FROM universe WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0] if row and row[0] else None
