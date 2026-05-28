"""Central configuration for trading-droppers.

Single source of truth for paths, the scan universe, screen thresholds, and the
optional SEC + LLM settings. Environment variables (loaded from a local ``.env``)
override the defaults where it makes sense.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: .env is convenient but not required
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a soft dependency
    pass

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = Path(os.getenv("TD_DB_PATH", str(DATA_DIR / "trading_droppers.db")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Universe --------------------------------------------------------------
# Index constituents that make up the scan universe. Russell 2000 is pulled
# from the iShares IWM holdings file; the rest come from Wikipedia tables.
UNIVERSE_INDICES = ["sp500", "sp400", "sp600", "nasdaq100", "russell2000"]
# Optional extra tickers from a watchlist file (one symbol per line, # comments ok).
WATCHLIST_FILE = ROOT / "watchlist.txt"

# --- Prices (yfinance) -----------------------------------------------------
PRICE_PERIOD = os.getenv("TD_PRICE_PERIOD", "6y")       # history depth
PRICE_INTERVAL = os.getenv("TD_PRICE_INTERVAL", "1wk")  # weekly, like the chart
YF_BATCH_SIZE = int(os.getenv("TD_YF_BATCH", "120"))    # tickers per download call

# --- SEC EDGAR -------------------------------------------------------------
# SEC requires a descriptive User-Agent containing contact info; generic/empty
# agents get blocked. Set SEC_USER_AGENT in .env, e.g. "trading-droppers you@mail.com".
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "trading-droppers research (set SEC_USER_AGENT in .env)"
)
SEC_RATE_LIMIT_PER_SEC = 8  # SEC permits up to 10 req/s; stay under it.
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


# --- Screen thresholds (all tunable) ---------------------------------------
@dataclass(frozen=True)
class DeRatingParams:
    """The ServiceNow thesis: price fell hard while the business kept growing."""

    min_drawdown: float = 0.30        # >= 30% below the trailing high
    min_revenue_growth: float = 0.08  # TTM revenue YoY >= 8%
    min_ps_compression: float = 0.20  # P/S now >= 20% below its level ~1y ago
    require_profitable: bool = False  # de-raters can be not-yet-profitable growers


@dataclass(frozen=True)
class QualityDropperParams:
    """Beaten-down but financially sturdy - built to avoid value traps."""

    min_drawdown: float = 0.35
    min_revenue_growth: float = 0.0          # not shrinking
    max_liabilities_to_assets: float = 0.70  # conservative leverage
    require_positive_net_income: bool = True


DERATING = DeRatingParams()
QUALITY = QualityDropperParams()

# --- LLM narrative layer (optional, OFF by default) ------------------------
# The hard numbers come from SEC XBRL. This optional layer summarises the
# MD&A / risk-factor narrative from the latest 10-K. See trading_droppers/llm.py.
LLM_ENABLED = os.getenv("TD_LLM_ENABLED", "false").lower() in ("1", "true", "yes")
LLM_PROVIDER = os.getenv("TD_LLM_PROVIDER", "anthropic")  # anthropic | openai | ollama
LLM_MODEL = os.getenv("TD_LLM_MODEL", "")  # blank -> provider default
