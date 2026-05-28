# trading-droppers

Find companies whose **stock price has fallen hard while the underlying business keeps growing** — that is, *de-rating*, not decline. When price drops but revenue/profit compound, the valuation multiple compresses to an unusual low. Those are the candidates worth a look.

## The thesis

The inspiration is the ServiceNow (NOW) weekly chart:

- **Revenue** (the stepped band) keeps climbing — the business never stopped compounding.
- **Price** peaked near $250 in early 2025, then fell to ~$100.
- **P/E** compressed from ~129x down to ~27x — the cheapest in years.

Same company, far cheaper multiple. The screeners here look for exactly this shape across a wide universe: a big drawdown, *growing* fundamentals, and a compressed valuation — while filtering out value traps (businesses that are cheap because they're actually deteriorating).

## How it works

```
                 ┌──────────────────────────────────────────────┐
  yfinance ─────▶│ prices: weekly history → drawdown, returns,   │──┐
                 │          P/S compression                       │  │
                 └──────────────────────────────────────────────┘  │
                 ┌──────────────────────────────────────────────┐  │   ┌─────────────┐
  SEC EDGAR ────▶│ fundamentals (XBRL): revenue, net income,     │──┼──▶│  SQLite DB  │
   companyfacts  │   assets, liabilities, equity (annual + TTM)   │  │   └──────┬──────┘
                 └──────────────────────────────────────────────┘  │          │
                 ┌──────────────────────────────────────────────┐  │          ▼
  universe ─────▶│ S&P 400/500/600 + Nasdaq-100 + Russell 2000   │──┘   ┌─────────────┐
                 └──────────────────────────────────────────────┘      │  screeners  │
                                                                        │ de-rating + │
                                                                        │ quality-drop│
                                                                        └──────┬──────┘
                                                                               ▼
                                                                     ┌───────────────────┐
                                                                     │ Streamlit dashboard│
                                                                     └───────────────────┘
```

Each stage is a standalone script that writes to a shared **SQLite** database (`data/trading_droppers.db`). The dashboard reads from it. Run stages independently or all at once.

## The screening systems

1. **De-rating** (`screens/derating.py`) — the ServiceNow shape. Big drawdown from the trailing high **and** growing TTM revenue **and** a compressed price-to-sales multiple vs. ~1 year ago. Allows not-yet-profitable growers.
2. **Quality dropper** (`screens/quality_dropper.py`) — beaten down but financially sturdy: large drawdown, revenue not shrinking, conservative leverage (liabilities/assets), and positive net income. Built to avoid value traps.

Thresholds live in `trading_droppers/config.py` and are easy to tune. Adding a third system is just another module in `trading_droppers/screens/`.

## Data sources

- **Prices** — [`yfinance`](https://github.com/ranaroussi/yfinance). Weekly history; no key required.
- **Fundamentals** — [SEC EDGAR XBRL `companyfacts`](https://www.sec.gov/edgar/sec-api-documentation). Free, exact, straight from the filings (10-K/10-Q). Requires only a descriptive `User-Agent`.
- **Universe** — index constituents from Wikipedia (S&P 400/500/600, Nasdaq-100) and the Russell 2000 from the iShares **IWM** holdings file.

## Setup

```powershell
# from C:\dev\trading-droppers
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # already done if you used the scaffold

copy .env.example .env                 # then set SEC_USER_AGENT to your email
```

> **SEC requires contact info.** Set `SEC_USER_AGENT="trading-droppers your-email@example.com"` in `.env`, or EDGAR will reject the requests.

## Usage

```powershell
python scripts/build_universe.py       # 1. build the ticker list
python scripts/fetch_prices.py         # 2. prices + drawdown/de-rating metrics
python scripts/fetch_fundamentals.py   # 3. SEC XBRL revenue/profit/assets/liabilities
python scripts/run_screens.py          # 4. rank candidates
# ...or everything in order:
python scripts/run_all.py

streamlit run dashboard/app.py         # explore the results
```

Start small with `python scripts/run_all.py --limit 50` to smoke-test before scanning the full ~2,500-name universe.

## Optional: LLM narrative layer

The hard numbers come from XBRL (no LLM needed). If you also want qualitative summaries of the latest 10-K's MD&A / risk factors, enable the pluggable layer:

```
TD_LLM_ENABLED=true
TD_LLM_PROVIDER=anthropic     # anthropic | openai | ollama
ANTHROPIC_API_KEY=...
```

then `pip install -r requirements-llm.txt`. See `trading_droppers/llm.py`.

## Project layout

```
trading_droppers/      core package (config, db, universe, prices, fundamentals, screens, llm)
scripts/               runnable pipeline stages
dashboard/             Streamlit app
data/                  SQLite DB + caches (gitignored)
```

## Disclaimer

For research and education only. Not investment advice. Fundamental data can be delayed, restated, or mis-mapped; always verify against the primary filing before acting.
