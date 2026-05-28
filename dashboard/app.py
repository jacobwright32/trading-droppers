"""Streamlit dashboard - ServiceNow-style chart for any ticker in the universe.

Run from the project root:

    streamlit run dashboard/app.py

The first time you pick a ticker the app downloads prices (yfinance) and
fundamentals (SEC EDGAR XBRL) and caches them in SQLite, so subsequent loads
of the same ticker are instant.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Streamlit Cloud injects secrets via st.secrets; surface them as env vars
# BEFORE trading_droppers.config is imported, since config (and the requests
# session in fundamentals.py) reads SEC_USER_AGENT once at import time.
try:
    for _key in ("SEC_USER_AGENT",):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except (FileNotFoundError, KeyError, Exception):  # noqa: BLE001
    # No secrets file locally is fine.
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_droppers import db
from trading_droppers import config
from trading_droppers import fundamentals as fund_mod
from trading_droppers import prices as price_mod

# Per-index ticker lists - tiny parquets committed to the repo. The dashboard
# never builds the universe at runtime; it just reads these. Refresh them with
# ``python scripts/build_universe_parquets.py``.
UNIVERSE_DIR = config.DATA_DIR / "universe"

INDEX_LABELS: dict[str, str] = {
    "sp500": "S&P 500",
    "sp400": "S&P 400 (mid-cap)",
    "sp600": "S&P 600 (small-cap)",
    "nasdaq100": "Nasdaq-100",
    "russell2000": "Russell 2000",
}


# --- cached data loaders ---------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _load_index_parquet(index_name: str) -> pd.DataFrame:
    path = UNIVERSE_DIR / f"{index_name}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "name", "cik"])
    df = pd.read_parquet(path)
    df["index_label"] = INDEX_LABELS.get(index_name, index_name)
    return df


def load_universe_from_parquets(indices: tuple[str, ...]) -> pd.DataFrame:
    """Union the per-index parquets for the chosen indices, dedupe by ticker."""
    if not indices:
        return pd.DataFrame(columns=["ticker", "name", "cik", "index_label"])
    frames = [_load_index_parquet(i) for i in indices]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if out.empty:
        return out
    # When a ticker appears in multiple selected indices (e.g. S&P 500 +
    # Nasdaq-100), keep the first listing - which is the higher-priority index
    # per the order the user picked them.
    out = out.drop_duplicates(subset="ticker", keep="first")
    return out.sort_values("ticker").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices_cached(ticker: str) -> pd.DataFrame:
    return price_mod.get_prices(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals_cached(ticker: str, cik: str | None) -> pd.DataFrame:
    return fund_mod.get_fundamentals(ticker, cik=cik)


# --- chart math ------------------------------------------------------------

def _ps_band(prices: pd.DataFrame, ttm: pd.DataFrame) -> pd.DataFrame:
    """Return per-price-tick band of (lo, mid, hi) scaled revenue-per-share.

    The band's center is ``rev_per_share x median_historical_P/S`` so it
    visually overlays price during typical valuation periods. The lo/hi edges
    span the 25th-75th percentile of historical P/S, giving a "typical
    valuation range" envelope.
    """
    if prices is None or prices.empty or ttm is None or ttm.empty:
        return pd.DataFrame()
    f = ttm.dropna(subset=["revenue_ttm", "shares_ttm"]).copy()
    if f.empty:
        return pd.DataFrame()
    f["rev_per_share"] = f["revenue_ttm"] / f["shares_ttm"]
    f = f[f["rev_per_share"] > 0]
    if f.empty:
        return pd.DataFrame()

    px = prices.sort_values("date")[["date", "close"]]
    f = f.sort_values("period_end")[["period_end", "rev_per_share"]]
    merged = pd.merge_asof(px, f, left_on="date", right_on="period_end", direction="backward")
    merged = merged.dropna(subset=["rev_per_share", "close"])
    if merged.empty:
        return pd.DataFrame()
    merged["ps"] = merged["close"] / merged["rev_per_share"]

    # Trim the band to the inter-tertile range so it stays visually subordinate
    # to price even when a stock has gone through a big multiple expansion or
    # compression (which widens the full IQR). The 35-65 spread feels closer to
    # the TrendSpider reference image.
    ps_med = float(merged["ps"].median())
    ps_lo = float(merged["ps"].quantile(0.35))
    ps_hi = float(merged["ps"].quantile(0.65))
    return pd.DataFrame(
        {
            "date": merged["date"].values,
            "rev_per_share": merged["rev_per_share"].values,
            "mid": merged["rev_per_share"].values * ps_med,
            "lo": merged["rev_per_share"].values * ps_lo,
            "hi": merged["rev_per_share"].values * ps_hi,
        }
    )


def _pe_series(prices: pd.DataFrame, ttm: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty or ttm is None or ttm.empty:
        return pd.DataFrame()
    f = ttm.dropna(subset=["eps_ttm"]).copy()
    if f.empty:
        return pd.DataFrame()
    px = prices.sort_values("date")[["date", "close"]]
    f = f.sort_values("period_end")[["period_end", "eps_ttm"]]
    merged = pd.merge_asof(px, f, left_on="date", right_on="period_end", direction="backward")
    merged["pe"] = merged["close"] / merged["eps_ttm"]
    merged.loc[merged["eps_ttm"] <= 0, "pe"] = np.nan  # losses -> no meaningful P/E
    return merged[["date", "pe", "eps_ttm"]]


# --- figure ----------------------------------------------------------------

def make_chart(ticker: str, prices: pd.DataFrame, ttm: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{"secondary_y": True}], [{}]],
    )

    # Revenue band - drawn first so price overlays it. Lower edge is plotted
    # invisible first, then the upper edge fills down to it, then we redraw the
    # upper edge as a stepped line with dot markers so it reads as "the
    # revenue line" (matches the TrendSpider visual).
    band = _ps_band(prices, ttm)
    if not band.empty:
        fig.add_trace(
            go.Scatter(
                x=band["date"], y=band["lo"],
                mode="lines",
                line=dict(width=0, shape="hv"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=band["date"], y=band["hi"],
                mode="lines",
                line=dict(width=0, shape="hv"),
                fill="tonexty",
                fillcolor="rgba(34, 139, 34, 0.30)",
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )
        # Stepped marker line along the top edge - this is "the revenue line".
        # Markers only show at each quarterly step so the chart stays clean.
        step_mask = band["rev_per_share"].diff().fillna(1).ne(0)
        fig.add_trace(
            go.Scatter(
                x=band["date"], y=band["hi"],
                mode="lines+markers",
                line=dict(color="rgba(200, 240, 200, 0.85)", width=1.4, shape="hv"),
                marker=dict(
                    color="rgba(220, 240, 220, 0.9)",
                    size=4,
                    opacity=step_mask.astype(float).values,
                ),
                name="Revenue (scaled)",
                customdata=band["rev_per_share"],
                hovertemplate="rev/share $%{customdata:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Price (weekly close).
    fig.add_trace(
        go.Scatter(
            x=prices["date"], y=prices["close"],
            mode="lines",
            line=dict(color="#4ade80", width=1.8),
            name=ticker,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Volume bars on a hidden secondary axis, scaled to bottom ~20% of the panel.
    if "volume" in prices.columns and prices["volume"].notna().any():
        fig.add_trace(
            go.Bar(
                x=prices["date"], y=prices["volume"],
                marker=dict(color="rgba(170, 170, 170, 0.35)"),
                showlegend=False,
                hovertemplate="vol %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1, secondary_y=True,
        )

    # P/E ratio panel.
    pe = _pe_series(prices, ttm)
    if not pe.empty and pe["pe"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pe["date"], y=pe["pe"],
                mode="lines",
                line=dict(color="rgba(220, 220, 220, 0.85)", width=1),
                name="P/E (TTM)",
                hovertemplate="%{x|%Y-%m-%d}<br>P/E %{y:.1f}x<extra></extra>",
            ),
            row=2, col=1,
        )
        finite = pe.dropna(subset=["pe"])
        if not finite.empty:
            peak = finite.loc[finite["pe"].idxmax()]
            trough = finite.loc[finite["pe"].idxmin()]
            last = finite.iloc[-1]
            seen: set[str] = set()
            for row_, color, yshift in (
                (peak, "rgba(220,220,220,0.85)", 12),
                (trough, "rgba(220,220,220,0.85)", -12),
                (last, "#4ade80", 14),
            ):
                # Annotations need a JSON-serialisable x value; Plotly accepts
                # ISO strings but chokes on pandas Timestamps.
                date_iso = pd.Timestamp(row_["date"]).strftime("%Y-%m-%d")
                if date_iso in seen:
                    continue
                seen.add(date_iso)
                # The P/E pane lives on yaxis3 because make_subplots(secondary_y=True)
                # in row 1 reserves yaxis2 for the volume secondary axis.
                fig.add_annotation(
                    xref="x2", yref="y3",
                    x=date_iso, y=float(row_["pe"]),
                    text=f"{row_['pe']:.0f}x P/E",
                    showarrow=False,
                    yshift=yshift,
                    font=dict(color=color, size=10),
                )

    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=40, r=60, t=20, b=30),
        plot_bgcolor="#0b1018",
        paper_bgcolor="#0b1018",
        showlegend=False,
        bargap=0.5,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=False, row=1, col=1, secondary_y=True, showticklabels=False
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        row=2, col=1, title_text="P/E (TTM)",
    )
    # Push volume bars into the bottom ~20% of the price pane.
    if "volume" in prices.columns and prices["volume"].notna().any():
        vmax = float(prices["volume"].max())
        if vmax > 0:
            fig.update_yaxes(range=[0, vmax * 5], row=1, col=1, secondary_y=True)
    return fig


# --- app -------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="trading-droppers", layout="wide", page_icon="📉")
    st.markdown("# trading-droppers")
    st.caption("Stock dropped, business kept compounding - the de-rating shape.")

    # Ensure the prices/fundamentals cache tables exist. Universe is parquet-based
    # so we don't touch it here.
    db.init_schema()

    # If the Screener page handed us an index along with a ticker, make sure
    # that index is selected on this page BEFORE the multiselect is rendered.
    handoff_index = st.session_state.pop("index_choice", None)
    if "indices_pick" not in st.session_state:
        st.session_state["indices_pick"] = ["sp500"]
    if handoff_index and handoff_index in INDEX_LABELS:
        current = list(st.session_state["indices_pick"])
        if handoff_index not in current:
            current.append(handoff_index)
            st.session_state["indices_pick"] = current

    with st.sidebar:
        st.subheader("Universe filter")
        chosen = st.multiselect(
            "Indices",
            options=list(INDEX_LABELS.keys()),
            key="indices_pick",
            format_func=lambda c: INDEX_LABELS.get(c, c),
        )
        require_cik = st.checkbox(
            "Only tickers with SEC CIK",
            value=True,
            help="Tickers without a CIK can't be matched to EDGAR fundamentals.",
        )

        uni = load_universe_from_parquets(tuple(chosen))
        if uni.empty:
            st.warning(
                "No indices selected (or no parquet files). Pick at least one index."
            )
            st.stop()
        if require_cik:
            uni = uni[uni["cik"].notna() & (uni["cik"].astype(str) != "")]
        if uni.empty:
            st.warning("No tickers match the current filters.")
            st.stop()

        tickers = uni["ticker"].tolist()
        name_by_ticker = dict(zip(uni["ticker"], uni["name"].fillna("")))
        cik_by_ticker = dict(zip(uni["ticker"], uni["cik"].fillna("")))
        # If the user landed here from the Screener page via the "View chart"
        # button, the ticker they picked is in session_state - default to it.
        handoff = st.session_state.pop("ticker_choice", None)
        if handoff and handoff in tickers:
            default_ticker = handoff
        elif "NOW" in tickers:
            default_ticker = "NOW"
        else:
            default_ticker = tickers[0]
        choice = st.selectbox(
            "Ticker",
            options=tickers,
            index=tickers.index(default_ticker),
            format_func=lambda t: f"{t}  -  {name_by_ticker.get(t, '')}",
        )

        refresh = st.button("Refresh data", help="Re-download prices and fundamentals.")
        st.caption(f"Universe: {len(uni):,} tickers")

    if refresh:
        load_prices_cached.clear()
        load_fundamentals_cached.clear()

    company = name_by_ticker.get(choice, "")
    st.markdown(f"## {choice} - {company}" if company else f"## {choice}")

    cik = cik_by_ticker.get(choice) or None
    with st.spinner(f"Loading prices for {choice}..."):
        prices = load_prices_cached(choice)
    with st.spinner(f"Loading fundamentals for {choice}..."):
        fund = load_fundamentals_cached(choice, cik)
    ttm = fund_mod.compute_ttm(fund) if fund is not None and not fund.empty else fund

    if prices is None or prices.empty:
        st.error("No price data could be downloaded for this ticker.")
        return

    # Top KPI strip
    c1, c2, c3, c4 = st.columns(4)
    last_close = float(prices["close"].iloc[-1])
    peak = float(prices["close"].max())
    drawdown = (last_close / peak - 1.0) if peak else 0.0
    c1.metric("Last close", f"${last_close:,.2f}")
    c2.metric("Trailing peak", f"${peak:,.2f}")
    c3.metric("Drawdown from peak", f"{drawdown:.0%}")
    if ttm is not None and not ttm.empty:
        rev_ttm = ttm["revenue_ttm"].dropna()
        if len(rev_ttm) >= 5:
            yoy = float(rev_ttm.iloc[-1] / rev_ttm.iloc[-5] - 1.0)
            c4.metric(
                "Revenue TTM",
                f"${rev_ttm.iloc[-1] / 1e9:,.2f}B",
                f"{yoy:+.1%} YoY",
            )
        else:
            c4.metric("Revenue TTM", "n/a")
    else:
        c4.metric("Revenue TTM", "n/a")

    fig = make_chart(choice, prices, ttm if ttm is not None else pd.DataFrame())
    st.plotly_chart(fig, use_container_width=True, theme=None)

    if fund is None or fund.empty:
        st.info(
            "No SEC fundamentals on file for this ticker - the chart shows price only."
        )

    with st.expander("Raw quarterly fundamentals (last 20)"):
        if fund is not None and not fund.empty:
            st.dataframe(fund.tail(20), use_container_width=True)
        else:
            st.write("(none)")


if __name__ == "__main__":
    main()
