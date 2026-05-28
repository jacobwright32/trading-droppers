"""Screener page - de-rating candidates ranked by drawdown x revenue growth.

Reads ``data/screening/snapshot.parquet`` (built by
``scripts/build_screening_snapshot.py``). Filtering is interactive but the
underlying data is a static snapshot, so the page renders instantly.

Click any row in the table and a "View chart for ..." button appears that
hands the selected ticker to the Chart page via ``st.session_state``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow ``streamlit run dashboard/app.py`` to find the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from trading_droppers import config

SNAPSHOT_PATH = config.DATA_DIR / "screening" / "snapshot.parquet"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_snapshot() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(SNAPSHOT_PATH)


def _fmt_pct(x: object) -> str:
    return f"{float(x):+.0%}" if pd.notna(x) else "-"


def _fmt_pct_unsigned(x: object) -> str:
    return f"{float(x):.0%}" if pd.notna(x) else "-"


def _fmt_billions(x: object) -> str:
    if not pd.notna(x):
        return "-"
    v = float(x)
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"


def main() -> None:
    st.set_page_config(
        page_title="Screener - trading-droppers", layout="wide", page_icon="📉"
    )
    st.markdown("# Screener")
    st.caption(
        "Stocks down hard while revenue keeps compounding - the ServiceNow shape."
    )

    df = load_snapshot()
    if df.empty:
        st.warning(
            "No snapshot found at `data/screening/snapshot.parquet`. "
            "Run `python scripts/build_screening_snapshot.py` and commit the parquet."
        )
        st.stop()

    snapshot_age = pd.Timestamp(SNAPSHOT_PATH.stat().st_mtime, unit="s")
    st.caption(
        f"Snapshot: {len(df):,} tickers screened, "
        f"built {snapshot_age:%Y-%m-%d %H:%M} UTC."
    )

    with st.sidebar:
        st.subheader("Thresholds")
        min_drawdown = st.slider(
            "Min drawdown from peak", 0.0, 0.80, 0.30, 0.05,
            help="Stock is at least this far below its 5-year high.",
            format="%.2f",
        )
        min_yoy = st.slider(
            "Min revenue TTM YoY", -0.10, 0.50, 0.08, 0.01,
            help="Trailing-twelve-month revenue grew at least this much YoY.",
            format="%.2f",
        )
        max_yoy = st.slider(
            "Max revenue TTM YoY", 0.10, 5.00, 1.00, 0.10,
            help="Caps the ranking - tiny-base microcaps with 10000%% YoY are "
                 "noisy, not de-rating candidates. Bump up if you want to see "
                 "them anyway.",
            format="%.2f",
        )
        min_revenue_b = st.slider(
            "Min revenue TTM ($B)", 0.0, 10.0, 0.5, 0.1,
            help="Cuts out microcaps - the de-rating thesis is about real "
                 "businesses with established revenue.",
            format="%.1f",
        )
        min_compression = st.slider(
            "Min P/S compression", -0.20, 0.80, 0.20, 0.05,
            help="P/S now is at least this much below P/S one year ago.",
            format="%.2f",
        )
        require_profitable = st.checkbox(
            "Require profitable (TTM)",
            value=False,
            help="Tighter quality bar; off by default so de-raters with recent "
                 "losses (like early-cycle SaaS) still show up.",
        )

    drawdown = df["drawdown"].astype(float)
    yoy = df["revenue_yoy"].astype(float)
    compression = df["ps_compression"].astype(float)
    revenue = df["revenue_ttm"].astype(float)

    mask = (drawdown <= -min_drawdown) & (yoy >= min_yoy) & (yoy <= max_yoy)
    mask &= revenue >= min_revenue_b * 1e9
    # P/S compression filter only applies when we have the data.
    mask &= compression.fillna(-99) >= min_compression
    if require_profitable:
        mask &= df["is_profitable"].fillna(False).astype(bool)

    candidates = df[mask].copy()
    if candidates.empty:
        st.info("No candidates match these thresholds. Try loosening them.")
        st.stop()

    # Score: bigger drawdown and faster growth both push you up the ranking.
    # We use the filtered (capped) YoY so a single microcap with 50000% growth
    # doesn't dominate the table.
    candidates["score"] = (
        candidates["drawdown"].abs() * candidates["revenue_yoy"].clip(upper=max_yoy)
    )
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)

    st.markdown(f"### {len(candidates)} candidates")

    display = pd.DataFrame(
        {
            "Ticker": candidates["ticker"],
            "Company": candidates["name"],
            "Drawdown": candidates["drawdown"].apply(_fmt_pct),
            "Revenue YoY": candidates["revenue_yoy"].apply(_fmt_pct),
            "P/S compression": candidates["ps_compression"].apply(_fmt_pct_unsigned),
            "Revenue TTM": candidates["revenue_ttm"].apply(_fmt_billions),
            "P/E TTM": candidates["pe_ttm"].apply(
                lambda x: f"{float(x):.0f}x" if pd.notna(x) and float(x) > 0 else "-"
            ),
            "Profitable": candidates["is_profitable"].apply(
                lambda x: "yes" if x is True else ("no" if x is False else "-")
            ),
        }
    )

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=min(640, 38 + 36 * len(display)),
    )

    selected_rows = (event.selection or {}).get("rows", []) if event else []
    if selected_rows:
        row = candidates.iloc[selected_rows[0]]
        st.markdown(f"#### Selected: **{row['ticker']}** - {row['name']}")
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("View chart", type="primary", use_container_width=True):
                st.session_state["ticker_choice"] = row["ticker"]
                st.session_state["cik_choice"] = row.get("cik") or None
                st.session_state["index_choice"] = row.get("index") or None
                st.switch_page("app.py")
        with c2:
            st.metric("Drawdown", _fmt_pct(row["drawdown"]))
        with c3:
            st.metric("Revenue YoY", _fmt_pct(row["revenue_yoy"]))


if __name__ == "__main__":
    main()
