"""
SIP Simulator — Streamlit Dashboard (Ledger Edition)
=====================================================
Reads from portfolio_ledger.json written by the monthly scheduler.
No API key required. No live backtest — shows actual recorded investments.

Run from fin-agents root:
  streamlit run simulator/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

_ROOT     = Path(__file__).resolve().parent.parent   # sip_execution_mas/
_FIN_ROOT = _ROOT.parent                              # fin-agents/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulator.ledger import aggregate_holdings, ledger_summary, load_ledger

_LEDGER_FILE      = Path(__file__).resolve().parent / "outputs" / "portfolio_ledger.json"
_DEFAULT_RANKINGS = _FIN_ROOT / "etf-selection-mas" / "outputs" / "rankings.json"

CORE_COLOR      = "#4C9BE8"
SATELLITE_COLOR = "#F4A261"
POSITIVE_COLOR  = "#2ECC71"
NEGATIVE_COLOR  = "#E74C3C"
BG              = "rgba(0,0,0,0)"
GRID            = "#1E2130"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers  (defined first — Streamlit executes top-to-bottom)
# ══════════════════════════════════════════════════════════════════════════════

def _usd(v: float) -> str:
    return f"${v:,.2f}"

def _pct(v: float, plus: bool = True) -> str:
    sign = "+" if plus and v >= 0 else ""
    return f"{sign}{v:.2f}%"

def _ter(v) -> str:
    return f"{v*100:.3f}%" if v else "N/A"


@st.cache_data(show_spinner=False)
def _load_ledger_cached(path: str) -> Dict[str, Any]:
    return load_ledger(path)


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_current_prices(tickers: tuple) -> Dict[str, float]:
    """Fetch latest close price for each ticker (cached 5 min)."""
    prices: Dict[str, float] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if not hist.empty:
                prices[ticker] = round(float(hist["Close"].iloc[-1]), 4)
        except Exception:
            pass
    return prices


def _compute_current_value(
    holdings: Dict[str, Any],
    prices: Dict[str, float],
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach live prices + current value + P&L to each holding.
    Returns enriched holdings dict and portfolio totals.
    """
    # Build a per-entry usd_inr map (use latest as fallback)
    latest_usd_inr = 83.5
    for entry in ledger.get("entries", []):
        latest_usd_inr = entry.get("usd_inr_rate", latest_usd_inr)

    enriched: Dict[str, Any] = {}
    total_value   = 0.0
    total_invested = 0.0

    for ticker, h in holdings.items():
        price = prices.get(ticker)
        inv   = h["invested_usd"]
        total_invested += inv

        if price:
            if h["region"] == "BSE":
                value = (h["total_units"] * price) / latest_usd_inr
            else:
                value = h["total_units"] * price
            pnl  = value - inv
            ret  = (pnl / inv * 100) if inv > 0 else 0.0
            total_value += value
        else:
            value = pnl = ret = None

        enriched[ticker] = {
            **h,
            "current_price": price,
            "current_value_usd": round(value, 2) if value is not None else None,
            "pnl_usd":           round(pnl, 2)   if pnl   is not None else None,
            "return_pct":        round(ret, 2)    if ret   is not None else None,
        }

    total_pnl = total_value - total_invested if total_value else None
    total_ret = (total_pnl / total_invested * 100) if total_invested > 0 and total_pnl is not None else None

    return enriched, {
        "total_invested":   round(total_invested, 2),
        "current_value":    round(total_value, 2),
        "total_pnl":        round(total_pnl, 2) if total_pnl is not None else None,
        "total_return_pct": round(total_ret, 2) if total_ret is not None else None,
        "usd_inr":          latest_usd_inr,
    }


def _monthly_timeline_df(ledger: Dict[str, Any], holdings_enriched: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a monthly timeline: cumulative invested vs estimated portfolio value.
    Each month's portfolio value is estimated using prices paid that month.
    """
    cum_units: Dict[str, float] = {}
    rows = []
    cum_invested = 0.0

    for entry in ledger.get("entries", []):
        usd_inr = entry.get("usd_inr_rate", 83.5)

        for pos in entry.get("positions", []):
            t = pos["ticker"]
            cum_units[t] = cum_units.get(t, 0.0) + pos["units_bought"]

        cum_invested += entry.get("total_invested_usd", 0)

        # Estimate value using price paid this month (period-end proxy)
        port_val = 0.0
        for pos in entry.get("positions", []):
            t     = pos["ticker"]
            units = cum_units.get(t, 0.0)
            price = pos.get("price_native", 0)
            if price > 0:
                if pos.get("region") == "BSE":
                    port_val += (units * price) / usd_inr
                else:
                    port_val += units * price

        rows.append({
            "Month":     entry["month"],
            "Invested":  round(cum_invested, 2),
            "Value":     round(port_val, 2),
            "P&L":       round(port_val - cum_invested, 2),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SIP Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📈 SIP Simulator")
    st.caption("Ledger-driven · No API key required")
    st.divider()

    # Ledger status
    st.subheader("📒 Portfolio Ledger")
    ledger_path = str(_LEDGER_FILE)
    ledger_raw  = _load_ledger_cached(ledger_path)
    summary     = ledger_summary(ledger_raw)

    if summary["months"] > 0:
        st.success(
            f"✓ **{summary['months']} months** recorded\n\n"
            f"First: {summary['first_investment']}\n\n"
            f"Last: {summary['last_investment']}\n\n"
            f"Total invested: **{_usd(summary['total_invested_usd'])}**"
        )
    else:
        st.warning(
            "No investment records yet.\n\n"
            "Run the scheduler to create the first entry:\n"
            "```\npython -m simulator.scheduler --now\n```"
        )

    st.divider()

    # Scheduler command reference
    with st.expander("🕐 Scheduler commands"):
        st.code(
            "# First investment (run now)\n"
            "python -m simulator.scheduler --now\n\n"
            "# Start monthly auto-scheduler\n"
            "python -m simulator.scheduler\n\n"
            "# Check ledger status\n"
            "python -m simulator.scheduler --status\n\n"
            "# Custom SIP / portfolio size\n"
            "python -m simulator.scheduler --now \\\n"
            "  --sip 750 --top 10 --core 5",
            language="bash",
        )

    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  Guard — need at least one ledger entry
# ══════════════════════════════════════════════════════════════════════════════

if summary["months"] == 0:
    st.title("📊 SIP Simulator")
    st.info(
        "**No investment records found yet.**\n\n"
        "Run the scheduler to record your first monthly investment:\n"
        "```bash\npython -m simulator.scheduler --now\n```\n\n"
        "Then click **🔄 Refresh data** in the sidebar."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  Load & enrich data
# ══════════════════════════════════════════════════════════════════════════════

ledger   = _load_ledger_cached(ledger_path)
holdings = aggregate_holdings(ledger)
tickers  = tuple(sorted(holdings.keys()))

with st.spinner("Fetching live prices …"):
    prices = _fetch_current_prices(tickers)

holdings_enriched, totals = _compute_current_value(holdings, prices, ledger)
boom_triggers = []
if os.path.exists(str(_DEFAULT_RANKINGS)):
    with open(str(_DEFAULT_RANKINGS)) as f:
        boom_triggers = json.load(f).get("boom_triggers_fired", [])

# ══════════════════════════════════════════════════════════════════════════════
#  Header + top metrics
# ══════════════════════════════════════════════════════════════════════════════

st.title("📊 SIP Simulator — Portfolio Dashboard")
st.caption(
    f"Last investment: **{summary['last_investment']}**  ·  "
    f"Live prices as of: **{datetime.now().strftime('%Y-%m-%d %H:%M')}**  ·  "
    f"USD/INR: **{totals['usd_inr']:.2f}**  ·  buy-only"
)

if boom_triggers:
    st.info("🚀  **Active Boom Triggers:** " + "  ·  ".join(f"`{t}`" for t in boom_triggers))

pnl = totals["total_pnl"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Invested",    _usd(totals["total_invested"]))
c2.metric(
    "Portfolio Value",
    _usd(totals["current_value"]),
    delta=f"+{_usd(pnl)}" if pnl and pnl >= 0 else (_usd(pnl) if pnl else "—"),
)
c3.metric(
    "Total Return",
    _pct(totals["total_return_pct"]) if totals["total_return_pct"] is not None else "—",
    delta=_pct(totals["total_return_pct"]) if totals["total_return_pct"] is not None else None,
    delta_color="normal" if pnl and pnl >= 0 else "inverse",
)
c4.metric("Months Invested", str(summary["months"]))

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_portfolio, tab_holdings, tab_history, tab_allocation, tab_timeline = st.tabs([
    "🗂 Portfolio",
    "💼 Holdings",
    "📒 Ledger History",
    "📐 Last Allocation",
    "📅 Timeline",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Portfolio overview
# ─────────────────────────────────────────────────────────────────────────────

with tab_portfolio:
    col_left, col_right = st.columns(2)

    # Portfolio composition by value (live)
    with col_left:
        st.subheader("Portfolio Composition (live value)")
        pie_data = [
            (t, h["current_value_usd"], h["bucket"])
            for t, h in holdings_enriched.items()
            if h.get("current_value_usd")
        ]
        if pie_data:
            labels  = [d[0] for d in pie_data]
            values  = [d[1] for d in pie_data]
            buckets = [d[2] for d in pie_data]
            fig_pie = go.Figure(go.Pie(
                labels        = labels,
                values        = values,
                hole          = 0.45,
                marker        = dict(
                    colors=[CORE_COLOR if b == "core" else SATELLITE_COLOR for b in buckets],
                    line=dict(color="#0E1117", width=2),
                ),
                textinfo      = "label+percent",
                hovertemplate = "<b>%{label}</b><br>Value: $%{value:,.2f}<extra></extra>",
            ))
            fig_pie.add_annotation(
                text=f"{_usd(totals['current_value'])}<br>total",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="white"),
            )
            fig_pie.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color="white")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Core vs Satellite current value
    with col_right:
        st.subheader("Core vs Satellite (live value)")
        core_val = sum(
            h["current_value_usd"] for h in holdings_enriched.values()
            if h.get("current_value_usd") and h["bucket"] == "core"
        )
        sat_val = sum(
            h["current_value_usd"] for h in holdings_enriched.values()
            if h.get("current_value_usd") and h["bucket"] == "satellite"
        )
        if core_val + sat_val > 0:
            fig_cs = go.Figure(go.Pie(
                labels        = ["Core", "Satellite"],
                values        = [core_val, sat_val],
                hole          = 0.45,
                marker        = dict(
                    colors=[CORE_COLOR, SATELLITE_COLOR],
                    line=dict(color="#0E1117", width=3),
                ),
                textinfo      = "label+percent+value",
                hovertemplate = "<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
            ))
            fig_cs.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color="white")),
            )
            st.plotly_chart(fig_cs, use_container_width=True)

    # Per-ETF return bar chart
    st.subheader("Per-ETF Returns (live)")
    ret_rows = [
        {
            "Ticker":  t,
            "Bucket":  h["bucket"].capitalize(),
            "Return %":h["return_pct"],
            "P&L":     h["pnl_usd"],
        }
        for t, h in holdings_enriched.items()
        if h.get("return_pct") is not None
    ]
    if ret_rows:
        ret_df = pd.DataFrame(ret_rows).sort_values("Return %")
        colors = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in ret_df["Return %"]]
        fig_ret = go.Figure(go.Bar(
            x=ret_df["Return %"], y=ret_df["Ticker"],
            orientation="h",
            marker_color=colors,
            customdata=ret_df[["P&L"]].values,
            hovertemplate="<b>%{y}</b><br>Return: %{x:.2f}%<br>P&L: $%{customdata[0]:,.2f}<extra></extra>",
        ))
        fig_ret.add_vline(x=0, line_color="white", line_width=1, opacity=0.4)
        fig_ret.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(color="white", gridcolor=GRID, ticksuffix="%"),
            yaxis=dict(color="white"),
            margin=dict(t=10, b=10),
            height=max(300, len(ret_df) * 40),
        )
        st.plotly_chart(fig_ret, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Holdings table
# ─────────────────────────────────────────────────────────────────────────────

with tab_holdings:
    st.subheader("Current Holdings (live prices)")
    rows = []
    for t, h in sorted(
        holdings_enriched.items(),
        key=lambda x: (x[1].get("return_pct") or 0),
        reverse=True,
    ):
        rows.append({
            "Ticker":          t,
            "Name":            h["name"],
            "Bucket":          h["bucket"].capitalize(),
            "Region":          h["region"],
            "Category":        h["category"],
            "Units":           round(h["total_units"], 4),
            "Price (native)":  h.get("current_price"),
            "Currency":        h["currency"],
            "Invested ($)":    h["invested_usd"],
            "Value ($)":       h.get("current_value_usd"),
            "P&L ($)":         h.get("pnl_usd"),
            "Return %":        h.get("return_pct"),
            "Trade On":        h["trade_on"],
        })

    df = pd.DataFrame(rows)

    def _color_val(val):
        if isinstance(val, (int, float)):
            return "color: #2ECC71" if val >= 0 else "color: #E74C3C"
        return ""

    st.dataframe(
        df.style.map(_color_val, subset=["P&L ($)", "Return %"]),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇ Download Holdings JSON",
        data=json.dumps(holdings_enriched, indent=2, default=str),
        file_name="holdings.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Ledger history (all recorded entries)
# ─────────────────────────────────────────────────────────────────────────────

with tab_history:
    st.subheader(f"All Investment Records ({summary['months']} months)")

    for entry in reversed(ledger.get("entries", [])):
        with st.expander(
            f"**{entry['month']}**  ·  {entry['date']}  ·  "
            f"${entry['total_invested_usd']:.2f} invested  ·  "
            f"{len(entry['positions'])} positions  ·  "
            f"USD/INR: {entry['usd_inr_rate']:.2f}"
        ):
            pos_rows = [
                {
                    "Ticker":     p["ticker"],
                    "Bucket":     p["bucket"].capitalize(),
                    "Region":     p["region"],
                    "Category":   p["category"],
                    "Score":      p["consensus_score"],
                    "Price":      p["price_native"],
                    "Currency":   p.get("currency", "USD" if p["region"] != "BSE" else "INR"),
                    "Units":      p["units_bought"],
                    "Invested ($)": p["monthly_usd"],
                }
                for p in entry["positions"]
            ]
            st.dataframe(pd.DataFrame(pos_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Rankings generated: {entry.get('rankings_generated_at', '—')}  ·  "
                f"Core: {_usd(entry.get('core_budget', 0))}  ·  "
                f"Satellite: {_usd(entry.get('satellite_budget', 0))}"
            )

    st.divider()
    st.download_button(
        "⬇ Download Full Ledger JSON",
        data=json.dumps(ledger, indent=2, default=str),
        file_name="portfolio_ledger.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Last allocation plan
# ─────────────────────────────────────────────────────────────────────────────

with tab_allocation:
    last_entry = ledger["entries"][-1]
    st.subheader(f"Allocation from {last_entry['month']}  ({last_entry['date']})")
    st.caption(
        f"Rankings: {last_entry.get('rankings_generated_at', '—')}  ·  "
        f"SIP: {_usd(last_entry.get('sip_amount', 0))}  ·  "
        f"Core: {_usd(last_entry.get('core_budget', 0))} / "
        f"Satellite: {_usd(last_entry.get('satellite_budget', 0))}"
    )

    core_pos = [p for p in last_entry["positions"] if p["bucket"] == "core"]
    sat_pos  = [p for p in last_entry["positions"] if p["bucket"] == "satellite"]

    def _alloc_table(positions):
        return pd.DataFrame([
            {
                "#":           i + 1,
                "Ticker":      p["ticker"],
                "Region":      p["region"],
                "Category":    p["category"],
                "Score":       round(p["consensus_score"], 4),
                "Weight":      f"{p['weight']*100:.1f}%",
                "Monthly ($)": p["monthly_usd"],
                "Price":       p["price_native"],
                "Currency":    p.get("currency", "USD" if p["region"] != "BSE" else "INR"),
                "Units":       p["units_bought"],
            }
            for i, p in enumerate(positions)
        ])

    if core_pos:
        st.markdown(f"**★ Core** — {_usd(last_entry.get('core_budget', 0))}/month")
        st.dataframe(_alloc_table(core_pos), hide_index=True, use_container_width=True)

    if sat_pos:
        st.markdown(f"**◈ Satellite** — {_usd(last_entry.get('satellite_budget', 0))}/month")
        st.dataframe(_alloc_table(sat_pos), hide_index=True, use_container_width=True)

    # Allocation donut from last entry
    all_pos  = last_entry["positions"]
    sip_amt  = last_entry.get("sip_amount", 500)
    fig_alloc = go.Figure(go.Pie(
        labels        = [p["ticker"] for p in all_pos],
        values        = [p["monthly_usd"] for p in all_pos],
        hole          = 0.45,
        marker        = dict(
            colors=[CORE_COLOR if p["bucket"] == "core" else SATELLITE_COLOR for p in all_pos],
            line=dict(color="#0E1117", width=2),
        ),
        textinfo      = "label+percent",
        hovertemplate = "<b>%{label}</b><br>$%{value:.2f}/month<extra></extra>",
    ))
    fig_alloc.add_annotation(
        text=f"${sip_amt:.0f}/mo", x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color="white"),
    )
    fig_alloc.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(color="white")),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Portfolio timeline
# ─────────────────────────────────────────────────────────────────────────────

with tab_timeline:
    tl_df = _monthly_timeline_df(ledger, holdings_enriched)

    if not tl_df.empty:
        st.subheader("Portfolio Growth vs Cumulative Investment")
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=tl_df["Month"], y=tl_df["Invested"],
            name="Cumulative Invested",
            line=dict(color="#888888", dash="dot", width=2),
            hovertemplate="<b>%{x}</b><br>Invested: $%{y:,.2f}<extra></extra>",
        ))
        fig_growth.add_trace(go.Scatter(
            x=tl_df["Month"], y=tl_df["Value"],
            name="Portfolio Value (est.)",
            line=dict(color=CORE_COLOR, width=3),
            fill="tonexty",
            fillcolor="rgba(76,155,232,0.12)",
            hovertemplate="<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>",
        ))
        # Add current live value as final dot
        if totals["current_value"] and not tl_df.empty:
            fig_growth.add_trace(go.Scatter(
                x=[tl_df["Month"].iloc[-1]],
                y=[totals["current_value"]],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=POSITIVE_COLOR if (totals["total_pnl"] or 0) >= 0 else NEGATIVE_COLOR,
                    symbol="diamond",
                ),
                text=[f"  Live: {_usd(totals['current_value'])}"],
                textposition="middle right",
                name="Live Value",
                hoverinfo="skip",
            ))
        fig_growth.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(color="white", gridcolor=GRID),
            yaxis=dict(color="white", gridcolor=GRID, tickprefix="$"),
            legend=dict(font=dict(color="white")),
            hovermode="x unified",
            margin=dict(t=20, b=20),
            height=400,
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("Monthly P&L")
        tl_df["Color"] = tl_df["P&L"].apply(
            lambda x: POSITIVE_COLOR if x >= 0 else NEGATIVE_COLOR
        )
        fig_pnl = go.Figure(go.Bar(
            x=tl_df["Month"], y=tl_df["P&L"],
            marker_color=tl_df["Color"],
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
        ))
        fig_pnl.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
        fig_pnl.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(color="white", gridcolor=GRID),
            yaxis=dict(color="white", gridcolor=GRID, tickprefix="$"),
            margin=dict(t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.subheader("Month-by-Month Table")
        st.dataframe(tl_df.drop(columns=["Color"]), hide_index=True, use_container_width=True)
