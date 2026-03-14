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
for _p in (str(_FIN_ROOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


def _color_val(val: Any) -> str:
    """Pandas Styler: green for non-negative numbers, red for negative."""
    if isinstance(val, (int, float)):
        return "color: #2ECC71" if val >= 0 else "color: #E74C3C"
    return ""


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
        _last_entry_sidebar = ledger_raw["entries"][-1] if ledger_raw.get("entries") else {}
        _va_tag = ""
        if _last_entry_sidebar.get("va_triggered"):
            _va_tag = f"  ·  VA x{_last_entry_sidebar.get('va_multiplier', 1.0):.2f}"
        st.success(
            f"✓ **{summary['months']} months** recorded\n\n"
            f"First: {summary['first_investment']}\n\n"
            f"Last: {summary['last_investment']}{_va_tag}\n\n"
            f"Total invested: **{_usd(summary['total_invested_usd'])}**"
        )
        _va_months = [e for e in ledger_raw.get("entries", []) if e.get("va_triggered")]
        if _va_months:
            st.info(f"⚡ **{len(_va_months)} VA month(s)** — crash accumulator fired")
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

    # ── Backtest panel ────────────────────────────────────────────────────────
    st.subheader("🔬 Backtest")
    bt_mode = st.toggle(
        "Enable Backtest",
        value=st.session_state.get("bt_mode", False),
        help="Simulate month-by-month SIP over a historical period",
    )
    st.session_state["bt_mode"] = bt_mode

    if bt_mode:
        bt_start = st.date_input(
            "Start date",
            value=datetime(datetime.now().year - 1, 1, 1).date(),
            key="sb_bt_start",
        )
        bt_end = st.date_input(
            "End date",
            value=datetime.now().date(),
            key="sb_bt_end",
        )
        bt_day = st.number_input(
            "Day of month (SIP date)",
            min_value=1, max_value=27, value=1, step=1,
            help="Day within each month on which to execute the SIP",
            key="sb_bt_day",
        )
        bt_sip = st.number_input(
            "Monthly SIP (USD)",
            min_value=50.0, max_value=50000.0,
            value=500.0, step=50.0,
            key="sb_bt_sip",
        )
        bt_core_pct = st.slider(
            "Core / Satellite split",
            min_value=50, max_value=90, value=70, step=5,
            format="%d%% core",
            help="Percentage of SIP allocated to core ETFs; remainder goes to satellite.",
            key="sb_bt_core_pct",
        )

        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

        if run_bt:
            if bt_start >= bt_end:
                st.error("Start must be before end date.")
            else:
                st.session_state["bt_running"] = {
                    "start":    bt_start,
                    "end":      bt_end,
                    "day":      int(bt_day),
                    "sip":      bt_sip,
                    "core_pct": bt_core_pct / 100.0,
                }

    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  Guard — need at least one ledger entry
# ══════════════════════════════════════════════════════════════════════════════

if summary["months"] == 0 and not st.session_state.get("bt_mode"):
    st.title("📊 SIP Simulator")
    st.info(
        "**No investment records found yet.**\n\n"
        "Run the scheduler to record your first monthly investment:\n"
        "```bash\npython -m simulator.scheduler --now\n```\n\n"
        "Then click **🔄 Refresh data** in the sidebar, or use the **🔬 Backtest** "
        "toggle to simulate a historical SIP without ledger data."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  Load & enrich data  (only when ledger has entries)
# ══════════════════════════════════════════════════════════════════════════════

_has_ledger = summary["months"] > 0

if _has_ledger:
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

    # ── Header + top metrics ──────────────────────────────────────────────────
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

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_portfolio, tab_holdings, tab_history, tab_allocation, tab_timeline, tab_va = st.tabs([
        "🗂 Portfolio",
        "💼 Holdings",
        "📒 Ledger History",
        "📐 Last Allocation",
        "📅 Timeline",
        "⚡ Value Averaging",
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest panel  (sidebar-triggered; renders even when no ledger entries)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("bt_mode"):

    # ── Execute if a run was requested ───────────────────────────────────────
    if "bt_running" in st.session_state:
        _params = st.session_state.pop("bt_running")
        from simulator.backtest import run_historical_backtest

        _log_lines: list = []
        with st.status("Running backtest …", expanded=True) as _status:
            def _bt_progress(msg: str) -> None:
                _log_lines.append(msg)
                _status.write(msg)

            try:
                _bt_result = run_historical_backtest(
                    start_date        = _params["start"],
                    end_date          = _params["end"],
                    sip_amount        = _params["sip"],
                    day_of_month      = _params["day"],
                    core_pct          = _params.get("core_pct", 0.70),
                    progress_callback = _bt_progress,
                )
                st.session_state["backtest_result"] = _bt_result
                _status.update(
                    label="Backtest complete — {} months run.".format(_bt_result["months_run"]),
                    state="complete",
                )
            except Exception as _exc:
                _status.update(label="Backtest failed: {}".format(_exc), state="error")
                st.error(str(_exc))

    # ── Results ───────────────────────────────────────────────────────────────
    if "backtest_result" in st.session_state:
        r = st.session_state["backtest_result"]

        st.divider()
        st.header("🔬 Backtest Results")
        _core_pct_used  = int(round(r.get("core_pct", 0.70) * 100))
        _sat_pct_used   = 100 - _core_pct_used
        st.caption(
            "Period: **{}** → **{}**  ·  {} months  ·  "
            "SIP ${:.0f}/month  ·  Core/Sat {}%/{}%  ·  buy-only  ·  no future data leakage".format(
                r["start_date"], r["end_date"], r["months_run"], r["sip_amount"],
                _core_pct_used, _sat_pct_used,
            )
        )

        # Summary metrics
        _va_months_bt = [e for e in r["monthly_entries"] if e.get("va_triggered")]
        _va_extra_usd = sum(
            e.get("effective_sip", e["sip_amount"]) - e["sip_amount"]
            for e in r["monthly_entries"]
        )
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        bm1.metric("Total Invested",  _usd(r["total_invested_usd"]))
        bm2.metric(
            "Portfolio Value",
            _usd(r["current_value_usd"]),
            delta=_usd(r["total_pnl_usd"]),
            delta_color="normal" if r["total_pnl_usd"] >= 0 else "inverse",
        )
        bm3.metric("Total Return", _pct(r["total_return_pct"]))
        bm4.metric("CAGR",         _pct(r["cagr"]))
        bm5.metric(
            "VA Events",
            f"{len(_va_months_bt)} month(s)",
            delta=f"+{_usd(_va_extra_usd)} extra" if _va_extra_usd > 0 else "none triggered",
            delta_color="normal" if _va_extra_usd > 0 else "off",
        )

        if r["skipped_months"]:
            st.warning("Skipped months: " + ", ".join(r["skipped_months"]))

        # ── Chart A: Portfolio growth ─────────────────────────────────────────
        st.subheader("Portfolio Growth vs Cumulative Investment")
        _cum_inv   = 0.0
        _cum_units: Dict[str, float] = {}
        _tl_rows   = []

        for _entry in r["monthly_entries"]:
            _usd_inr_e = _entry["usd_inr_rate"]
            _cum_inv  += _entry["total_invested_usd"]
            for _pos in _entry["positions"]:
                _t = _pos["ticker"]
                _cum_units[_t] = _cum_units.get(_t, 0.0) + _pos["units_bought"]

            _port_val = 0.0
            for _pos in _entry["positions"]:
                _u = _cum_units.get(_pos["ticker"], 0.0)
                _p = _pos["price_native"]
                if _p > 0:
                    _port_val += (_u * _p / _usd_inr_e) if _pos["region"] == "BSE" else (_u * _p)

            _tl_rows.append({
                "Month":    _entry["month"],
                "Invested": round(_cum_inv, 2),
                "Value":    round(_port_val, 2),
                "P&L":      round(_port_val - _cum_inv, 2),
            })

        if _tl_rows:
            _bt_df = pd.DataFrame(_tl_rows)
            _fig_growth = go.Figure()
            _fig_growth.add_trace(go.Scatter(
                x=_bt_df["Month"], y=_bt_df["Invested"],
                name="Cumulative Invested",
                line=dict(color="#888888", dash="dot", width=2),
                hovertemplate="<b>%{x}</b><br>Invested: $%{y:,.2f}<extra></extra>",
            ))
            _fig_growth.add_trace(go.Scatter(
                x=_bt_df["Month"], y=_bt_df["Value"],
                name="Portfolio Value (est.)",
                line=dict(color=CORE_COLOR, width=3),
                fill="tonexty",
                fillcolor="rgba(76,155,232,0.12)",
                hovertemplate="<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>",
            ))
            # Benchmark lines: 100% USCA (US core anchor) and 100% NIFTYBEES.NS
            _bench_specs = [
                ("USCA",          "100% USCA (US Large-Cap)", "#9B59B6"),
                ("NIFTYBEES.NS",  "100% NIFTY (India)",       "#E67E22"),
            ]
            for _bt_ticker, _bt_label, _bt_color in _bench_specs:
                _bt_units = 0.0
                _bt_rows  = []
                for _be in r["monthly_entries"]:
                    _bt_pos = next(
                        (p for p in _be["positions"] if p["ticker"] == _bt_ticker), None
                    )
                    if not _bt_pos or not _bt_pos["price_native"]:
                        continue
                    _bp      = _bt_pos["price_native"]
                    _busd_inr = _be["usd_inr_rate"]
                    _bsip    = _be["sip_amount"]
                    # Accumulate units as if 100% SIP went here
                    if _bt_ticker.endswith(".NS"):
                        _bt_units += (_bsip * _busd_inr) / _bp
                        _bval = (_bt_units * _bp) / _busd_inr
                    else:
                        _bt_units += _bsip / _bp
                        _bval = _bt_units * _bp
                    _bt_rows.append({"Month": _be["month"], "Value": round(_bval, 2)})
                if _bt_rows:
                    _bt_bench_df = pd.DataFrame(_bt_rows)
                    _fig_growth.add_trace(go.Scatter(
                        x=_bt_bench_df["Month"], y=_bt_bench_df["Value"],
                        name=_bt_label,
                        line=dict(color=_bt_color, width=2, dash="dash"),
                        hovertemplate=f"<b>%{{x}}</b><br>{_bt_label}: $%{{y:,.2f}}<extra></extra>",
                    ))

            if r["current_value_usd"] and not _bt_df.empty:
                _live_color = POSITIVE_COLOR if r["total_pnl_usd"] >= 0 else NEGATIVE_COLOR
                _fig_growth.add_trace(go.Scatter(
                    x=[_bt_df["Month"].iloc[-1]],
                    y=[r["current_value_usd"]],
                    mode="markers+text",
                    marker=dict(size=12, color=_live_color, symbol="diamond"),
                    text=["  Live: " + _usd(r["current_value_usd"])],
                    textposition="middle right",
                    name="Live Value",
                    hoverinfo="skip",
                ))
            # Mark VA trigger months with vertical lines
            for _va_e in _va_months_bt:
                _va_row = next((row for row in _tl_rows if row["Month"] == _va_e["month"]), None)
                if _va_row:
                    _va_color = "#FFD700" if _va_e.get("va_multiplier", 1.0) < 1.5 else "#FF6B35"
                    _fig_growth.add_vline(
                        x=_va_e["month"],
                        line_color=_va_color, line_dash="dot", line_width=1.5,
                        annotation_text=f"VA x{_va_e.get('va_multiplier', 1.0):.2f}",
                        annotation_font_color=_va_color,
                        annotation_font_size=10,
                    )
            _fig_growth.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(color="white", gridcolor=GRID),
                yaxis=dict(color="white", gridcolor=GRID, tickprefix="$"),
                legend=dict(font=dict(color="white")),
                hovermode="x unified",
                height=400,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(_fig_growth, use_container_width=True)

        # ── Chart B: Per-ETF returns ──────────────────────────────────────────
        st.subheader("Per-ETF Returns (live prices)")
        if r["holdings"]:
            _h_rows = [
                {
                    "Ticker":   t,
                    "Return %": h["return_pct"],
                    "P&L ($)":  h["pnl_usd"],
                    "Bucket":   h.get("bucket", "?").capitalize(),
                    "Region":   h.get("region", "?"),
                }
                for t, h in r["holdings"].items()
                if h.get("return_pct") is not None
            ]
            _h_df = pd.DataFrame(_h_rows).sort_values("Return %")
            _fig_ret = go.Figure(go.Bar(
                x=_h_df["Return %"], y=_h_df["Ticker"],
                orientation="h",
                marker_color=[POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in _h_df["Return %"]],
                customdata=_h_df[["P&L ($)", "Bucket"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Return: %{x:.2f}%<br>"
                    "P&L: $%{customdata[0]:,.2f}<br>Bucket: %{customdata[1]}<extra></extra>"
                ),
            ))
            _fig_ret.add_vline(x=0, line_color="white", line_width=1, opacity=0.4)
            _fig_ret.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(color="white", gridcolor=GRID, ticksuffix="%"),
                yaxis=dict(color="white"),
                height=max(300, len(_h_df) * 40),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(_fig_ret, use_container_width=True)

        # ── Chart C: Monthly ETF selection heatmap ────────────────────────────
        st.subheader("Monthly ETF Selection Heatmap")
        _all_tickers_seen = sorted({
            pos["ticker"]
            for _entry in r["monthly_entries"]
            for pos in _entry["positions"]
        })
        _hm_rows = [
            [
                {pos["ticker"]: pos["consensus_score"] for pos in _entry["positions"]}.get(t, 0.0)
                for t in _all_tickers_seen
            ]
            for _entry in r["monthly_entries"]
        ]
        if _hm_rows and _all_tickers_seen:
            _fig_hm = go.Figure(go.Heatmap(
                z=_hm_rows,
                x=_all_tickers_seen,
                y=[e["month"] for e in r["monthly_entries"]],
                colorscale="Blues",
                zmin=0.0, zmax=1.0,
                hovertemplate="<b>%{x}</b><br>%{y}<br>Score: %{z:.3f}<extra></extra>",
            ))
            _fig_hm.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(color="white", tickangle=-45),
                yaxis=dict(color="white"),
                height=max(300, len(_hm_rows) * 35),
                margin=dict(t=10, b=60),
            )
            st.plotly_chart(_fig_hm, use_container_width=True)

        # ── Per-month expandable details ──────────────────────────────────────
        st.subheader("Month-by-Month Details")
        for _entry in reversed(r["monthly_entries"]):
            _scorer_badge = "🤖 Gemini" if _entry.get("scorer") == "gemini" else "📊 VADER"
            _va_triggered = _entry.get("va_triggered", False)
            _va_mult      = _entry.get("va_multiplier", 1.0)
            _eff_sip      = _entry.get("effective_sip", _entry["sip_amount"])
            _va_badge     = f"  ·  ⚡ VA x{_va_mult:.2f}" if _va_triggered else ""
            with st.expander(
                "**{}**  ·  {}  ·  {}{}  ·  {} positions  ·  {}".format(
                    _entry["month"], _entry["date"],
                    _usd(_eff_sip), _va_badge,
                    len(_entry["positions"]), _scorer_badge,
                )
            ):
                if _va_triggered:
                    st.info(
                        f"⚡ **Crash-Accumulator VA fired** — "
                        f"Base SIP: {_usd(_entry['sip_amount'])}  →  "
                        f"Effective SIP: {_usd(_eff_sip)}  (x{_va_mult:.2f})"
                    )
                st.dataframe(pd.DataFrame([
                    {
                        "Ticker":    p["ticker"],
                        "Bucket":    p["bucket"].capitalize(),
                        "Region":    p["region"],
                        "Category":  p["category"],
                        "Consensus": round(p["consensus_score"], 4),
                        "Sentiment": round(p["sentiment_score"], 4),
                        "Expense":   round(p["expense_score"], 4),
                        "Weight %":  "{:.1f}%".format(p["weight"] * 100),
                        "USD/mo":    p["monthly_usd"],
                        "Price":     p["price_native"],
                        "Currency":  p["currency"],
                        "Units":     p["units_bought"],
                    }
                    for p in _entry["positions"]
                ]), hide_index=True, use_container_width=True)
                if _entry.get("boom_triggers"):
                    st.caption("Boom triggers: " + "  ·  ".join(
                        "`{}`".format(t) for t in _entry["boom_triggers"]
                    ))
                if _entry.get("macro_summary"):
                    st.caption(_entry["macro_summary"])

        # ── Holdings summary + download ───────────────────────────────────────
        st.divider()
        st.subheader("Aggregate Holdings")
        if r["holdings"]:
            _hold_df = pd.DataFrame([
                {
                    "Ticker":         t,
                    "Name":           h.get("name", t),
                    "Bucket":         h.get("bucket", "?").capitalize(),
                    "Region":         h.get("region", "?"),
                    "Units":          h["total_units"],
                    "Price (native)": h["current_price_native"],
                    "Currency":       h.get("currency", "USD"),
                    "Invested ($)":   h["invested_usd"],
                    "Value ($)":      h["value_usd"],
                    "P&L ($)":        h["pnl_usd"],
                    "Return %":       h["return_pct"],
                }
                for t, h in sorted(
                    r["holdings"].items(),
                    key=lambda x: x[1].get("return_pct", 0),
                    reverse=True,
                )
            ])
            st.dataframe(
                _hold_df.style.map(_color_val, subset=["P&L ($)", "Return %"]),
                hide_index=True,
                use_container_width=True,
            )

        st.download_button(
            "⬇ Download Backtest JSON",
            data=json.dumps(r, indent=2, default=str),
            file_name="backtest_{}_{}.json".format(r["start_date"], r["end_date"]),
            mime="application/json",
        )

# ── Stop here if no ledger — tab variables not defined below ─────────────────
if not _has_ledger:
    st.stop()

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
        _e_va      = entry.get("va_triggered", False)
        _e_va_mult = entry.get("va_multiplier", 1.0)
        _e_eff_sip = entry.get("effective_sip", entry.get("sip_amount", entry["total_invested_usd"]))
        _e_scorer  = "🤖 Gemini" if entry.get("scorer") == "gemini" else "📊 VADER"
        _e_va_tag  = f"  ·  ⚡ VA x{_e_va_mult:.2f}" if _e_va else ""
        with st.expander(
            f"**{entry['month']}**  ·  {entry['date']}  ·  "
            f"${_e_eff_sip:.2f} invested{_e_va_tag}  ·  "
            f"{len(entry['positions'])} positions  ·  "
            f"USD/INR: {entry['usd_inr_rate']:.2f}  ·  {_e_scorer}"
        ):
            if _e_va:
                st.info(
                    f"⚡ **Crash-Accumulator VA** — "
                    f"Base: {_usd(entry.get('sip_amount', 0))}  →  "
                    f"Effective: {_usd(_e_eff_sip)}  (x{_e_va_mult:.2f})"
                )
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
            _boom = entry.get("boom_triggers", [])
            _macro = entry.get("macro_summary", "")
            st.caption(
                f"Rankings generated: {entry.get('rankings_generated_at', '—')}  ·  "
                f"Core: {_usd(entry.get('core_budget', 0))}  ·  "
                f"Satellite: {_usd(entry.get('satellite_budget', 0))}"
                + (f"  ·  Boom: {', '.join(_boom)}" if _boom else "")
            )
            if _macro:
                st.caption(_macro[:200])

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
    _la_va      = last_entry.get("va_triggered", False)
    _la_va_mult = last_entry.get("va_multiplier", 1.0)
    _la_eff_sip = last_entry.get("effective_sip", last_entry.get("sip_amount", 0))
    _la_scorer  = last_entry.get("scorer", "vader")
    st.subheader(f"Allocation from {last_entry['month']}  ({last_entry['date']})")
    st.caption(
        f"Rankings: {last_entry.get('rankings_generated_at', '—')}  ·  "
        f"Scorer: {'Gemini' if _la_scorer == 'gemini' else 'VADER'}  ·  "
        f"Base SIP: {_usd(last_entry.get('sip_amount', 0))}  ·  "
        f"Effective SIP: {_usd(_la_eff_sip)}  ·  "
        f"Core: {_usd(last_entry.get('core_budget', 0))} / "
        f"Satellite: {_usd(last_entry.get('satellite_budget', 0))}"
    )
    if _la_va:
        st.warning(f"⚡ **Crash-Accumulator VA fired this month** — x{_la_va_mult:.2f} multiplier applied")

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


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — Value Averaging history
# ─────────────────────────────────────────────────────────────────────────────

with tab_va:
    st.subheader("Crash-Accumulator Value-Averaging (VA) History")
    st.caption(
        "VA fires when panic sentiment is detected.  "
        "Tier 1 (momentum > -15%): x1.20 | Tier 2 (momentum <= -15%): x1.50"
    )

    _all_entries = ledger.get("entries", [])
    _va_rows = []
    for _e in _all_entries:
        _va_rows.append({
            "Month":          _e["month"],
            "Base SIP ($)":   _e.get("sip_amount", 0),
            "Effective ($)":  _e.get("effective_sip", _e.get("sip_amount", 0)),
            "VA Triggered":   "Yes" if _e.get("va_triggered") else "No",
            "Multiplier":     _e.get("va_multiplier", 1.0),
            "Extra Deployed": round(
                _e.get("effective_sip", _e.get("sip_amount", 0)) - _e.get("sip_amount", 0), 2
            ),
            "Scorer":         "Gemini" if _e.get("scorer") == "gemini" else "VADER",
        })

    if _va_rows:
        _va_df = pd.DataFrame(_va_rows)

        # Stacked bar: base SIP + extra (VA top-up) per month
        _fig_va = go.Figure()
        _fig_va.add_trace(go.Bar(
            x=_va_df["Month"], y=_va_df["Base SIP ($)"],
            name="Base SIP",
            marker_color=CORE_COLOR,
            hovertemplate="<b>%{x}</b><br>Base SIP: $%{y:,.2f}<extra></extra>",
        ))
        _va_extra_series = _va_df["Extra Deployed"].clip(lower=0)
        _fig_va.add_trace(go.Bar(
            x=_va_df["Month"], y=_va_extra_series,
            name="VA Top-up",
            marker_color="#FFD700",
            hovertemplate="<b>%{x}</b><br>VA Extra: $%{y:,.2f}<extra></extra>",
        ))
        _fig_va.update_layout(
            barmode="stack",
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(color="white", gridcolor=GRID),
            yaxis=dict(color="white", gridcolor=GRID, tickprefix="$"),
            legend=dict(font=dict(color="white")),
            height=320,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(_fig_va, use_container_width=True)

        # Summary stats
        _total_va_months = int(_va_df["VA Triggered"].eq("Yes").sum())
        _total_va_extra  = round(_va_df["Extra Deployed"].sum(), 2)
        _total_base      = round(_va_df["Base SIP ($)"].sum(), 2)
        _total_eff       = round(_va_df["Effective ($)"].sum(), 2)

        _vc1, _vc2, _vc3, _vc4 = st.columns(4)
        _vc1.metric("VA Months", str(_total_va_months))
        _vc2.metric("Extra Capital Deployed", _usd(_total_va_extra))
        _vc3.metric("Base SIP Total", _usd(_total_base))
        _vc4.metric("Effective Total", _usd(_total_eff))

        # Scorer breakdown
        _gemini_months = int(_va_df["Scorer"].eq("Gemini").sum())
        _vader_months  = int(_va_df["Scorer"].eq("VADER").sum())
        st.caption(
            f"Scorer breakdown:  🤖 Gemini: {_gemini_months} months  ·  "
            f"📊 VADER: {_vader_months} months"
        )

        # Detailed table
        st.subheader("Month-by-Month VA Log")
        st.dataframe(
            _va_df.style.map(
                lambda v: "color: #FFD700; font-weight: bold" if v == "Yes" else "",
                subset=["VA Triggered"]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No ledger entries yet. Run the scheduler to record your first investment.")
