# app_simple.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting backend
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False

# Fallback HTTP for Stooq
import requests
from io import StringIO

# Primary data
try:
    import yfinance as yf
except Exception as e:
    st.error("This app needs the 'yfinance' package.\n\nOn your machine, run:\n\npip install -r requirements.txt")
    raise e

st.set_page_config(page_title="Friendly Stock Projector", page_icon="📈", layout="wide")

# ==============================
# Formatting helpers
# ==============================
def _fmt_usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"${x:,.2f}"

def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.2%}"

def _normalize_symbol(sym: str) -> str:
    return sym.strip().upper().replace("/", "-").replace(".", "-")

# ==============================
# Data loaders (robust with fallback)
# ==============================
@st.cache_data(show_spinner=False)
def _fetch_stooq_equity(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    """Fallback to Stooq CSV for US equities/ETFs: ticker.us"""
    sym = symbol.strip().upper()
    stq = f"{sym.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stq}&i=d"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200 or not r.text or r.text.lower().startswith("<!doctype"):
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Date" not in df or "Close" not in df:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df[["Close"]].rename(columns={"Close": "close"}).dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.loc[start:]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_prices(symbol: str, years: int) -> Tuple[pd.DataFrame, str]:
    """Download daily prices for 1 symbol. Returns (DataFrame, source_label)."""
    symbol = _normalize_symbol(symbol)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years + 1)

    # Try yfinance download (few retries)
    for attempt in range(2):
        try:
            df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True, interval="1d", threads=False)
            if df is not None and not df.empty:
                df = df.rename(columns={"Close": "close"})[["close"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df, "Yahoo Finance"
        except Exception:
            pass
        time.sleep(0.4 * (attempt + 1))

    # Ticker().history fallback
    try:
        hist = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            hist = hist.rename(columns={"Close": "close"})[["close"]].dropna()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.loc[start:]
            if not hist.empty:
                return hist, "Yahoo Finance"
    except Exception:
        pass

    # Stooq fallback (equities/ETFs)
    stq = _fetch_stooq_equity(symbol, start=start)
    if not stq.empty:
        return stq, "Stooq"

    return pd.DataFrame(), ""

# ==============================
# Estimation (simple; uses last ~90 days by default)
# ==============================
@dataclass
class EstimationConfig:
    lookback_days_for_stats: int = 90  # estimate recent behavior

def estimate_drift_vol(close: pd.Series, cfg: EstimationConfig) -> Tuple[float, float]:
    """Estimate daily drift (mu) and daily volatility (sigma) from recent returns."""
    lr = np.log(close).diff().dropna()
    if len(lr) < 20:
        win = lr
    else:
        win = lr.tail(cfg.lookback_days_for_stats)
    mu = float(win.mean()) if len(win) else 0.0
    sig = float(win.std(ddof=1)) if len(win) else 0.02
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sig) or sig <= 0:
        sig = float(lr.std(ddof=1)) if len(lr) > 2 else 0.02
        if not np.isfinite(sig) or sig <= 0:
            sig = 0.02
    return mu, sig

# ==============================
# Regime detection (hidden, no sliders)
# ==============================
@st.cache_data(show_spinner=False)
def get_spy_series(years: int) -> pd.Series:
    df, _src = fetch_prices("SPY", years)
    if df.empty:
        return pd.Series(dtype=float)
    return df["close"].dropna()

def determine_regime(spy_close: pd.Series) -> str:
    """Bull/Neutral/Bear by 63-day momentum thresholds from SPY."""
    if spy_close is None or spy_close.dropna().empty:
        return "Neutral"
    mom63 = spy_close.pct_change(63)
    if mom63.dropna().empty:
        return "Neutral"
    # Use historical quantiles to set thresholds (approx terciles)
    hi = float(mom63.quantile(0.66))
    lo = float(mom63.quantile(0.33))
    last = float(mom63.dropna().iloc[-1]) if len(mom63.dropna()) else 0.0
    if last >= hi:
        return "Bull"
    if last <= lo:
        return "Bear"
    return "Neutral"

def regime_adjust_mu(mu: float, regime: str) -> float:
    """Industry-standard style multipliers (modest)."""
    mult = {"Bull": 1.30, "Neutral": 1.00, "Bear": 0.60}.get(regime, 1.00)
    return mu * mult

# ==============================
# GARCH-like volatility (hidden)
# ==============================
@dataclass
class GarchParams:
    omega: float = 1e-6       # baseline variance
    alpha: float = 0.06       # response to new shocks
    beta: float  = 0.90       # persistence

def simulate_path_realistic(
    S0: float,
    mu: float,
    sigma0: float,
    days: int,
    n_paths: int,
    seed: Optional[int],
    regime: str,
    mean_revert_kappa: float = 0.002,   # small daily pull toward long-run
    shock_prob: float = 0.03,           # small macro shock probability
    shock_sigma_mult: float = 2.0,      # ~2 sigma downward shock
    gparams: GarchParams = GarchParams(),
    long_run_window: int = 252,         # proxy for long-run level
) -> Tuple[np.ndarray, dict]:
    """Simulate prices with regime-adjusted drift, GARCH-like vol, mean reversion, and downside shocks."""
    rng = np.random.default_rng(int(seed) if seed is not None else None)

    # Regime-adjusted drift
    mu_adj = regime_adjust_mu(mu, regime)

    # Long-run anchor (log-price median over ~1y). If not enough history, anchor at S0.
    long_run_log = math.log(S0)

    # Initialize conditional variance from sigma0
    var0 = max(1e-10, float(sigma0)**2)
    omega, alpha, beta = gparams.omega, gparams.alpha, gparams.beta

    # Pre-draw normals and shock uniforms
    Z = rng.standard_normal(size=(n_paths, days))
    U = rng.random(size=(n_paths, days))

    prices = np.empty((n_paths, days + 1), dtype=float)
    prices[:, 0] = S0

    # Start variance same for all paths
    var_t = np.full(shape=(n_paths,), fill_value=var0, dtype=float)
    logP_t = np.full(shape=(n_paths,), fill_value=math.log(S0), dtype=float)

    for t in range(days):
        # GARCH(1,1) update for variance using previous step's shock. We don't have realized r_{t-1} yet for t=0,
        # so we use var0 as a starting point (this is a standard warm start simplification).
        # For simulation, we update var_t pathwise after we draw r_t.
        sigma_t = np.sqrt(np.maximum(var_t, 1e-12))

        # Mean reversion term toward long-run log price (acts on log returns)
        mr_term = -mean_revert_kappa * (logP_t - long_run_log)

        # Random shock term
        eps = Z[:, t]

        # Downside macro-shock: with small probability, add an extra negative move
        shock_mask = U[:, t] < shock_prob
        shock = np.zeros_like(eps)
        if shock_mask.any():
            shock[shock_mask] = -shock_sigma_mult * sigma_t[shock_mask] * np.abs(eps[shock_mask])

        # Log-return for this step
        r_t = (mu_adj + mr_term) + sigma_t * eps + shock

        # Update log price and store
        logP_t = logP_t + r_t
        prices[:, t + 1] = np.exp(logP_t)

        # Update conditional variance via GARCH(1,1): var_{t+1} = omega + alpha * r_t^2 + beta * var_t
        var_t = omega + alpha * (r_t**2) + beta * var_t

    # Summary percentiles / mean path
    pct = {
        "p10": np.percentile(prices, 10, axis=0),
        "p20": np.percentile(prices, 20, axis=0),
        "p50": np.percentile(prices, 50, axis=0),
        "p80": np.percentile(prices, 80, axis=0),
        "p90": np.percentile(prices, 90, axis=0),
    }
    mean_path = prices.mean(axis=0)
    return mean_path, pct

def chance_of_down_day(mu: float, sig: float) -> float:
    """
    Under a normal assumption, P(return < 0) = Phi((-mu)/sig).
    If sigma is ~0, treat as ~50%.
    """
    if sig <= 1e-12:
        return 0.5
    z = (-mu) / sig
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ==============================
# UI
# ==============================
st.title("📈 Friendly Stock Projector")
st.caption("A plain-English stock projection—no jargon required.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Stock symbol", value="AAPL").strip().upper()
    lookback_years = st.slider("How much history to use", 1, 10, 5, help="We learn from recent price moves in this window.")
    horizon = st.select_slider("How far ahead to project", options=[21, 63, 126, 252], value=126,
                               format_func=lambda d: {21: "1 month (~21 days)", 63: "3 months (~63 days)", 126: "6 months (~126 days)", 252: "1 year (~252 days)"}[d])
    n_paths = st.slider("How many simulated futures", 200, 5000, 1500, step=100, help="More paths = smoother cones but slower.")
    seed = st.number_input("Random seed (optional)", value=42, step=1)

if not ticker:
    st.info("Enter a stock symbol to begin (example: AAPL).")
    st.stop()

with st.spinner("Downloading recent prices…"):
    df_tkr, src_tkr = fetch_prices(ticker, lookback_years)
    spy_close = get_spy_series(lookback_years)

if df_tkr.empty or "close" not in df_tkr.columns or df_tkr["close"].dropna().empty:
    st.error("We couldn't load data for that symbol. Try another (e.g., AAPL, MSFT, SPY).")
    st.stop()

close = df_tkr["close"].dropna()
S0 = float(close.iloc[-1])

# Estimate simple daily drift and vol from the stock
cfg = EstimationConfig()
mu, sig = estimate_drift_vol(close, cfg)
ann_vol = sig * math.sqrt(252.0)
p_down = chance_of_down_day(mu, sig)

# Hidden: determine market regime from SPY (defaults to Neutral if unavailable)
regime = determine_regime(spy_close)

# Simulate with realistic mechanics (same simple UI)
with st.spinner("Projecting forward…"):
    mean_path, pct = simulate_path_realistic(
        S0=S0, mu=mu, sigma0=sig, days=int(horizon), n_paths=int(n_paths), seed=int(seed),
        regime=regime,  # hidden adjustment
        mean_revert_kappa=0.002,  # gentle pull
        shock_prob=0.03,          # mild macro shock chance
        shock_sigma_mult=2.0,     # ~2σ shock size
    )

# ==============================
# Top cards (plain English, unchanged style)
# ==============================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", _fmt_usd(S0))
c2.metric("Typical daily swing", _fmt_pct(ann_vol / math.sqrt(252.0)))
c3.metric("Chance of a down day", _fmt_pct(p_down))
c4.metric("Data source", src_tkr or "Unknown")

st.divider()

# ==============================
# Chart (same look)
# ==============================
st.subheader(f"Projected price for {ticker}")

plot_df = pd.DataFrame({
    "Day": np.arange(0, horizon + 1),
    "Mean": mean_path[:horizon + 1],
    "P10": pct["p10"][:horizon + 1],
    "P20": pct["p20"][:horizon + 1],
    "P50": pct["p50"][:horizon + 1],
    "P80": pct["p80"][:horizon + 1],
    "P90": pct["p90"][:horizon + 1],
})

if PLOTLY_OK:
    fig = go.Figure()
    # 80% cone
    fig.add_trace(go.Scatter(x=plot_df["Day"], y=plot_df["P90"], name="Upper (90th)", mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=plot_df["Day"], y=plot_df["P10"], name="Lower (10th)", mode="lines",
                             fill="tonexty", line=dict(width=0), hoverinfo="skip", showlegend=False,
                             fillcolor="rgba(150,150,250,0.2)"))
    # Median line
    fig.add_trace(go.Scatter(x=plot_df["Day"], y=plot_df["P50"], name="Median projection", mode="lines"))
    # Today marker
    fig.add_trace(go.Scatter(x=[0], y=[S0], mode="markers+text", text=["today"], textposition="top center",
                             name="Today"))

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Days ahead",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(plot_df["Day"], plot_df["P10"], plot_df["P90"], alpha=0.2, label="80% cone")
    ax.plot(plot_df["Day"], plot_df["P50"], label="Median projection")
    ax.scatter([0], [S0], label="Today")
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ==============================
# Key numbers & plain-English summary (same UX)
# ==============================
proj_1m = plot_df.loc[plot_df["Day"] == 21, "P50"].iloc[0] if (plot_df["Day"] == 21).any() else np.nan
proj_3m = plot_df.loc[plot_df["Day"] == 63, "P50"].iloc[0] if (plot_df["Day"] == 63).any() else np.nan
proj_6m = plot_df.loc[plot_df["Day"] == 126, "P50"].iloc[0] if (plot_df["Day"] == 126).any() else np.nan
proj_1y = plot_df.loc[plot_df["Day"] == 252, "P50"].iloc[0] if (plot_df["Day"] == 252).any() else np.nan

st.markdown("### Key checkpoints")
st.dataframe(pd.DataFrame([
    {"Horizon": "1 month (~21d)", "Projected price (median)": _fmt_usd(proj_1m)},
    {"Horizon": "3 months (~63d)", "Projected price (median)": _fmt_usd(proj_3m)},
    {"Horizon": "6 months (~126d)", "Projected price (median)": _fmt_usd(proj_6m)},
    {"Horizon": "1 year (~252d)", "Projected price (median)": _fmt_usd(proj_1y)},
]), use_container_width=True, hide_index=True)

# Verdict based on 1y median projection
ret_1y = (proj_1y / S0 - 1.0) if np.isfinite(proj_1y) and S0 > 0 else np.nan
if np.isfinite(ret_1y):
    verdict = "Favorable" if ret_1y > 0.05 else ("High risk" if ret_1y < -0.05 else "Mixed")
else:
    verdict = "Mixed"

st.markdown("### What this means")
st.write(
    f"- Today: **{_fmt_usd(S0)}**\n"
    f"- Typical daily move: about **{_fmt_pct(sig)}** (this varies over time)\n"
    f"- Chance of a down day tomorrow: **{_fmt_pct(p_down)}**\n"
    f"- In plain English: Based on recent behavior and market context, the stock’s middle path reaches "
    f"**{_fmt_usd(proj_1y)}** in about a year "
    f"(that’s {(_fmt_pct(ret_1y) if np.isfinite(ret_1y) else 'n/a')} from today). "
    f"Our simple verdict right now: **{verdict}**."
)

st.caption("Educational use only — not investment advice.")
