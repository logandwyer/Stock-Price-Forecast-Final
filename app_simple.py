# app_simple.py
from __future__ import annotations

import math
import sys
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

try:
    import yfinance as yf
except Exception as e:
    st.error("This app needs the 'yfinance' package.\n\nOn your machine, run:\n\npip install -r requirements.txt")
    raise e

st.set_page_config(page_title="Friendly Stock Projector", page_icon="📈", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
def _fmt_usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"${x:,.2f}"

def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.2%}"

# -------------------------------
# Data
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(symbol: str, years: int) -> pd.DataFrame:
    """
    Download daily prices for 1 symbol.
    We query a little extra history to ensure moving averages work.
    """
    symbol = symbol.strip().upper().replace("/", "-").replace(".", "-")
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years + 1)

    # Try download; if it fails, try Ticker().history
    for attempt in range(2):
        try:
            df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), progress=False, auto_adjust=True, interval="1d", threads=False)
            if df is not None and not df.empty:
                df = df.rename(columns={"Close": "close"})[["close"]].dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df
        except Exception:
            pass
        time.sleep(0.4 * (attempt + 1))

    try:
        hist = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            hist = hist.rename(columns={"Close": "close"})[["close"]].dropna()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.loc[start:]
            return hist
    except Exception:
        pass

    return pd.DataFrame()

# -------------------------------
# Simple stats & simulation
# -------------------------------
@dataclass
class EstimationConfig:
    lookback_days_for_stats: int = 90  # recent window to estimate "typical" daily move

def estimate_drift_vol(close: pd.Series, cfg: EstimationConfig) -> Tuple[float, float]:
    """
    Estimate daily drift (mu) and daily volatility (sigma) from recent returns.
    """
    lr = np.log(close).diff().dropna()
    if len(lr) < 20:
        # Not enough data, fallback to whole series
        win = lr
    else:
        win = lr.tail(cfg.lookback_days_for_stats)
    mu = float(win.mean())
    sig = float(win.std(ddof=1))
    # Safety clamps
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sig) or sig <= 0:
        sig = float(lr.std(ddof=1)) if len(lr) > 2 else 0.02
        if not np.isfinite(sig) or sig <= 0:
            sig = 0.02
    return mu, sig

def chance_of_down_day(mu: float, sig: float) -> float:
    """
    Under a normal assumption, P(return < 0) = Phi((-mu)/sig).
    If sigma is ~0, treat as ~50%.
    """
    if sig <= 1e-12:
        return 0.5
    z = (-mu) / sig
    # Standard normal CDF via error function
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def simulate_mean_path(S0: float, mu: float, sigma: float, days: int, n_paths: int, seed: Optional[int]) -> Tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    Z = rng.standard_normal(size=(n_paths, days))
    r = mu + sigma * Z
    prices = np.empty((n_paths, days + 1), dtype=float)
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(np.cumsum(r, axis=1))
    # Percentile bands for cones
    pct = {
        "p10": np.percentile(prices, 10, axis=0),
        "p20": np.percentile(prices, 20, axis=0),
        "p50": np.percentile(prices, 50, axis=0),  # median path
        "p80": np.percentile(prices, 80, axis=0),
        "p90": np.percentile(prices, 90, axis=0),
    }
    mean_path = prices.mean(axis=0)
    return mean_path, pct

# -------------------------------
# UI
# -------------------------------
st.title("📈 Friendly Stock Projector")
st.caption("A plain‑English stock projection—no jargon required.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Stock symbol", value="AAPL").strip().upper()
    lookback_years = st.slider("How much history to use", 1, 10, 5, help="The app learns from recent price moves in this window.")
    horizon = st.select_slider("How far ahead to project", options=[21, 63, 126, 252], value=126,
                               format_func=lambda d: {21: "1 month (~21 days)", 63: "3 months (~63 days)", 126: "6 months (~126 days)", 252: "1 year (~252 days)"}[d])
    n_paths = st.slider("How many simulated futures", 200, 5000, 1500, step=100, help="More paths = smoother cones but slower.")
    seed = st.number_input("Random seed (optional)", value=42, step=1)
    st.caption("Data source: Yahoo Finance (via yfinance)")

if not ticker:
    st.info("Enter a stock symbol to begin (example: AAPL).")
    st.stop()

with st.spinner("Downloading recent prices…"):
    df = fetch_prices(ticker, lookback_years)

if df.empty:
    st.error("We couldn't load data for that symbol. Try a different one (like AAPL or MSFT).")
    st.stop()

close = df["close"].dropna()
S0 = float(close.iloc[-1])

# Compute simple stats
cfg = EstimationConfig()
mu, sig = estimate_drift_vol(close, cfg)
p_down = chance_of_down_day(mu, sig)
ann_vol = sig * math.sqrt(252.0)

# Simulate future mean path + cones
with st.spinner("Projecting forward…"):
    mean_path, pct = simulate_mean_path(S0=S0, mu=mu, sigma=sig, days=int(horizon), n_paths=int(n_paths), seed=int(seed))

# -------------------------------
# Top cards (plain English)
# -------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", _fmt_usd(S0))
c2.metric("Typical daily swing", _fmt_pct(ann_vol / math.sqrt(252.0)))
c3.metric("Chance of a down day", _fmt_pct(p_down))
c4.metric("Data used", f"{len(close):,} days")

st.divider()

# -------------------------------
# Chart
# -------------------------------
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

# -------------------------------
# Key numbers & plain-English summary
# -------------------------------
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
    f"- In plain English: Based on recent behavior, the stock’s middle‑of‑the‑road path reaches "
    f"**{_fmt_usd(proj_1y)}** in about a year (that’s {(_fmt_pct(ret_1y) if np.isfinite(ret_1y) else 'n/a')} from today). "
    f"Our simple verdict right now: **{verdict}**."
)

st.caption("Educational use only — not investment advice.")
