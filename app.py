# app.py (Python 3.13–friendly bootstrap)
from __future__ import annotations
import sys, subprocess, importlib

def _ensure_yfinance():
    try:
        importlib.import_module("yfinance")
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "yfinance==0.2.43"])
        except Exception as e:
            print("Auto-install of yfinance failed:", e)

_ensure_yfinance()

# ---- rest of the app (identical to prior), but without strict REQUIRED pins ----
import os, math, time, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

PLOTLY_OK = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False

SKLEARN_OK = True
try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    SKLEARN_OK = False

import streamlit as st
import requests
from io import StringIO
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception as e:
    st.error("`yfinance` is not available. Please ensure your requirements include `yfinance==0.2.43`.")
    raise e

st.set_page_config(page_title="Macro-Informed Stock Forecaster", page_icon="📈", layout="wide")
st.title("📈 Macro-Informed Stock Forecaster")
st.caption("Regime switching • GARCH-like volatility • Macro shocks • Optional ML drift")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    lookback_years = st.slider("Lookback window (years)", 1, 10, 5)
    regime_on = st.toggle("Enable regime switching", value=True)
    garch_on = st.toggle("Enable GARCH-like conditional volatility", value=True)
    ml_on = st.toggle("Enable ML drift (21d Gradient Boosting)", value=False)
    shock_prob = st.slider("Downward shock probability (daily)", 0.0, 0.10, 0.01, 0.005)
    shock_sev = st.slider("Shock severity (multiplier on vol)", 0.0, 5.0, 1.5, 0.1)
    n_paths = st.slider("Monte Carlo paths", 500, 6000, 2000, 100)
    seed = st.number_input("Random seed (optional)", value=42, step=1)
    st.caption("Data: Yahoo Finance (yfinance).")

HORIZONS = {"1 day": 1, "5 days": 5, "1 month (~21d)": 21, "6 months (~126d)": 126, "1 year (~252d)": 252}

def logret(s): return np.log(s).diff()



@st.cache_data(show_spinner=False)
def _fetch_stooq(symbol: str) -> pd.DataFrame:
    """
    Fallback to Stooq CSV. Supports US equities/ETFs via *.us (e.g., spy.us, aapl.us) and ^vix.
    Returns df with 'Close' and DatetimeIndex, or empty DataFrame.
    """
    sym = symbol.strip().upper()
    if sym.startswith("^"):
        stq = sym.lower()  # ^VIX -> ^vix
    else:
        stq = f"{sym.lower()}.us"  # AAPL -> aapl.us, SPY -> spy.us
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
        df = df[["Close"]].rename(columns={"Close": symbol})
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_one(symbol: str, start: str, min_rows: int = 60, tries: int = 2) -> pd.DataFrame:
    """Robust single-symbol fetch with retries, Ticker().history fallback, then Stooq CSV fallback."""
    yahoo_sym = symbol.replace("/", "-").replace(".", "-").strip()

    # 1) yfinance download (few retries)
    for attempt in range(tries):
        try:
            df = yf.download(yahoo_sym, start=start, progress=False, auto_adjust=True, interval="1d", threads=False)
            if df is not None and not df.empty:
                out = df[["Close"]].rename(columns={"Close": symbol}).dropna()
                if len(out) >= min_rows or symbol in {"SPY", "^VIX", "^TNX", "^IRX"}:
                    return out
        except Exception:
            pass
        time.sleep(0.4 * (attempt + 1))

    # 2) yfinance Ticker().history fallback
    try:
        hist = yf.Ticker(yahoo_sym).history(period="max", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            out = hist[["Close"]].rename(columns={"Close": symbol}).dropna()
            out = out.loc[pd.to_datetime(start):]
            if not out.empty:
                return out
    except Exception:
        pass

    # 3) Stooq CSV fallback (works for most US equities/ETFs and ^VIX)
    stq = _fetch_stooq(symbol)
    if not stq.empty:
        stq = stq.loc[pd.to_datetime(start):]
        return stq

    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_all(ticker: str, lookback_years: int) -> pd.DataFrame:
    end = pd.Timestamp.today(tz="US/Eastern").normalize()
    start = end - pd.DateOffset(years=int(lookback_years)+1)  # pad by a year for indicators
    syms = [ticker, "SPY", "^VIX", "^TNX", "^IRX"]
    frames = {}
    failed = []

    for s in syms:
        part = fetch_one(s, start.strftime("%Y-%m-%d"))
        if not part.empty:
            frames[s] = part
        else:
            failed.append(s)
        time.sleep(0.1)  # be kind to upstream

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames.values(), axis=1).dropna(how="all")

    # Last-resort retry for the main ticker if it's missing
    if ticker not in data.columns or data[ticker].dropna().empty:
        short_start = (end - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        part = fetch_one(ticker, short_start, min_rows=20, tries=2)
        if not part.empty:
            data = data.drop(columns=[c for c in data.columns if c == ticker], errors="ignore")
            data = pd.concat([data, part], axis=1)
        else:
            failed = list(set(failed + [ticker]))

    # Surface failed symbols to the UI once (non-blocking)
    if failed:
        st.warning("Some data sources were unavailable and were skipped: " + ", ".join(sorted(set(failed))) +
                   ". The forecast will proceed with what’s available.")

    return data



@dataclass
class RegimeThresholds:
    vix_hi_q: float; vix_lo_q: float; mom_hi_q: float; mom_lo_q: float; spread_hi_q: float; spread_lo_q: float

def compute_indicators(df: pd.DataFrame):
    ind = pd.DataFrame(index=df.index)
    ind["VIX"] = df["^VIX"] if "^VIX" in df.columns else pd.Series(index=df.index, dtype=float); ind["SPY_mom63"] = df["SPY"].pct_change(63) if "SPY" in df.columns else pd.Series(index=df.index, dtype=float)
    ind["TNX"] = df["^TNX"] if "^TNX" in df.columns else pd.Series(index=df.index, dtype=float); ind["IRX"] = df["^IRX"] if "^IRX" in df.columns else pd.Series(index=df.index, dtype=float); ind["Spread"] = (ind["TNX"] - ind["IRX"]).fillna(0.0)
    thr = RegimeThresholds(
        vix_hi_q=ind["VIX"].quantile(0.66), vix_lo_q=ind["VIX"].quantile(0.33),
        mom_hi_q=ind["SPY_mom63"].quantile(0.66), mom_lo_q=ind["SPY_mom63"].quantile(0.33),
        spread_hi_q=ind["Spread"].quantile(0.66), spread_lo_q=ind["Spread"].quantile(0.33),
    )
    return ind, thr

def classify_regime(row, thr: RegimeThresholds) -> str:
    bullish = (row["VIX"] <= thr.vix_lo_q) + (row["SPY_mom63"] >= thr.mom_hi_q) + (row["Spread"] >= thr.spread_hi_q)
    bearish = (row["VIX"] >= thr.vix_hi_q) + (row["SPY_mom63"] <= thr.mom_lo_q) + (row["Spread"] <= thr.spread_lo_q)
    if bullish >= 2: return "Bull"
    if bearish >= 2: return "Bear"
    return "Neutral"

def estimate_regime_params(rets: pd.Series, regimes: pd.Series):
    params = {}
    for reg in ["Bull","Neutral","Bear"]:
        sub = rets[regimes==reg].dropna()
        if len(sub) < 30: mu, sig = rets.mean(), rets.std(ddof=1)
        else: mu, sig = sub.mean(), sub.std(ddof=1)
        params[reg] = (float(mu), float(sig))
    return params

@dataclass
class GARCHParams: omega: float; alpha: float; beta: float
def estimate_garch_like(returns: pd.Series) -> GARCHParams:
    s2 = np.nanvar(returns, ddof=1); alpha, beta = 0.06, 0.88
    omega = s2 * (1 - alpha - beta)
    if omega <= 0 or not np.isfinite(omega): omega = max(1e-8, s2 * 0.02)
    return GARCHParams(float(omega), float(alpha), float(beta))

def garch_vol_path(gp: GARCHParams, resid: np.ndarray, start_var: float) -> np.ndarray:
    T = resid.shape[0]; h = np.empty(T); h[0] = start_var
    for t in range(1, T):
        h[t] = gp.omega + gp.alpha * (resid[t-1]**2) * h[t-1] + gp.beta * h[t-1]
        if not np.isfinite(h[t]) or h[t] <= 1e-12: h[t] = h[t-1]
    return h

def fit_ml_drift(features: pd.DataFrame, target_21d: pd.Series):
    if not SKLEARN_OK: return None
    X = features.dropna(); y = target_21d.reindex(X.index).dropna(); X = X.reindex(y.index)
    if len(X) < 200: return None
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=1337, n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8)
    try: model.fit(X, y)
    except Exception: return None
    return model

def simulate_paths(S0, steps, n_paths, base_mu, base_sig, regimes, regime_params, use_regime, use_garch, gp, shock_prob, shock_sev, ml_daily_drift, seed=None):
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    if use_regime and regimes is not None and regime_params is not None:
        last_regime = regimes.dropna().iloc[-1]
        reg_names = ["Bull","Neutral","Bear"]; reg_index = {"Bull":0,"Neutral":1,"Bear":2}
        P = np.array([[0.90,0.08,0.02],[0.10,0.80,0.10],[0.03,0.12,0.85]])
        rseq = np.empty(steps, dtype=object); cur = reg_index.get(last_regime,1)
        for t in range(steps):
            cur = rng.choice([0,1,2], p=P[cur]); rseq[t] = reg_names[cur]
    else:
        rseq = np.array(["Neutral"]*steps, dtype=object)
    Z = rng.standard_normal(size=(n_paths, steps))
    if use_garch and gp is not None:
        base_var = base_sig**2; H = np.empty((n_paths, steps))
        for i in range(n_paths): H[i,:] = garch_vol_path(gp, Z[i,:], start_var=base_var)
        sig_t = np.sqrt(H)
    else:
        sig_t = np.full((n_paths, steps), base_sig, dtype=float)
    mu_t = np.full((n_paths, steps), base_mu, dtype=float)
    if use_regime and regime_params is not None:
        mu_map = {k:v[0] for k,v in regime_params.items()}
        reg_mu = np.array([mu_map.get(r, base_mu) for r in rseq])
        mu_t = np.tile(reg_mu, (n_paths,1))
    if ml_daily_drift is not None: mu_t = mu_t + ml_daily_drift
    if shock_prob > 0 and shock_sev > 0:
        shocks = rng.random(size=(n_paths, steps)) < shock_prob
        ann_vol = max(1e-8, base_sig * math.sqrt(252.0))
        shock_scale = max(1e-6, shock_sev * ann_vol)
        shock_sizes = -np.abs(rng.lognormal(mean=0.0, sigma=shock_scale, size=(n_paths, steps)))
        shock_term = shocks * shock_sizes
    else:
        shock_term = 0.0
    r = mu_t + sig_t * Z + shock_term
    prices = np.empty((n_paths, steps+1), dtype=float); prices[:,0] = S0
    prices[:,1:] = S0 * np.exp(np.cumsum(r, axis=1))
    mean_path = prices.mean(axis=0)
    return prices, mean_path

def gbm_expectations(S0: float, mu: float, sigma: float, horizons: dict) -> pd.DataFrame:
    rows = []
    for name, h in horizons.items():
        rows.append({"Horizon": name, "Days": h, "GBM Expected Price": round(S0 * math.exp(mu*h), 2),
                     "Annualized Vol (σ)": round(sigma * math.sqrt(252.0), 3)})
    return pd.DataFrame(rows)

if ticker == "": st.stop()

with st.spinner("📡 Loading market data from Yahoo Finance…"):
    df = fetch_all(ticker, lookback_years)
# Normalize index to tz-naive for safe comparisons
try:
    df.index = pd.to_datetime(df.index).tz_localize(None)
except Exception:
    pass

after_fetch_df = df

if df.empty or ticker not in df.columns or df[ticker].dropna().empty:
    st.error("Failed to load data for the requested ticker. Please try a different symbol."); st.stop()

cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)
data = df.loc[df.index >= cutoff].dropna()
if len(data) < 252: data = df.dropna()

close = data[ticker].dropna()

with st.spinner("🧭 Computing regimes and indicators…"):
    ind, thr = compute_indicators(data)
    regimes_series = ind.apply(lambda r: classify_regime(r, thr), axis=1)
    lret = logret(close).dropna()
    reg_params = estimate_regime_params(lret, regimes_series.reindex(lret.index).fillna("Neutral"))

trail = lret.tail(252 if len(lret) > 252 else len(lret))
base_mu = float(trail.mean()); base_sig = float(trail.std(ddof=1))
if not np.isfinite(base_sig) or base_sig <= 0: base_sig = float(lret.std(ddof=1))

gp = None
if garch_on:
    with st.spinner("📐 Estimating GARCH-like variance parameters…"):
        gp = estimate_garch_like(lret.fillna(0.0))

ml_model = None; ml_daily_drift = None
features = pd.DataFrame(index=data.index)
features["VIX"] = ind["VIX"]; features["SPY_mom63"] = ind["SPY_mom63"]; features["Spread"] = ind["Spread"]
features["Vol20"] = logret(close).rolling(20).std().fillna(method="bfill")
target_21d = close.pct_change(21).shift(-21)

if ml_on and SKLEARN_OK:
    with st.spinner("🤖 Fitting Gradient Boosting for 21-day drift adjustment…"):
        ml_model = fit_ml_drift(features, target_21d)
        if ml_model is not None:
            xrow = features.iloc[[-1]].dropna()
            if not xrow.empty:
                try:
                    pred21 = float(ml_model.predict(xrow)[0]); ml_daily_drift = pred21 / 21.0
                except Exception:
                    ml_daily_drift = None
else:
    if ml_on and not SKLEARN_OK: st.info("scikit-learn not available — ML drift disabled.")

S0 = float(close.iloc[-1])
gbm_df = gbm_expectations(S0, base_mu, base_sig, HORIZONS)

max_steps = max(HORIZONS.values())
with st.spinner("🧪 Running Monte Carlo simulation…"):
    _, mean_path = simulate_paths(
        S0=S0, steps=max_steps, n_paths=int(n_paths), base_mu=base_mu, base_sig=base_sig,
        regimes=regimes_series, regime_params=reg_params, use_regime=regime_on,
        use_garch=garch_on, gp=gp, shock_prob=float(shock_prob), shock_sev=float(shock_sev),
        ml_daily_drift=ml_daily_drift, seed=int(seed) if seed is not None else None
    )

forecast_points = {name: float(mean_path[d]) for name, d in HORIZONS.items()}
def valuation_summary(S0: float, m6: float, m12: float, ann_vol: float) -> str:
    r6 = (m6 / S0) - 1.0; r12 = (m12 / S0) - 1.0
    if r12 < -0.05 and ann_vol > 0.30: return "⚠️ Appears **overvalued / high risk**."
    if r12 > 0.05 and ann_vol <= 0.35: return "✅ Appears **undervalued / favorable**."
    return "ℹ️ **Mixed** signal."

ann_vol_here = base_sig * math.sqrt(252.0)
summary_text = valuation_summary(S0, forecast_points["6 months (~126d)"], forecast_points["1 year (~252d)"], ann_vol_here)

tab1, tab2 = st.tabs(["🔮 Forecast", "🧪 Diagnostics"])
with tab1:
    st.subheader(f"Forecast for {ticker}")
    c1, c2 = st.columns([1,1])
    with c1: st.write("**GBM Baseline (analytical)**"); st.dataframe(gbm_df, use_container_width=True)
    with c2:
        st.write("**Monte Carlo Forecast (mean path only)**")
        plot_df = pd.DataFrame({"Day": np.arange(0, max_steps+1), "MeanPrice": mean_path})
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["Day"], y=plot_df["MeanPrice"], mode="lines", name="Mean Path"))
            for name, d in HORIZONS.items():
                fig.add_trace(go.Scatter(x=[d], y=[mean_path[d]], mode="markers+text", text=[name], textposition="top center", name=f"{name}"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Days Ahead", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(plot_df["Day"], plot_df["MeanPrice"], label="Mean Path")
            for name, d in HORIZONS.items(): ax.scatter([d], [mean_path[d]]); ax.text(d, mean_path[d], name, ha="center", va="bottom", fontsize=8)
            ax.set_xlabel("Days Ahead"); ax.set_ylabel("Price"); ax.set_title("Mean Monte Carlo Path"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    st.divider(); st.markdown(f"**Summary:** {summary_text}")
    st.caption("This is not investment advice. Simulation is for educational use only.")

def walk_forward_validation(close: pd.Series, feats: pd.DataFrame, regimes: pd.Series,
                            base_mu: float, base_sig: float, regime_params, use_regime, use_garch, gp,
                            shock_prob: float, shock_sev: float, ml_model):
    horizon = 21; idx = close.index
    last_300 = idx[-(252+21):] if len(idx) >= 273 else idx
    n_eval = 0; sq_err = 0.0; covered = 0; rng_seed = 2025
    for t in range(len(last_300) - horizon):
        end_t = last_300[t]; fut_t = last_300[t + horizon]
        window_close = close.loc[:end_t]
        if len(window_close) < 252: continue
        S0 = float(window_close.iloc[-1])
        trail = logret(window_close).dropna().tail(252); mu_b, sig_b = trail.mean(), trail.std(ddof=1)
        mu_use = float(base_mu if np.isfinite(base_mu) else mu_b); sig_use = float(base_sig if np.isfinite(base_sig) else sig_b)
        reg_series = regimes.loc[:end_t]; reg_params = regime_params
        ml_drift = None
        if SKLEARN_OK and ml_model is not None and ml_on:
            xrow = feats.loc[end_t:end_t]
            if not xrow.empty:
                try: pred21 = float(ml_model.predict(xrow)[0]); ml_drift = pred21 / horizon
                except Exception: ml_drift = None
        _, mpath = simulate_paths(S0=S0, steps=horizon, n_paths=400, base_mu=mu_use, base_sig=sig_use,
                                  regimes=reg_series, regime_params=reg_params, use_regime=use_regime, use_garch=use_garch, gp=gp,
                                  shock_prob=shock_prob, shock_sev=shock_sev, ml_daily_drift=ml_drift, seed=rng_seed + t)
        pred_ret = math.log(mpath[-1] / mpath[0]); true_ret = math.log(float(close.loc[fut_t]) / S0)
        sq_err += (pred_ret - true_ret)**2; n_eval += 1
        prices, _ = simulate_paths(S0=S0, steps=horizon, n_paths=400, base_mu=mu_use, base_sig=sig_use,
                                   regimes=reg_series, regime_params=reg_params, use_regime=use_regime, use_garch=use_garch, gp=gp,
                                   shock_prob=shock_prob, shock_sev=shock_sev, ml_daily_drift=ml_drift, seed=rng_seed + 999 + t)
        end_prices = prices[:, -1]; lo = np.percentile(end_prices, 5); hi = np.percentile(end_prices, 95)
        covered += 1 if (float(close.loc[fut_t]) >= lo and float(close.loc[fut_t]) <= hi) else 0
    if n_eval == 0: return {"RMSE_21d_logret": float("nan"), "Coverage_5_95": float("nan")}
    return {"RMSE_21d_logret": math.sqrt(sq_err / n_eval), "Coverage_5_95": covered / n_eval}

with tab2:
    st.subheader("Walk-Forward Validation (21-day horizon)")
    with st.spinner("🔎 Evaluating recent out-of-sample performance…"):
        diag = walk_forward_validation(
            close=close, feats=features, regimes=regimes_series, base_mu=base_mu, base_sig=base_sig,
            regime_params=reg_params, use_regime=regime_on, use_garch=garch_on, gp=gp,
            shock_prob=float(shock_prob), shock_sev=float(shock_sev), ml_model=ml_model
        )
    d_df = pd.DataFrame([{
        "RMSE (21d log-return)": round(diag["RMSE_21d_logret"], 4) if np.isfinite(diag["RMSE_21d_logret"]) else np.nan,
        "Coverage (5–95%)": f"{diag['Coverage_5_95']*100:.1f}%" if np.isfinite(diag["Coverage_5_95"]) else "n/a",
    }])
    st.dataframe(d_df, use_container_width=True)
    st.caption("Diagnostics use a short rolling origin test over roughly the last year.")
