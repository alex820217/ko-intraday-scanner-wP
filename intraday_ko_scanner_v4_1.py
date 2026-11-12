import time as time_module
from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import ta
import yfinance as yf
from decision_logic import SignalInput, decide_signal
from market_universe import MARKET_GROUPS, build_universe
from basiswert_ids import basiswert_meta
from position_manager import load_positions, save_position, delete_position, close_position, load_history, evaluate_position
from product_resolver import resolve_wkn, quote_age_minutes, freshness_label

st.set_page_config(page_title="KO Market Radar V4.1 Early", page_icon="📡", layout="wide")
st.markdown("""<style>
.block-container{max-width:1500px;padding-top:2.35rem;padding-bottom:3rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #e6eaf0;padding:14px 16px;border-radius:16px;box-shadow:0 5px 18px rgba(15,23,42,.05)}
div[data-testid="stDataFrame"],[data-testid="stExpander"]{border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;box-shadow:0 4px 16px rgba(15,23,42,.04)}
h1,h2,h3{letter-spacing:-.025em}.small-label{color:#64748b;font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em}.signal-box{padding:14px 16px;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0;margin:.4rem 0 1rem}.sc-box{padding:14px 16px;border-radius:14px;background:#eef6ff;border:1px solid #cfe4ff;margin:.5rem 0 1rem}
</style>""",unsafe_allow_html=True)
st.markdown('<h1 style="margin:0 0 .25rem 0;font-size:2.45rem;line-height:1.15">📡 KO Market Radar V4.1 Early</h1>', unsafe_allow_html=True)
st.caption("60m Regime + 15m Bias + 5m Setup + 3m Trigger, zweistufig und konfliktgefiltert")


MIN_15M_BARS = 210
MIN_5M_BARS = 60
MIN_3M_BARS = 80


def series(frame, column):
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


@st.cache_data(ttl=300, show_spinner=False)
def download_one(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period, auto_adjust=True,
                       prepost=False, progress=False, threads=False)
    if data.empty:
        return data
    out = pd.DataFrame(index=data.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = series(data, col)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=180, show_spinner=False)
def download_three_minute(ticker):
    """Yahoo has no native 3m interval. Build confirmed 3m candles from 1m data."""
    raw = download_one(ticker, "1m", "7d")
    if raw.empty:
        return raw
    agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    bars = raw.resample("3min", origin="start_day", label="right", closed="right").agg(agg)
    return bars.dropna(subset=["Open", "High", "Low", "Close"])


def market_open(ticker):
    if ticker.endswith(".DE") or ticker in {"^GDAXI", "^STOXX50E"}:
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        return now.weekday() < 5 and time(9, 0) <= now.time() < time(17, 30)
    if ticker.startswith("^") or ticker.isalpha():
        now = datetime.now(ZoneInfo("America/New_York"))
        return now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)
    return True


def add_indicators(df):
    d = df.copy()
    c, h, l, v = d.Close, d.High, d.Low, d.Volume.fillna(0)
    for n in [9, 20, 50, 200]:
        d[f"EMA{n}"] = ta.trend.EMAIndicator(c, window=n).ema_indicator()
    d["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    d["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    d["ADX"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    d["MACD_HIST"] = ta.trend.MACD(c).macd_diff()
    typical = (h + l + c) / 3
    dates = pd.Series(d.index.date, index=d.index)
    cumulative_volume = v.groupby(dates).cumsum()
    d["VWAP"] = ((typical * v).groupby(dates).cumsum() / cumulative_volume.replace(0, np.nan)).fillna(d.EMA20)
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)
    return d


def score_15m(d):
    r, p = d.iloc[-1], d.iloc[-2]
    price, atr = float(r.Close), float(r.ATR)
    vol_present = d.Volume.tail(30).sum() > 0
    vol_ok = (not vol_present) or pd.isna(r.VOL_RATIO) or r.VOL_RATIO >= 0.8
    lc = {
        "15m Trend": price > r.EMA20 > r.EMA50 > r.EMA200,
        "Momentum": 52 <= r.RSI <= 68 and r.MACD_HIST > 0,
        "ADX": r.ADX >= 20, "VWAP": price > r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close > p.EMA20 and price > r.EMA20,
        "Nicht überdehnt": (price-r.EMA20) <= 1.25*atr,
    }
    sc = {
        "15m Trend": price < r.EMA20 < r.EMA50 < r.EMA200,
        "Momentum": 32 <= r.RSI <= 48 and r.MACD_HIST < 0,
        "ADX": r.ADX >= 20, "VWAP": price < r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close < p.EMA20 and price < r.EMA20,
        "Nicht überdehnt": (r.EMA20-price) <= 1.25*atr,
    }
    weights = {"15m Trend":25,"Momentum":15,"ADX":15,"VWAP":10,"Volumen":10,"2-Bar":15,"Nicht überdehnt":10}
    return sum(weights[k] for k,v in lc.items() if v), sum(weights[k] for k,v in sc.items() if v), lc, sc


def timing_5m(d, direction):
    r, p = d.iloc[-1], d.iloc[-2]
    if direction == "LONG":
        checks = {
            "5m Kurs > EMA9": r.Close > r.EMA9,
            "5m EMA9 > EMA20": r.EMA9 > r.EMA20,
            "5m MACD positiv": r.MACD_HIST > 0,
            "5m Trigger bestätigt": r.Close > p.High,
        }
    else:
        checks = {
            "5m Kurs < EMA9": r.Close < r.EMA9,
            "5m EMA9 < EMA20": r.EMA9 < r.EMA20,
            "5m MACD negativ": r.MACD_HIST < 0,
            "5m Trigger bestätigt": r.Close < p.Low,
        }
    return checks, sum(checks.values())



def trigger_3m(d, direction):
    r, p = d.iloc[-1], d.iloc[-2]
    if direction == "LONG":
        checks = {
            "3m Kurs > EMA9": r.Close > r.EMA9,
            "3m EMA9 > EMA20": r.EMA9 > r.EMA20,
            "3m RSI > 52": r.RSI > 52,
            "3m Hoch-Ausbruch": r.Close > p.High,
        }
    else:
        checks = {
            "3m Kurs < EMA9": r.Close < r.EMA9,
            "3m EMA9 < EMA20": r.EMA9 < r.EMA20,
            "3m RSI < 48": r.RSI < 48,
            "3m Tief-Ausbruch": r.Close < p.Low,
        }
    return checks, sum(bool(v) for v in checks.values())


def setup_5m(d, direction):
    r = d.iloc[-1]
    if direction == "LONG":
        checks = {
            "5m Kurs > VWAP": r.Close > r.VWAP,
            "5m EMA9 > EMA20": r.EMA9 > r.EMA20,
            "5m RSI > 50": r.RSI > 50,
            "5m MACD positiv": r.MACD_HIST > 0,
        }
    else:
        checks = {
            "5m Kurs < VWAP": r.Close < r.VWAP,
            "5m EMA9 < EMA20": r.EMA9 < r.EMA20,
            "5m RSI < 50": r.RSI < 50,
            "5m MACD negativ": r.MACD_HIST < 0,
        }
    return checks, sum(bool(v) for v in checks.values())


def bias_15m(d):
    r = d.iloc[-1]
    long_checks = {
        "15m Kurs > EMA20": r.Close > r.EMA20,
        "15m EMA20 > EMA50": r.EMA20 > r.EMA50,
        "15m EMA50 > EMA200": r.EMA50 > r.EMA200,
        "15m ADX >= 18": r.ADX >= 18,
        "15m MACD positiv": r.MACD_HIST > 0,
    }
    short_checks = {
        "15m Kurs < EMA20": r.Close < r.EMA20,
        "15m EMA20 < EMA50": r.EMA20 < r.EMA50,
        "15m EMA50 < EMA200": r.EMA50 < r.EMA200,
        "15m ADX >= 18": r.ADX >= 18,
        "15m MACD negativ": r.MACD_HIST < 0,
    }
    long_count = sum(bool(v) for v in long_checks.values())
    short_count = sum(bool(v) for v in short_checks.values())
    direction = "LONG" if long_count >= 4 and long_count > short_count else "SHORT" if short_count >= 4 and short_count > long_count else None
    return direction, long_checks, short_checks, long_count, short_count


def regime_60m_from_15m(raw15):
    """Build a closed 60m context from 15m candles without another market-data request."""
    agg = {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    h1_raw = raw15.resample("60min", origin="start_day", label="right", closed="right").agg(agg).dropna()
    if len(h1_raw) < 60:
        return "NEUTRAL", {}, pd.DataFrame()
    h1 = add_indicators(h1_raw.iloc[:-1]).dropna()
    if len(h1) < 3:
        return "NEUTRAL", {}, h1
    r, p = h1.iloc[-1], h1.iloc[-2]
    ema20_slope = float(r.EMA20 - p.EMA20)
    long_checks = {
        "60m Kurs > EMA20": r.Close > r.EMA20,
        "60m EMA20 > EMA50": r.EMA20 > r.EMA50,
        "60m EMA20 steigt": ema20_slope > 0,
        "60m MACD nicht negativ": r.MACD_HIST >= 0,
    }
    short_checks = {
        "60m Kurs < EMA20": r.Close < r.EMA20,
        "60m EMA20 < EMA50": r.EMA20 < r.EMA50,
        "60m EMA20 fällt": ema20_slope < 0,
        "60m MACD nicht positiv": r.MACD_HIST <= 0,
    }
    lc = sum(bool(v) for v in long_checks.values())
    sc = sum(bool(v) for v in short_checks.values())
    regime = "LONG" if lc >= 3 and lc > sc else "SHORT" if sc >= 3 and sc > lc else "NEUTRAL"
    return regime, (long_checks if regime == "LONG" else short_checks if regime == "SHORT" else {**long_checks, **short_checks}), h1


def grade_setup(signal, score, adx, extension_atr, rsi_entry, context_aligned, has_3m):
    issues = []
    trade_like = "EARLY" in signal or "CONFIRMED" in signal
    if not context_aligned:
        issues.append("60m/15m Kontext nicht ausgerichtet")
    if not has_3m and "EARLY" in signal:
        issues.append("3m Trigger fehlt")
    if adx < 20:
        issues.append(f"ADX nur {adx:.1f}")
    if extension_atr > 0.75:
        issues.append(f"Entry {extension_atr:.2f} ATR von EMA20")
    if rsi_entry == "🟡 HEISS":
        issues.append("RSI bereits heiß")
    if "PRE" in signal:
        issues.append("Entry-Trigger noch nicht vollständig")
    if "WATCH" in signal:
        issues.append("Setup noch unvollständig")
    if "WAIT" in signal or "NO TRADE" in signal or "CONFLICT" in signal or "HOT ENTRY" in signal or "LATE ENTRY" in signal:
        return "-", "; ".join(issues) if issues else "Kein handelbares Setup"
    # Base grade from score, then cap based on quality defects.
    grade = "A+" if score >= 90 else "A" if score >= 85 else "B" if score >= 75 else "C" if score >= 70 else "-"
    if adx < 20 or extension_atr > 0.75 or rsi_entry == "🟡 HEISS":
        grade = "B" if grade in {"A+", "A"} else grade
    if len(issues) >= 2 and grade in {"A+", "A", "B"}:
        grade = "C"
    why = "A-Setup erfüllt" if grade in {"A+", "A"} and not issues else ("; ".join(issues) if issues else "Score unter A-Schwelle")
    return grade, why

def scan_asset(meta, spread_ok, event_risk, require_open):
    # Last row can still be forming, therefore only closed candles are evaluated.
    raw15 = download_one(meta["ticker"], "15m", "30d")
    raw5 = download_one(meta["ticker"], "5m", "5d")
    d15 = add_indicators(raw15.iloc[:-1]).dropna()
    d5 = add_indicators(raw5.iloc[:-1]).dropna()
    if len(d15) < MIN_15M_BARS or len(d5) < MIN_5M_BARS:
        raise ValueError(f"zu wenige Basisdaten (15m={len(d15)}, 5m={len(d5)})")

    old_long, old_short, old_lc, old_sc = score_15m(d15)
    bias, bias_lc, bias_sc, bias_long_count, bias_short_count = bias_15m(d15)
    preferred = bias or ("LONG" if old_long >= old_short else "SHORT")
    setup_checks, setup_count = setup_5m(d5, preferred)
    regime60, regime_checks, d60 = regime_60m_from_15m(raw15)
    context_aligned = regime60 == preferred or regime60 == "NEUTRAL"

    # Stage 2: expensive 1m download is requested only for plausible candidates.
    pre_candidate = bias is not None and setup_count >= 2 and context_aligned
    if pre_candidate:
        raw3 = download_three_minute(meta["ticker"])
        d3_candidate = add_indicators(raw3.iloc[:-1]).dropna() if not raw3.empty else pd.DataFrame()
    else:
        d3_candidate = pd.DataFrame()
    has_3m = len(d3_candidate) >= MIN_3M_BARS
    d3 = d3_candidate if has_3m else d5.copy()
    if has_3m:
        trigger_checks, trigger_count = trigger_3m(d3, preferred)
    else:
        trigger_checks = {"3m Daten verfügbar": False}
        trigger_count = 0

    r15, r5 = d15.iloc[-1], d5.iloc[-1]
    r3 = d3.iloc[-1]
    price = float(r3.Close) if has_3m else float(r5.Close)
    atr = float(r5.ATR)
    atr_pct = atr / price * 100 if price else 0
    extension_atr = abs(float(r5.Close) - float(r5.EMA20)) / float(r5.ATR) if float(r5.ATR) > 0 else np.inf
    entry_quality = "🟢 OPTIMAL" if extension_atr <= 0.75 else "🟡 AKZEPTABEL" if extension_atr <= 1.0 else "🔴 SPÄT"
    late_entry = extension_atr > 1.0
    rsi15 = float(r15.RSI)
    rsi_hot = (preferred == "LONG" and rsi15 > 72) or (preferred == "SHORT" and rsi15 < 28)
    rsi_warm = (preferred == "LONG" and 68 < rsi15 <= 72) or (preferred == "SHORT" and 28 <= rsi15 < 32)
    rsi_entry_status = "🔴 ÜBERHITZT" if rsi_hot else "🟡 HEISS" if rsi_warm else "🟢 NORMAL"
    volatility_ok = 0.05 <= atr_pct <= 5.0
    is_open = market_open(meta["ticker"])
    no_trade_reasons = []
    if require_open and not is_open: no_trade_reasons.append("Markt geschlossen")
    if not spread_ok: no_trade_reasons.append("KO-Spread nicht akzeptabel")
    if event_risk: no_trade_reasons.append("Event-Risiko aktiv")
    if not volatility_ok: no_trade_reasons.append("Volatilität ungeeignet")

    bias_points = (bias_long_count if preferred == "LONG" else bias_short_count) * 8
    setup_points = setup_count * 7
    trigger_points = trigger_count * 7
    score = min(100, bias_points + setup_points + trigger_points)
    confirmed_old = (old_long >= 80 and all(old_lc.values())) if preferred == "LONG" else (old_short >= 80 and all(old_sc.values()))

    if no_trade_reasons:
        signal, quality, reason, direction = "⛔ NO TRADE", "BLOCKED", "; ".join(no_trade_reasons), None
    elif bias and regime60 != "NEUTRAL" and regime60 != preferred:
        signal, quality, reason, direction = "⚠️ CONTEXT CONFLICT", "NO ENTRY", f"15m {preferred}, aber 60m-Regime {regime60}", None
    elif rsi_hot and bias and setup_count >= 3:
        signal, quality, reason, direction = "⛔ HOT ENTRY", "WAIT PULLBACK", f"RSI {rsi15:.1f} ist für einen neuen {preferred}-Einstieg überhitzt", preferred
    elif late_entry and bias and setup_count >= 3:
        signal, quality, reason, direction = "⛔ LATE ENTRY", "WAIT PULLBACK", f"Bewegung bereits {extension_atr:.2f} ATR von EMA20 entfernt", preferred
    elif bias and confirmed_old and setup_count >= 3 and ((has_3m and trigger_count >= 3) or not has_3m) and score >= 80:
        signal = "🟢🟢 CONFIRMED LONG" if preferred == "LONG" else "🔴🔴 CONFIRMED SHORT"
        quality = "A" if score < 90 else "A+"
        reason = "15m und 5m streng bestätigt; 3m-Fallback aktiv" if not has_3m else "15m, 5m und 3m vollständig bestätigt"
        direction = preferred
    elif has_3m and bias and setup_count >= 3 and trigger_count >= 3 and score >= 70:
        signal = "🟢 EARLY LONG" if preferred == "LONG" else "🔴 EARLY SHORT"
        quality, reason, direction = "EARLY", "15m Bias, 5m Setup und 3m Trigger bestätigt", preferred
    elif bias and setup_count >= 3:
        signal = "🟠 PRE-LONG" if preferred == "LONG" else "🟠 PRE-SHORT"
        quality = "PRE"
        reason = f"3m Trigger erst {trigger_count}/4" if has_3m else "3m nicht verfügbar; 5m/15m-Fallback aktiv"
        direction = preferred
    elif bias:
        signal = "🟡 WATCH LONG" if preferred == "LONG" else "🟡 WATCH SHORT"
        quality, reason, direction = "WATCH", f"5m Setup erst {setup_count}/4", preferred
    else:
        signal, quality, reason, direction = "⚪ WAIT", "NEUTRAL", "Kein eindeutiger 15m Bias", None

    if direction == "LONG":
        stop, t1, t2, ko = price - 1.2*atr, price + 2.16*atr, price + 3.36*atr, price - 1.7*atr
    elif direction == "SHORT":
        stop, t1, t2, ko = price + 1.2*atr, price - 2.16*atr, price - 3.36*atr, price + 1.7*atr
    else:
        stop=t1=t2=ko=np.nan
    checks = {**regime_checks, **(bias_lc if preferred == "LONG" else bias_sc), **setup_checks, **trigger_checks,
              "Nicht zu spät": not late_entry, "RSI nicht überhitzt": not rsi_hot,
              "3m Daten verfügbar": has_3m, "Volatilität geeignet": volatility_ok}
    setup_grade, why_not_a = grade_setup(signal, score, float(r15.ADX), extension_atr, rsi_entry_status, context_aligned, has_3m)
    underlying_isin, sc_search = basiswert_meta(meta["ticker"], meta["name"], direction)
    return {
        "Gruppe":meta["group"], "Asset":meta["name"], "Ticker":meta["ticker"],
        "Signal":signal, "Qualität":quality, "Score":score, "Setup Grade":setup_grade,
        "Setup-Hinweis":why_not_a, "Basiswert ISIN":underlying_isin, "SC Suche":sc_search,
        "Long":old_long, "Short":old_short, "60m Regime":regime60, "15m Bias":bias or "NEUTRAL",
        "5m Setup":f"{setup_count}/4", "3m Trigger":f"{trigger_count}/4" if has_3m else "FALLBACK",
        "Entry Status":entry_quality, "RSI Entry":rsi_entry_status, "EMA20 Dist. ATR":extension_atr,
        "Kurs":price, "RSI":float(r15.RSI), "ADX":float(r15.ADX), "ATR %":atr_pct,
        "Markt":"🟢 Offen" if is_open else "🔴 Geschlossen", "Grund":reason,
        "Entry":price if direction in {"LONG","SHORT"} else np.nan, "Stop":stop,
        "Ziel 1":t1, "Ziel 2":t2, "CRV":1.8 if direction in {"LONG","SHORT"} else np.nan,
        "KO Referenz":ko, "KO Abstand %":abs(price-ko)/price*100 if np.isfinite(ko) else np.nan,
        "Checks":checks, "Timestamp 15m":d15.index[-1], "Timestamp 5m":d5.index[-1], "Timestamp 3m":d3.index[-1],
    }, d15, d5, d3



def trading_chart(df, timeframe_label, ema_cols, bars, levels=None, height=330):
    """Readable intraday OHLC chart with candles, overlays and optional trade levels."""
    d = df.tail(bars).copy().reset_index()
    time_col = d.columns[0]
    d = d.rename(columns={time_col:"Time"})
    # Candle body must have a visible height even for very small bodies.
    d["BodyLow"] = d[["Open","Close"]].min(axis=1)
    d["BodyHigh"] = d[["Open","Close"]].max(axis=1)
    d["Direction"] = np.where(d["Close"] >= d["Open"], "Up", "Down")
    price_values = pd.concat([d["Low"], d["High"]]).dropna()
    if price_values.empty:
        st.info(f"Keine {timeframe_label}-Chartdaten verfügbar.")
        return
    pad = max((price_values.max()-price_values.min()) * 0.08, abs(price_values.iloc[-1]) * 0.001)
    domain = [float(price_values.min()-pad), float(price_values.max()+pad)]
    x = alt.X("Time:T", axis=alt.Axis(title=None, format="%H:%M", labelAngle=0, labelOverlap=True))
    y = alt.Y("Low:Q", scale=alt.Scale(domain=domain, zero=False), axis=alt.Axis(title=None, format="~f"))
    wicks = alt.Chart(d).mark_rule(strokeWidth=1).encode(x=x, y=y, y2="High:Q", color=alt.Color("Direction:N", scale=alt.Scale(domain=["Up","Down"], range=["#16a34a","#dc2626"]), legend=None))
    bodies = alt.Chart(d).mark_bar(size=5).encode(x=x, y=alt.Y("BodyLow:Q", scale=alt.Scale(domain=domain, zero=False)), y2="BodyHigh:Q", color=alt.Color("Direction:N", scale=alt.Scale(domain=["Up","Down"], range=["#22c55e","#ef4444"]), legend=None), tooltip=[alt.Tooltip("Time:T"), "Open:Q","High:Q","Low:Q","Close:Q"])
    layers = [wicks, bodies]
    colors = {"EMA9":"#f59e0b", "EMA20":"#2563eb", "EMA50":"#8b5cf6", "EMA200":"#dc2626", "VWAP":"#0891b2"}
    for col in ema_cols:
        if col in d.columns:
            layers.append(alt.Chart(d).mark_line(strokeWidth=1.6, color=colors.get(col,"#64748b")).encode(x=x, y=alt.Y(f"{col}:Q", scale=alt.Scale(domain=domain, zero=False)), tooltip=[alt.Tooltip(f"{col}:Q")]))
    if levels:
        level_colors = {"Entry":"#111827", "Stop":"#dc2626", "TP1":"#16a34a", "TP2":"#059669"}
        for name, value in levels.items():
            if value is not None and np.isfinite(value):
                ld = pd.DataFrame({"Level":[float(value)], "Name":[name]})
                layers.append(alt.Chart(ld).mark_rule(strokeDash=[6,4], strokeWidth=1.3, color=level_colors.get(name,"#111827")).encode(y=alt.Y("Level:Q", scale=alt.Scale(domain=domain, zero=False)), tooltip=["Name:N","Level:Q"]))
    chart = alt.layer(*layers).properties(height=height).interactive(bind_y=False)
    st.altair_chart(chart, use_container_width=True)

st.sidebar.header("Marktuniversum")
default_groups = ["Deutschland", "US Tech", "Indizes", "Metalle", "Energie", "Forex"]
selected_groups = st.sidebar.multiselect("Marktgruppen", list(MARKET_GROUPS), default=default_groups)
max_assets = st.sidebar.slider("Maximale Assets pro Scan", 10, 60, 54, 2)
require_open = st.sidebar.checkbox("Geschlossene Märkte als NO TRADE", True)
spread_ok = st.sidebar.checkbox("KO-Spread manuell akzeptabel", True)
event_risk = st.sidebar.checkbox("Event-Risiko aktiv", False)
top_n = st.sidebar.slider("Top-Ergebnisse anzeigen", 5, 30, 15)

universe = build_universe(selected_groups)
all_items = list(universe.values())[:max_assets]
# Vorfilter V3.1: geschlossene Cash-Märkte werden VOR jedem yfinance-Aufruf entfernt.
# Futures/FX bleiben aktiv, da deren Handelszeiten komplexer als eine einzelne Cash-Session sind.
if require_open:
    items = [meta for meta in all_items if market_open(meta["ticker"])]
    skipped_closed = [meta for meta in all_items if not market_open(meta["ticker"])]
else:
    items = all_items
    skipped_closed = []
if not items:
    st.warning("Mindestens eine Marktgruppe auswählen."); st.stop()

if st.sidebar.button("Cache leeren und neu scannen"):
    st.cache_data.clear(); st.rerun()

progress = st.progress(0, text="Scan wird vorbereitet …")
status = st.empty()
rows, data15, data5, data3, errors = [], {}, {}, {}, []
start = time_module.time()
for n, meta in enumerate(items, start=1):
    status.info(f"Prüfe {n}/{len(items)}: {meta['name']} ({meta['ticker']})")
    try:
        row, d15, d5, d3 = scan_asset(meta, spread_ok, event_risk, require_open)
        rows.append(row); data15[meta["ticker"]]=d15; data5[meta["ticker"]]=d5; data3[meta["ticker"]]=d3
    except Exception as exc:
        errors.append({"Asset":meta["name"], "Ticker":meta["ticker"], "Fehler":str(exc)})
    progress.progress(n/len(items), text=f"{n}/{len(items)} Assets geprüft")
status.empty(); progress.empty()

if not rows:
    st.error("Keine Assets konnten ausgewertet werden.")
    if errors: st.dataframe(pd.DataFrame(errors), hide_index=True)
    st.stop()

scan = pd.DataFrame(rows)
priority = {"🟢 EARLY LONG":0,"🔴 EARLY SHORT":0,"🟢🟢 CONFIRMED LONG":0,"🔴🔴 CONFIRMED SHORT":0,"🟠 PRE-LONG":1,"🟠 PRE-SHORT":1,"🟡 WATCH LONG":2,"🟡 WATCH SHORT":2,"⚠️ CONTEXT CONFLICT":3,"⚪ WAIT":3,"⛔ HOT ENTRY":4,"⛔ LATE ENTRY":4,"⛔ NO TRADE":5}
scan["Prio"] = scan.Signal.map(priority).fillna(9)
grade_rank = {"A+":0, "A":1, "B":2, "C":3, "-":9}
scan["GradeRank"] = scan["Setup Grade"].map(grade_rank).fillna(9)
scan = scan.sort_values(["Prio","GradeRank","Score","ADX"], ascending=[True,True,False,False])

trade_count = int(scan.Signal.str.contains("EARLY|CONFIRMED", regex=True).sum())
watch_count = int(scan.Signal.str.contains("PRE|WATCH", regex=True).sum())
# Compact status strip: operational counters stay visible but no longer dominate the trading view.
st.caption(
    f"**{trade_count} Trade** · {watch_count} PRE/WATCH · "
    f"{len(rows)}/{len(items)} analysiert · {len(skipped_closed)} übersprungen · "
    f"{time_module.time()-start:.1f}s"
)

st.header("🔥 Top Market Radar")
show=scan.head(top_n).copy()
compact_cols=["Asset","Signal","Setup Grade","Score","60m Regime","15m Bias","5m Setup","3m Trigger","Entry Status","Setup-Hinweis","SC Suche"]
def _grade_bg(v): return {"A+":"background-color:#dcfce7;color:#087443;font-weight:750","A":"background-color:#e8f5e9;color:#16733a;font-weight:750","B":"background-color:#fef3c7;color:#8a6300;font-weight:750","C":"background-color:#ffedd5;color:#a64b00;font-weight:750","-":"color:#94a3b8"}.get(v,"")
def _entry_bg(v):
    v=str(v)
    if "OPTIMAL" in v:return "background-color:#dcfce7;color:#137443;font-weight:650"
    if "AKZEPTABEL" in v:return "background-color:#fef3c7;color:#8a6300;font-weight:650"
    if "SPÄT" in v:return "background-color:#fee2e2;color:#b42318;font-weight:650"
    return ""
compact=show[compact_cols].copy();compact.index=range(1,len(compact)+1);compact.index.name="#"
st.dataframe(compact.style.map(_grade_bg,subset=["Setup Grade"]).map(_entry_bg,subset=["Entry Status"]),use_container_width=True,height=min(720,42+36*len(compact)))
st.caption("Ranking: Trade-Status → Grade → Score → ADX. RSI, ATR, CRV, ISIN und Kursdaten findest du in der Detailansicht.")

with st.expander("📊 Alle technischen Daten anzeigen"):
    st.dataframe(scan.drop(columns=["Prio","GradeRank","Checks"]), use_container_width=True, hide_index=True)
with st.expander(f"Geschlossene Märkte übersprungen ({len(skipped_closed)})"):
    if skipped_closed:
        st.dataframe(pd.DataFrame(skipped_closed), use_container_width=True, hide_index=True)
    else:
        st.success("Keine Cash-Märkte übersprungen.")
with st.expander(f"Nicht auswertbare Assets ({len(errors)})"):

    if errors: st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else: st.success("Keine Datenfehler.")

st.header("🎯 Setup-Details")
selected = st.selectbox("Asset", scan.Asset.tolist())
r = scan[scan.Asset == selected].iloc[0]
c=st.columns(8)
c[0].metric("Signal",r.Signal); c[1].metric("Setup Grade",r["Setup Grade"]); c[2].metric("Score",int(r.Score))
c[3].metric("60m Regime",r["60m Regime"]); c[4].metric("15m Bias",r["15m Bias"]); c[5].metric("5m Setup",r["5m Setup"]); c[6].metric("3m Trigger",r["3m Trigger"]); c[7].metric("Entry",r["Entry Status"])
st.markdown(f"""<div class="signal-box"><span class="small-label">Setup-Hinweis</span><br><b>{r['Setup-Hinweis']}</b><br><span style="color:#64748b">{r.Grund}</span></div>""",unsafe_allow_html=True)
st.markdown(f"""<div class="sc-box"><span class="small-label">Scalable Capital Suche</span><br><span style="font-size:1.15rem;font-weight:750">{r['SC Suche']}</span><br><span style="color:#64748b">Basiswert-ISIN: {r['Basiswert ISIN'] or '–'} · Ticker: {r.Ticker}</span></div>""",unsafe_allow_html=True)
with st.expander("Technische Kriterien anzeigen",expanded=False):
    st.dataframe(pd.DataFrame([{"Kriterium":k,"Status":"✅" if v else "❌"} for k,v in r.Checks.items()]), hide_index=True, use_container_width=True)
if np.isfinite(r.Entry):
    st.subheader("💰 Trade-Level")
    levels=st.columns(6)
    for col,label in zip(levels,["Entry","Stop","Ziel 1","Ziel 2","CRV","KO Abstand %"]):
        col.metric(label,f"{r[label]:.4f}" if label not in {"CRV","KO Abstand %"} else f"{r[label]:.2f}")

ticker=r.Ticker
st.subheader("📈 Chartanalyse")
chart_tab1,chart_tab2,chart_tab3 = st.tabs(["3m · Entry", "5m · Setup", "15m · Trend"])
trade_levels = None
if np.isfinite(r.Entry):
    trade_levels = {"Entry":r.Entry, "Stop":r.Stop, "TP1":r["Ziel 1"], "TP2":r["Ziel 2"]}
with chart_tab1:
    if r["3m Trigger"] == "FALLBACK":
        st.info("Für dieses Asset fehlen ausreichende 3m-Daten. Der Scanner verwendet den 5m-Fallback; EARLY ist damit gesperrt.")
        trading_chart(data5[ticker], "5m Fallback", ["EMA9","EMA20"], 80, trade_levels)
    else:
        trading_chart(data3[ticker], "3m", ["EMA9","EMA20"], 90, trade_levels)
with chart_tab2:
    trading_chart(data5[ticker], "5m", ["EMA9","EMA20","VWAP"], 90, trade_levels)
with chart_tab3:
    trading_chart(data15[ticker], "15m", ["EMA20","EMA50","EMA200","VWAP"], 80, trade_levels)

st.caption(f"Letzte 3m-Kerze: {r['Timestamp 3m']} | Letzte 5m-Kerze: {r['Timestamp 5m']} | Letzte 15m-Kerze: {r['Timestamp 15m']}")
st.warning("V4.1 Early nutzt 60m Regime, 15m Bias, 5m Setup und 3m Trigger. 1m-Daten werden nur für plausible Kandidaten geladen. RSI über 72 im LONG bzw. unter 28 im SHORT blockiert neue Einstiege. 3m-Fallback erlaubt kein EARLY-Signal. KO-Level, Spread, Emittent, Bezugsverhältnis, Handelszeit und Event-Risiko vor einer manuellen Order prüfen.")


# ================================================================
# Offene KO-Positionen
# ================================================================
st.divider()
st.header("💼 Offene KO-Positionen")
st.caption("Die WKN ist der Positionsschlüssel. Produktdaten werden nicht automatisch aus Scalable Capital gelesen; KO-Level, Ratio und Kaufdaten werden einmalig manuell übernommen.")

# Position mapping always uses the complete configured universe, even when a market is closed.
asset_options = {}
for _group_name, _group_assets in MARKET_GROUPS.items():
    for _asset_name, _ticker_value in _group_assets.items():
        asset_options[f"{_asset_name} ({_ticker_value})"] = _ticker_value
asset_options = dict(sorted(asset_options.items(), key=lambda item: item[0].casefold()))

with st.expander("➕ Position mit WKN hinzufügen", expanded=False):
    st.write("Nur WKN, persönlicher Kaufpreis und Stückzahl eingeben. Öffentliche Produktdaten werden anschließend bestmöglich aufgelöst.")
    with st.form("simple_position_lookup"):
        c1,c2,c3 = st.columns(3)
        lookup_wkn = c1.text_input("WKN *", max_chars=6).strip().upper()
        personal_buy = c2.number_input("Kaufpreis Zertifikat € *", min_value=0.0001, format="%.4f")
        personal_qty = c3.number_input("Stückzahl *", min_value=1.0, step=1.0)
        lookup = st.form_submit_button("Produkt suchen", type="primary")
    if lookup:
        with st.spinner("Öffentliche Produktdaten werden gesucht …"):
            st.session_state["resolved_product"] = resolve_wkn(lookup_wkn)
            st.session_state["personal_buy"] = personal_buy
            st.session_state["personal_qty"] = personal_qty

    product = st.session_state.get("resolved_product")
    if product:
        age = quote_age_minutes(product.get("quote_timestamp"), product.get("fetched_at"))
        fresh = freshness_label(age)
        st.subheader(f"Produktprüfung: {product.get('wkn')}")
        if product.get("error"):
            st.warning(product["error"])
        a,b,c,d = st.columns(4)
        a.metric("Geld", "–" if product.get("bid") is None else f"{product['bid']:.4f} {product.get('currency') or ''}")
        b.metric("Brief", "–" if product.get("ask") is None else f"{product['ask']:.4f} {product.get('currency') or ''}")
        c.metric("Spread", "–" if product.get("spread_pct") is None else f"{product['spread_pct']:.2f} %")
        d.metric("Datenstatus", fresh)
        st.write(f"**Kursquelle:** {product.get('source_name') or '–'}")
        st.write(f"**Kurszeit:** {product.get('quote_timestamp') or 'kein Zeitstempel verfügbar'}")
        st.write(f"**Abrufzeit:** {product.get('fetched_at') or '–'}")
        if product.get("source_url"):
            st.markdown(f"[Produktquelle öffnen]({product['source_url']})")
        st.caption("Der Positionswert wird mit dem Geldkurs berechnet. Der Briefkurs dient als Kauf-/Spread-Referenz.")

        details = pd.DataFrame([
            {"Feld":"Produkt", "Wert":product.get("product_name")},
            {"Feld":"ISIN", "Wert":product.get("isin")},
            {"Feld":"Emittent", "Wert":product.get("issuer")},
            {"Feld":"Basiswert", "Wert":product.get("underlying")},
            {"Feld":"Richtung", "Wert":product.get("direction")},
            {"Feld":"KO-Level", "Wert":product.get("ko_level")},
            {"Feld":"Bezugsverhältnis", "Wert":product.get("ratio")},
            {"Feld":"Hebel", "Wert":product.get("leverage")},
        ])
        st.dataframe(details, hide_index=True, use_container_width=True)

        # Normalize common product names, e.g. "Nvidia Nasdaq ..." -> NVDA.
        needle = (product.get("underlying") or "").lower()
        aliases = {
            "nvidia": "NVDA", "microsoft": "MSFT", "apple": "AAPL",
            "amazon": "AMZN", "tesla": "TSLA", "amd": "AMD",
            "dax": "^GDAXI", "nasdaq 100": "^NDX", "gold": "GC=F",
            "eur/usd": "EURUSD=X", "wti": "CL=F",
        }
        target_ticker = next((ticker for alias, ticker in aliases.items() if alias in needle), None)
        automatic_asset = next((label for label, ticker in asset_options.items() if ticker == target_ticker), None)
        if not automatic_asset:
            matches = [label for label in asset_options if label.split(" (")[0].lower() in needle]
            automatic_asset = matches[0] if matches else None
        if automatic_asset:
            st.success(f"✅ Basiswert automatisch erkannt: {automatic_asset}")
        missing_core = not product.get("direction") or product.get("ko_level") is None or not automatic_asset
        if missing_core:
            st.error("Die Kerndaten konnten nicht eindeutig zugeordnet werden. Bitte unten nur die fehlenden Felder bestätigen.")
        with st.form("confirm_resolved_product"):
            selected_asset = st.selectbox("Basiswert-Zuordnung", list(asset_options), index=list(asset_options).index(automatic_asset) if automatic_asset in asset_options else 0)
            x,y,z = st.columns(3)
            direction = x.selectbox("Richtung", ["LONG","SHORT"], index=0 if product.get("direction") != "SHORT" else 1)
            ko_level = y.number_input("KO-Level", min_value=0.000001, value=float(product.get("ko_level") or 0.000001), format="%.6f")
            ratio = z.number_input("Bezugsverhältnis", min_value=0.000001, value=float(product.get("ratio") or 0.01), format="%.6f")
            selected_ticker = asset_options[selected_asset]
            _match = scan.loc[scan.Ticker == selected_ticker, "Kurs"]
            if not _match.empty:
                _default_underlying = float(_match.iloc[0])
            else:
                try:
                    _fallback = download_one(selected_ticker, "15m", "5d")
                    _default_underlying = float(_fallback["Close"].iloc[-1]) if not _fallback.empty else 1.0
                except Exception:
                    _default_underlying = 1.0
            underlying_entry = st.number_input("Basiswertkurs bei Kauf", min_value=0.000001, value=_default_underlying, format="%.6f")
            save = st.form_submit_button("Position übernehmen", type="primary")
        if save:
            save_position({
                "wkn": product["wkn"], "isin": product.get("isin") or "", "asset": selected_asset.split(" (")[0],
                "ticker": asset_options[selected_asset], "direction": direction, "ko_level": ko_level,
                "ratio": ratio, "underlying_entry": underlying_entry,
                "certificate_entry": st.session_state.get("personal_buy"),
                "certificate_current": product.get("bid") or 0,
                "quantity": st.session_state.get("personal_qty"), "initial_stop": None,
                "issuer": product.get("issuer") or "", "product_name": product.get("product_name") or "",
                "bid": product.get("bid"), "ask": product.get("ask"),
                "spread_abs": product.get("spread_abs"), "spread_pct": product.get("spread_pct"),
                "quote_source": product.get("source_name"), "quote_url": product.get("source_url"),
                "quote_timestamp": product.get("quote_timestamp"), "quote_fetched_at": product.get("fetched_at"),
                "leverage": product.get("leverage"),
            })
            st.success(f"Position {product['wkn']} gespeichert.")
            st.session_state.pop("resolved_product", None)
            st.rerun()

positions = load_positions()
if not positions:
    st.info("Noch keine offene Position gespeichert.")
else:
    for pos in positions:
        ticker_pos = pos["ticker"]
        with st.container(border=True):
            st.subheader(f"{pos['wkn']} | {pos['asset']} | {pos['direction']}")
            quote_age = quote_age_minutes(pos.get("quote_timestamp"), pos.get("quote_fetched_at"))
            q1,q2,q3,q4 = st.columns(4)
            q1.metric("Geld", "–" if pos.get("bid") is None else f"{float(pos['bid']):.4f}")
            q2.metric("Brief", "–" if pos.get("ask") is None else f"{float(pos['ask']):.4f}")
            q3.metric("Spread", "–" if pos.get("spread_pct") is None else f"{float(pos['spread_pct']):.2f} %")
            q4.metric("Produktdaten", freshness_label(quote_age))
            st.caption(f"Kursquelle: {pos.get('quote_source') or '–'} | Kurszeit: {pos.get('quote_timestamp') or '–'} | Abrufzeit: {pos.get('quote_fetched_at') or '–'}")
            if ticker_pos not in data15 or ticker_pos not in data5:
                st.warning("Basiswert wurde in diesem Scan nicht geladen. Aktiviere die passende Marktgruppe bzw. den geöffneten Markt.")
                continue
            result = evaluate_position(pos, data15[ticker_pos], data5[ticker_pos])
            st.markdown(f"### {result['action']}")
            st.write(result["reason"])
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Basiswert aktuell", f"{result['current_underlying']:.4f}")
            c2.metric("Fortschritt", f"{result['r_value']:+.2f} R")
            c3.metric("Aktiver Stop", f"{result['active_stop']:.4f}")
            c4.metric("TP1", f"{result['tp1']:.4f}")
            c5.metric("TP2", f"{result['tp2']:.4f}")
            c6.metric("KO-Abstand", f"{result['ko_distance_pct']:.2f} %")
            st.write(f"**KO-Status:** {result['ko_status']} | **Stop-KO-Puffer:** {result['stop_ko_gap']:.4f} Basiswertpunkte")
            checks = pd.DataFrame([
                {"Prüfung":"15m Trend", "Status":"✅" if result["trend_ok"] else "❌"},
                {"Prüfung":"VWAP", "Status":"✅" if result["vwap_ok"] else "❌"},
                {"Prüfung":"Momentum", "Status":"✅" if result["momentum_ok"] else "❌"},
            ])
            st.dataframe(checks, hide_index=True, use_container_width=True)
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Investment", f"{result['invested']:.2f} €")
            p2.metric("Aktueller Wert", "–" if result["current_value"] is None else f"{result['current_value']:.2f} €")
            p3.metric("P/L", "–" if result["pl_eur"] is None else f"{result['pl_eur']:+.2f} €")
            p4.metric("Performance", "–" if result["pl_pct"] is None else f"{result['pl_pct']:+.2f} %")
            with st.expander("Position bearbeiten / schließen"):
                new_price = st.number_input("Aktueller Zertifikatkurs €", min_value=0.0, value=float(pos.get("certificate_current") or 0), format="%.4f", key=f"price_{pos['wkn']}")
                x1,x2,x3 = st.columns(3)
                if x1.button("Kurs aktualisieren", key=f"upd_{pos['wkn']}"):
                    updated = dict(pos); updated["certificate_current"] = new_price
                    save_position(updated); st.rerun()
                exit_price = x2.number_input("Verkaufskurs €", min_value=0.0, format="%.4f", key=f"exit_{pos['wkn']}")
                if x2.button("Trade schließen", key=f"close_{pos['wkn']}"):
                    if exit_price <= 0: st.error("Verkaufskurs eingeben.")
                    else: close_position(pos["wkn"], exit_price); st.rerun()
                if x3.button("Position löschen", key=f"delete_{pos['wkn']}"):
                    delete_position(pos["wkn"]); st.rerun()

history = load_history()
with st.expander(f"📚 Trade-Historie ({len(history)})"):
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        st.download_button("Historie als CSV herunterladen", pd.DataFrame(history).to_csv(index=False).encode("utf-8"), "trade_history.csv", "text/csv")
    else:
        st.info("Noch keine geschlossenen Trades.")

st.warning("Positionsstatus ist technische Entscheidungshilfe. Zertifikatskurs, Spread, Emittentenpreisstellung, Finanzierung und mögliche Gaps werden nicht automatisch aus Scalable Capital übernommen. Den KO-Level niemals als Stop-Loss verwenden; Orders und Stop-Anpassungen bleiben manuell.")
