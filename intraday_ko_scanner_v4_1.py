import time as time_module
from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
from decision_logic import SignalInput, decide_signal
from market_universe import MARKET_GROUPS, build_universe
from position_manager import load_positions, save_position, delete_position, close_position, load_history, evaluate_position
from product_resolver import resolve_wkn, quote_age_minutes, freshness_label

# Position mapping is deliberately independent from the current live scan.
# Existing positions must remain selectable even if their cash market is closed.
ASSET_ALIASES = {
    "NVDA": ("nvidia", "nvidia corp", "nvidia corporation", "nvidia nasdaq"),
    "MSFT": ("microsoft", "microsoft corp", "microsoft corporation"),
    "AAPL": ("apple", "apple inc"),
    "AMZN": ("amazon", "amazon.com"),
    "META": ("meta", "meta platforms"),
    "GOOGL": ("alphabet", "google"),
    "TSLA": ("tesla", "tesla inc"),
    "AMD": ("amd", "advanced micro devices"),
    "AVGO": ("broadcom",),
    "NFLX": ("netflix",),
    "PLTR": ("palantir",),
    "MU": ("micron",),
    "QCOM": ("qualcomm",),
    "INTC": ("intel",),
    "ADBE": ("adobe",),
    "CRM": ("salesforce",),
    "^GDAXI": ("dax", "dax 40", "deutscher aktienindex"),
    "^NDX": ("nasdaq 100", "nasdaq100", "ndx"),
    "^GSPC": ("s&p 500", "sp 500", "s&p500"),
    "GC=F": ("gold",),
    "CL=F": ("wti", "wti oil", "crude oil"),
    "EURUSD=X": ("eur/usd", "eurusd", "euro us dollar"),
}

def normalize_underlying(value):
    import re
    return " ".join(re.sub(r"[^a-z0-9&+/ ]+", " ", (value or "").lower()).split())

def build_position_asset_options():
    options = {}
    for _group_name, group_assets in MARKET_GROUPS.items():
        for asset_name, ticker_value in group_assets.items():
            options[f"{asset_name} ({ticker_value})"] = ticker_value
    return dict(sorted(options.items(), key=lambda item: item[0].casefold()))

def auto_map_underlying(underlying, options):
    needle = normalize_underlying(underlying)
    ticker_to_label = {ticker: label for label, ticker in options.items()}
    if not needle:
        return None, None, 0.0, "Basiswertname fehlt"
    for ticker, aliases in ASSET_ALIASES.items():
        if ticker not in ticker_to_label:
            continue
        for alias in aliases:
            alias_norm = normalize_underlying(alias)
            if alias_norm in needle or needle in alias_norm:
                return ticker_to_label[ticker], ticker, 1.0, f"Alias: {alias}"
    for label, ticker in options.items():
        asset_name = normalize_underlying(label.split(" (")[0])
        if asset_name and (asset_name in needle or needle in asset_name):
            return label, ticker, 0.95, "Namensübereinstimmung"
    return None, None, 0.0, "keine eindeutige Zuordnung"

st.set_page_config(page_title="KO Market Radar V3", page_icon="📡", layout="wide")
st.title("📡 KO Market Radar V4.1.1")
st.caption("Market Radar plus vereinfachte WKN-Positionserfassung mit Geld/Brief, Quelle und Zeitstempel")

MIN_15M_BARS = 210
MIN_5M_BARS = 60


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


def scan_asset(meta, spread_ok, event_risk, require_open):
    d15 = add_indicators(download_one(meta["ticker"], "15m", "30d")).dropna()
    d5 = add_indicators(download_one(meta["ticker"], "5m", "5d")).dropna()
    if len(d15) < MIN_15M_BARS or len(d5) < MIN_5M_BARS:
        raise ValueError(f"zu wenige Daten (15m={len(d15)}, 5m={len(d5)})")

    ls, ss, lc, sc = score_15m(d15)
    preferred = "LONG" if ls >= ss else "SHORT"
    t5, timing_count = timing_5m(d5, preferred)
    # Entry Timing: bis zu 10 zusätzliche Punkte, READY benötigt mindestens 3/4 Timing-Bedingungen.
    adjusted_long = min(100, ls + (timing_count * 2 if preferred == "LONG" else 0))
    adjusted_short = min(100, ss + (timing_count * 2 if preferred == "SHORT" else 0))

    r, p = d15.iloc[-1], d15.iloc[-2]
    price, atr = float(r.Close), float(r.ATR)
    stop = price - 1.2*atr if preferred == "LONG" else price + 1.2*atr
    target1 = price + 1.8*abs(price-stop) if preferred == "LONG" else price - 1.8*abs(price-stop)
    rr = abs(target1-price)/abs(price-stop)
    is_open = market_open(meta["ticker"])
    decision = decide_signal(SignalInput(
        price=price, prev_close=float(p.Close), ema20=float(r.EMA20), ema50=float(r.EMA50),
        ema200=float(r.EMA200), prev_ema20=float(p.EMA20), rsi=float(r.RSI),
        macd_hist=float(r.MACD_HIST), adx=float(r.ADX), vwap=float(r.VWAP), atr=atr,
        long_score=adjusted_long, short_score=adjusted_short, rr=rr,
        market_open=(is_open or not require_open), data_fresh=True, confirmed_bar=True,
        spread_ok=spread_ok, event_risk=event_risk,
        volatility_ok=0.05 <= atr/price*100 <= 5.0,
    ))

    # READY wird auf WATCH zurückgestuft, falls das 5m Timing noch nicht bestätigt ist.
    if decision["signal"] in {"🟢 READY LONG", "🔴 READY SHORT"} and timing_count < 3:
        decision["signal"] = "🟡 WATCH " + decision["direction"]
        decision["quality"] = "WATCH"
        decision["reason"] = "15m Setup gültig, 5m Entry-Timing noch nicht bestätigt"

    direction = decision["direction"]
    if direction == "LONG":
        stop, t1, t2, ko = price-1.2*atr, price+2.16*atr, price+3.36*atr, price-1.7*atr
    elif direction == "SHORT":
        stop, t1, t2, ko = price+1.2*atr, price-2.16*atr, price-3.36*atr, price+1.7*atr
    else:
        stop=t1=t2=ko=np.nan

    checks = lc if direction == "LONG" else sc if direction == "SHORT" else (lc if ls >= ss else sc)
    checks = {**checks, **t5}
    return {
        "Gruppe":meta["group"], "Asset":meta["name"], "Ticker":meta["ticker"],
        "Signal":decision["signal"], "Qualität":decision["quality"], "Score":decision["score"],
        "Long":adjusted_long, "Short":adjusted_short, "5m Timing":f"{timing_count}/4",
        "Kurs":price, "RSI":float(r.RSI), "ADX":float(r.ADX), "ATR %":atr/price*100,
        "Markt":"🟢 Offen" if is_open else "🔴 Geschlossen", "Grund":decision["reason"],
        "Entry":price if direction in {"LONG","SHORT"} else np.nan, "Stop":stop,
        "Ziel 1":t1, "Ziel 2":t2, "CRV":rr if direction in {"LONG","SHORT"} else np.nan,
        "KO Referenz":ko, "KO Abstand %":abs(price-ko)/price*100 if np.isfinite(ko) else np.nan,
        "Checks":checks, "Timestamp 15m":d15.index[-1], "Timestamp 5m":d5.index[-1],
    }, d15, d5


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
rows, data15, data5, errors = [], {}, {}, []
start = time_module.time()
for n, meta in enumerate(items, start=1):
    status.info(f"Prüfe {n}/{len(items)}: {meta['name']} ({meta['ticker']})")
    try:
        row, d15, d5 = scan_asset(meta, spread_ok, event_risk, require_open)
        rows.append(row); data15[meta["ticker"]]=d15; data5[meta["ticker"]]=d5
    except Exception as exc:
        errors.append({"Asset":meta["name"], "Ticker":meta["ticker"], "Fehler":str(exc)})
    progress.progress(n/len(items), text=f"{n}/{len(items)} Assets geprüft")
status.empty(); progress.empty()

if not rows:
    st.error("Keine Assets konnten ausgewertet werden.")
    if errors: st.dataframe(pd.DataFrame(errors), hide_index=True)
    st.stop()

scan = pd.DataFrame(rows)
priority = {"🟢 READY LONG":0,"🔴 READY SHORT":0,"🟡 WATCH LONG":1,"🟡 WATCH SHORT":1,"⚪ WAIT":2,"⛔ NO TRADE":3}
scan["Prio"] = scan.Signal.map(priority).fillna(9)
scan = scan.sort_values(["Prio","Score","ADX"], ascending=[True,False,False])

m = st.columns(6)
m[0].metric("Scan aktiv",len(items)); m[1].metric("Analysiert",len(rows)); m[2].metric("Übersprungen",len(skipped_closed))
m[3].metric("READY",int(scan.Signal.str.contains("READY").sum()))
m[4].metric("WATCH",int(scan.Signal.str.contains("WATCH").sum()))
m[5].metric("Laufzeit",f"{time_module.time()-start:.1f}s")

st.header("🔥 Top Market Radar")
show = scan.head(top_n).copy()
st.dataframe(show[["Gruppe","Asset","Ticker","Signal","Qualität","Score","Long","Short","5m Timing","RSI","ADX","ATR %","CRV","Markt","Grund"]]
    .style.format({"RSI":"{:.1f}","ADX":"{:.1f}","ATR %":"{:.2f}","CRV":"{:.2f}"}),
    use_container_width=True, hide_index=True)

with st.expander("Alle analysierten Assets"):
    st.dataframe(scan.drop(columns=["Prio","Checks"]), use_container_width=True, hide_index=True)
with st.expander(f"Geschlossene Märkte übersprungen ({len(skipped_closed)})"):
    if skipped_closed:
        st.dataframe(pd.DataFrame(skipped_closed), use_container_width=True, hide_index=True)
    else:
        st.success("Keine Cash-Märkte übersprungen.")
with st.expander(f"Nicht auswertbare Assets ({len(errors)})"):

    if errors: st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else: st.success("Keine Datenfehler.")

st.header("🔍 Setup-Details")
selected = st.selectbox("Asset", scan.Asset.tolist())
r = scan[scan.Asset == selected].iloc[0]
c=st.columns(6)
c[0].metric("Signal",r.Signal); c[1].metric("Qualität",r.Qualität); c[2].metric("Score",int(r.Score))
c[3].metric("5m Timing",r["5m Timing"]); c[4].metric("Long",int(r.Long)); c[5].metric("Short",int(r.Short))
st.info(r.Grund)
st.dataframe(pd.DataFrame([{"Kriterium":k,"Status":"✅" if v else "❌"} for k,v in r.Checks.items()]), hide_index=True, use_container_width=True)
if np.isfinite(r.Entry):
    levels=st.columns(6)
    for col,label in zip(levels,["Entry","Stop","Ziel 1","Ziel 2","CRV","KO Abstand %"]):
        col.metric(label,f"{r[label]:.4f}" if label not in {"CRV","KO Abstand %"} else f"{r[label]:.2f}")

ticker=r.Ticker
left,right=st.columns(2)
with left:
    st.subheader("5-Minuten Entry-Timing")
    st.line_chart(data5[ticker][["Close","EMA9","EMA20","VWAP"]].tail(160))
with right:
    st.subheader("15-Minuten Trendbestätigung")
    st.line_chart(data15[ticker][["Close","EMA20","EMA50","EMA200","VWAP"]].tail(160))

st.caption(f"Letzte 5m-Kerze: {r['Timestamp 5m']} | Letzte 15m-Kerze: {r['Timestamp 15m']}")
st.warning("V3 scannt ein breites, aber bewusst liquides Universum und nicht jede weltweit notierte Aktie. READY ist nur ein Kandidat. KO-Level, Spread, Emittent, Bezugsverhältnis, Handelszeit und Event-Risiko vor einer manuellen Order prüfen.")


# ================================================================
# Offene KO-Positionen
# ================================================================
st.divider()
st.header("💼 Offene KO-Positionen")
st.caption("Die WKN ist der Positionsschlüssel. Produktdaten werden nicht automatisch aus Scalable Capital gelesen; KO-Level, Ratio und Kaufdaten werden einmalig manuell übernommen.")

asset_options = build_position_asset_options()

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

        # Basiswert automatisch gegen das vollständige Marktuniversum auflösen.
        automatic_asset, automatic_ticker, mapping_confidence, mapping_reason = auto_map_underlying(
            product.get("underlying"), asset_options
        )
        if automatic_asset:
            st.success(
                f"✅ Basiswert automatisch erkannt: {automatic_asset} "
                f"({mapping_confidence:.0%}; {mapping_reason})"
            )
        else:
            st.warning(
                f"⚠️ Basiswert nicht automatisch erkannt: "
                f"{product.get('underlying') or 'keine Angabe'}"
            )
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
            scan_match = scan.loc[scan.Ticker == selected_ticker, "Kurs"]
            if not scan_match.empty:
                default_underlying_entry = float(scan_match.iloc[0])
            else:
                try:
                    fallback = download_one(selected_ticker, "15m", "5d")
                    default_underlying_entry = float(fallback["Close"].iloc[-1]) if not fallback.empty else 1.0
                except Exception:
                    default_underlying_entry = 1.0
            underlying_entry = st.number_input(
                "Basiswertkurs bei Kauf", min_value=0.000001,
                value=default_underlying_entry, format="%.6f"
            )
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
