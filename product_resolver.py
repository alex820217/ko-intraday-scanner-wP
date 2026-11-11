from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.onvista.de"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


@dataclass
class ProductData:
    wkn: str
    isin: Optional[str] = None
    product_name: Optional[str] = None
    issuer: Optional[str] = None
    underlying: Optional[str] = None
    direction: Optional[str] = None
    ko_level: Optional[float] = None
    ratio: Optional[float] = None
    leverage: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_volume: Optional[int] = None
    ask_volume: Optional[int] = None
    spread_abs: Optional[float] = None
    spread_pct: Optional[float] = None
    currency: Optional[str] = None
    quote_timestamp: Optional[str] = None
    fetched_at: Optional[str] = None
    source_name: str = "Onvista"
    source_url: Optional[str] = None
    quote_verified: bool = False
    quote_warning: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _number(value: str | None) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9,.-]", "", value)
    if not cleaned:
        return None
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _integer(value: str | None) -> Optional[int]:
    parsed = _number(value)
    return int(parsed) if parsed is not None else None


def _first(pattern: str, text: str, flags=re.I | re.S) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _find_product_url(wkn: str, session: requests.Session) -> Optional[str]:
    response = session.get(f"{BASE}/suche/?searchValue={quote(wkn)}", timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        label = link.get_text(" ", strip=True)
        if wkn.upper() in (href + " " + label).upper() and "/derivate/" in href:
            return urljoin(BASE, href)
    return None


def _parse_quotes(text: str) -> Optional[dict]:
    # Example: Geld • 8.000 Stk. 9,140 EUR / Brief • 8.000 Stk. 9,150 EUR
    # The first number is volume, the second number is the actual quote.
    bid_match = re.search(
        r"Geld\s*[•·]?\s*([0-9.]+)\s*Stk\.?\s*([0-9.,]+)\s*(EUR|USD|GBP|GBp|CHF)",
        text, re.I,
    )
    ask_match = re.search(
        r"Brief\s*[•·]?\s*([0-9.]+)\s*Stk\.?\s*([0-9.,]+)\s*(EUR|USD|GBP|GBp|CHF)",
        text, re.I,
    )
    if not (bid_match and ask_match):
        return None
    return {
        "bid_volume": _integer(bid_match.group(1)),
        "bid": _number(bid_match.group(2)),
        "ask_volume": _integer(ask_match.group(1)),
        "ask": _number(ask_match.group(2)),
        "currency": bid_match.group(3),
    }


def _validate_quote(data: Optional[dict]) -> tuple[bool, Optional[str]]:
    if not data:
        return False, "Geld-/Briefblock nicht eindeutig gefunden"
    bid, ask = data.get("bid"), data.get("ask")
    if bid is None or ask is None:
        return False, "Geld oder Brief fehlt"
    if bid <= 0 or ask <= 0:
        return False, "Kurs muss positiv sein"
    if ask < bid:
        return False, "Brief liegt unter Geld"
    spread_pct = (ask - bid) / bid * 100
    if spread_pct > 20:
        return False, f"Spread unplausibel hoch ({spread_pct:.2f} %)"
    return True, None


def resolve_wkn(wkn: str) -> dict:
    wkn = re.sub(r"\s+", "", wkn).upper()
    result = ProductData(
        wkn=wkn,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not re.fullmatch(r"[A-Z0-9]{6}", wkn):
        result.error = "WKN muss genau 6 Buchstaben/Ziffern enthalten."
        return result.to_dict()

    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        product_url = _find_product_url(wkn, session)
        if not product_url:
            result.error = "Keine öffentlich erreichbare Produktseite für diese WKN gefunden."
            return result.to_dict()
        response = session.get(product_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.stripped_strings)
        result.source_url = product_url
        result.product_name = soup.title.get_text(" ", strip=True).split("•")[0].strip() if soup.title else None
        result.isin = _first(r"ISIN\s+([A-Z]{2}[A-Z0-9]{10})", text)
        result.issuer = _first(r"Emittent\s+(.+?)(?=\s+(?:Basiswert|Geld|Brief|WKN|ISIN|Hebel))", text)
        result.underlying = _first(r"Basiswert\s+(.+?)(?=\s+(?:Basispreis|K\.O\.|K\.O|Hebel|Bezugsverhältnis))", text)
        direction_blob = (result.product_name or "") + " " + text[:1200]
        result.direction = (
            "LONG" if re.search(r"\b(?:LONG|CALL)\b", direction_blob, re.I)
            else "SHORT" if re.search(r"\b(?:SHORT|PUT)\b", direction_blob, re.I)
            else None
        )
        result.ko_level = _number(_first(r"(?:K\.O\.-?Schwelle|K\.O\.|K\.O)\s+([0-9.,]+)", text))
        result.ratio = _number(_first(r"Bezugsverhältnis\s+([0-9.,]+)", text))
        result.leverage = _number(_first(r"Hebel\s+([0-9.,]+)\s*x?", text))

        quote_data = _parse_quotes(text)
        verified, warning = _validate_quote(quote_data)
        result.quote_verified = verified
        result.quote_warning = warning
        if quote_data:
            result.bid_volume = quote_data["bid_volume"]
            result.ask_volume = quote_data["ask_volume"]
            result.currency = quote_data["currency"]
            if verified:
                result.bid = quote_data["bid"]
                result.ask = quote_data["ask"]
                result.spread_abs = result.ask - result.bid
                result.spread_pct = result.spread_abs / result.bid * 100

        result.quote_timestamp = (
            _first(r"(\d{2}\.\d{2}\.\d{4},\s*\d{2}:\d{2}:\d{2})", text)
            or _first(r"(\d{2}\.\d{2}\.\d{2},\s*\d{2}:\d{2}:\d{2})", text)
        )
        if not all([result.underlying, result.direction, result.ko_level is not None]):
            result.error = "Kerndaten nicht vollständig oder eindeutig. Bitte Basiswert/Richtung/KO-Level prüfen."
    except requests.RequestException as exc:
        result.error = f"Produktquelle nicht erreichbar: {exc.__class__.__name__}"
    except Exception as exc:
        result.error = f"Produktdaten nicht verarbeitbar: {exc.__class__.__name__}"
    return result.to_dict()


def quote_age_minutes(quote_timestamp: Optional[str], fetched_at: Optional[str] = None) -> Optional[float]:
    if not quote_timestamp:
        return None
    now = datetime.fromisoformat(fetched_at) if fetched_at else datetime.now(timezone.utc)
    local_tz = now.astimezone().tzinfo
    for fmt in ("%d.%m.%Y, %H:%M:%S", "%d.%m.%y, %H:%M:%S"):
        try:
            quote_dt = datetime.strptime(quote_timestamp, fmt).replace(tzinfo=local_tz)
            return max(0.0, (now.astimezone() - quote_dt).total_seconds() / 60)
        except ValueError:
            continue
    return None


def freshness_label(age_minutes: Optional[float]) -> str:
    if age_minutes is None:
        return "⚫ KEIN ZEITSTEMPEL"
    if age_minutes <= 5:
        return "🟢 AKTUELL"
    if age_minutes <= 15:
        return "🟡 VERZÖGERT"
    return "🔴 VERALTET"
