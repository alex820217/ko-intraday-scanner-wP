from __future__ import annotations

import re
from dataclasses import dataclass, asdict
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
    spread_abs: Optional[float] = None
    spread_pct: Optional[float] = None
    currency: Optional[str] = None
    quote_timestamp: Optional[str] = None
    fetched_at: Optional[str] = None
    source_name: str = "Onvista"
    source_url: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _number(value: str | None) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9,.-]", "", value).strip()
    if not cleaned:
        return None
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first(patterns, text, flags=re.I | re.S):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return None


def _find_product_url(wkn: str, session: requests.Session) -> Optional[str]:
    search_url = f"{BASE}/suche/?searchValue={quote(wkn)}"
    response = session.get(search_url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        label = link.get_text(" ", strip=True)
        if wkn.upper() in (href + " " + label).upper() and "/derivate/" in href:
            return urljoin(BASE, href)
    # Fallback: direkte URL aus eingebetteten Daten/HTML suchen.
    match = re.search(r'("|\')(?P<url>/derivate/[^"\']*' + re.escape(wkn) + r'[^"\']*)("|\')', response.text, re.I)
    return urljoin(BASE, match.group("url")) if match else None


def resolve_wkn(wkn: str) -> dict:
    wkn = re.sub(r"\s+", "", wkn).upper()
    result = ProductData(wkn=wkn, fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if not re.fullmatch(r"[A-Z0-9]{6}", wkn):
        result.error = "WKN muss aus genau 6 Buchstaben/Ziffern bestehen."
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
        result.product_name = (soup.title.get_text(" ", strip=True).split("•")[0].strip() if soup.title else None)
        result.isin = _first([r"ISIN\s+([A-Z]{2}[A-Z0-9]{10})", r"\b([A-Z]{2}[A-Z0-9]{10})\b"], text)
        result.issuer = _first([r"Emittent\s+(.+?)(?=\s+(?:Basiswert|Geld|Brief|WKN|ISIN|Hebel))"], text)
        result.underlying = _first([r"Basiswert\s+(.+?)(?=\s+(?:Basispreis|K\.O\.|K\.O|Hebel|Bezugsverhältnis|Geld|Brief))"], text)
        direction_text = (result.product_name or "") + " " + text[:800]
        result.direction = "LONG" if re.search(r"\b(?:LONG|CALL)\b", direction_text, re.I) else "SHORT" if re.search(r"\b(?:SHORT|PUT)\b", direction_text, re.I) else None
        result.ko_level = _number(_first([r"K\.O\.-?Schwelle\s+([0-9.,]+)", r"K\.O\.\s+([0-9.,]+)", r"KO-Barriere\s+([0-9.,]+)"], text))
        result.ratio = _number(_first([r"Bezugsverhältnis\s+([0-9.,]+)"], text))
        result.leverage = _number(_first([r"Hebel\s+([0-9.,]+)\s*x?"], text))
        result.bid = _number(_first([r"Geld(?:kurs)?\s*(?:•[^0-9]*)?([0-9.,]+)\s*(?:EUR|USD|GBP|GBp)?"], text))
        result.ask = _number(_first([r"Brief(?:kurs)?\s*(?:•[^0-9]*)?([0-9.,]+)\s*(?:EUR|USD|GBP|GBp)?"], text))
        result.currency = _first([r"(?:Geld|Brief).*?\b(EUR|USD|GBP|GBp|CHF)\b"], text)
        result.quote_timestamp = _first([
            r"(\d{2}\.\d{2}\.\d{4},\s*\d{2}:\d{2}:\d{2})",
            r"(\d{2}\.\d{2}\.\d{2},\s*\d{2}:\d{2}:\d{2})",
        ], text)
        if result.bid is not None and result.ask is not None:
            result.spread_abs = result.ask - result.bid
            result.spread_pct = (result.spread_abs / result.bid * 100) if result.bid > 0 else None
        required = [result.underlying, result.direction, result.ko_level]
        if not all(v is not None for v in required):
            result.error = "Produkt gefunden, aber Basiswert/Richtung/KO-Level nicht vollständig oder eindeutig lesbar. Bitte fehlende Angaben prüfen."
    except requests.RequestException as exc:
        result.error = f"Produktquelle nicht erreichbar: {exc.__class__.__name__}"
    except Exception as exc:
        result.error = f"Produktdaten konnten nicht verarbeitet werden: {exc.__class__.__name__}"
    return result.to_dict()


def quote_age_minutes(quote_timestamp: Optional[str], fetched_at: Optional[str]) -> Optional[float]:
    if not quote_timestamp:
        return None
    for fmt in ("%d.%m.%Y, %H:%M:%S", "%d.%m.%y, %H:%M:%S"):
        try:
            quote_dt = datetime.strptime(quote_timestamp, fmt).astimezone()
            now = datetime.fromisoformat(fetched_at) if fetched_at else datetime.now(timezone.utc)
            if quote_dt.tzinfo is None:
                quote_dt = quote_dt.replace(tzinfo=now.astimezone().tzinfo)
            return max(0.0, (now.astimezone() - quote_dt.astimezone()).total_seconds() / 60)
        except ValueError:
            pass
    return None


def freshness_label(age: Optional[float]) -> str:
    if age is None:
        return "⚫ KEIN ZEITSTEMPEL"
    if age <= 5:
        return "🟢 AKTUELL"
    if age <= 15:
        return "🟡 VERZÖGERT"
    return "🔴 VERALTET"
