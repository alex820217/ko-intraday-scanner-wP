"""Asset aliases and matching for KO product underlyings.

Used by the position manager independently of current market-session filtering.
"""
import re
from difflib import SequenceMatcher

# Labels/tickers aligned with market_universe.py.
ALIASES = {
    "NVDA": ["nvidia", "nvidia corp", "nvidia corporation", "nvidia corp registered shares", "nvd"],
    "MSFT": ["microsoft", "microsoft corp", "microsoft corporation"],
    "AAPL": ["apple", "apple inc"],
    "AMZN": ["amazon", "amazon.com", "amazon com"],
    "META": ["meta", "meta platforms"],
    "GOOGL": ["alphabet", "google", "alphabet inc"],
    "TSLA": ["tesla", "tesla inc"],
    "AMD": ["amd", "advanced micro devices"],
    "AVGO": ["broadcom", "broadcom inc"],
    "NFLX": ["netflix", "netflix inc"],
    "PLTR": ["palantir", "palantir technologies"],
    "MU": ["micron", "micron technology"],
    "QCOM": ["qualcomm"],
    "INTC": ["intel", "intel corp"],
    "ADBE": ["adobe"],
    "CRM": ["salesforce"],
    "^GDAXI": ["dax", "dax 40", "deutscher aktienindex"],
    "^NDX": ["nasdaq 100", "nasdaq100", "ndx"],
    "^GSPC": ["s&p 500", "sp 500", "s&p500"],
    "GC=F": ["gold"],
    "CL=F": ["wti", "wti oil", "crude oil"],
    "EURUSD=X": ["eur/usd", "eurusd", "euro us dollar"],
}


def _norm(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9&+/ ]+", " ", text)
    return " ".join(text.split())


def resolve_underlying_to_ticker(underlying, asset_options):
    """Return (label, ticker, confidence, reason) or (None, None, 0, reason)."""
    needle = _norm(underlying)
    if not needle:
        return None, None, 0.0, "Basiswertname fehlt"

    ticker_to_label = {ticker: label for label, ticker in asset_options.items()}

    # First use explicit aliases for common derivative underlyings.
    for ticker, aliases in ALIASES.items():
        if ticker not in ticker_to_label:
            continue
        for alias in aliases:
            a = _norm(alias)
            if a and (a in needle or needle in a):
                return ticker_to_label[ticker], ticker, 1.0, f"Alias: {alias}"

    # Then compare the complete configured asset names.
    best = (None, None, 0.0, "keine eindeutige Zuordnung")
    for label, ticker in asset_options.items():
        asset_name = _norm(label.split(" (")[0])
        if asset_name and (asset_name in needle or needle in asset_name):
            return label, ticker, 0.95, "Namensübereinstimmung"
        score = SequenceMatcher(None, needle, asset_name).ratio()
        if score > best[2]:
            best = (label, ticker, score, "Ähnlichkeitsvergleich")
    return best if best[2] >= 0.72 else (None, None, best[2], "keine eindeutige Zuordnung")
