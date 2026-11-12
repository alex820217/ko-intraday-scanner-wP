# Stable identifiers/search hints for Scalable Capital.
# ISIN is preferred because Scalable supports searching by underlying ISIN.
BASISWERT_IDS = {
    "NVDA": {"isin": "US67066G1040", "sc_name": "NVIDIA"},
    "MSFT": {"isin": "US5949181045", "sc_name": "Microsoft"},
    "AAPL": {"isin": "US0378331005", "sc_name": "Apple"},
    "AMZN": {"isin": "US0231351067", "sc_name": "Amazon"},
    "META": {"isin": "US30303M1027", "sc_name": "Meta Platforms"},
    "GOOGL": {"isin": "US02079K3059", "sc_name": "Alphabet"},
    "TSLA": {"isin": "US88160R1014", "sc_name": "Tesla"},
    "AMD": {"isin": "US0079031078", "sc_name": "AMD"},
    "AIR.DE": {"isin": "NL0000235190", "sc_name": "Airbus"},
    "RHM.DE": {"isin": "DE0007030009", "sc_name": "Rheinmetall"},
    "VNA.DE": {"isin": "DE000A1ML7J1", "sc_name": "Vonovia"},
    "ALV.DE": {"isin": "DE0008404005", "sc_name": "Allianz"},
    "DTE.DE": {"isin": "DE0005557508", "sc_name": "Deutsche Telekom"},
    "SAP.DE": {"isin": "DE0007164600", "sc_name": "SAP"},
    "BAS.DE": {"isin": "DE000BASF111", "sc_name": "BASF"},
    "IFX.DE": {"isin": "DE0006231004", "sc_name": "Infineon"},
    "DBK.DE": {"isin": "DE0005140008", "sc_name": "Deutsche Bank"},
    "CBK.DE": {"isin": "DE000CBK1001", "sc_name": "Commerzbank"},
    "ADS.DE": {"isin": "DE000A1EWWW0", "sc_name": "adidas"},
    "DHL.DE": {"isin": "DE0005552004", "sc_name": "DHL Group"},
    "^GDAXI": {"isin": "DE0008469008", "sc_name": "DAX"},
    "^STOXX50E": {"isin": "EU0009658145", "sc_name": "EURO STOXX 50"},
    "^GSPC": {"isin": "US78378X1072", "sc_name": "S&P 500"},
    "^NDX": {"isin": "", "sc_name": "Nasdaq 100"},
    "GC=F": {"isin": "", "sc_name": "Gold"},
    "SI=F": {"isin": "", "sc_name": "Silber"},
    "PL=F": {"isin": "", "sc_name": "Platin"},
    "HG=F": {"isin": "", "sc_name": "Kupfer"},
    "CL=F": {"isin": "", "sc_name": "WTI Öl"},
    "BZ=F": {"isin": "", "sc_name": "Brent Öl"},
    "NG=F": {"isin": "", "sc_name": "Natural Gas"},
    "HO=F": {"isin": "", "sc_name": "Heizöl"},
    "EURUSD=X": {"isin": "", "sc_name": "EUR/USD"},
    "GBPUSD=X": {"isin": "", "sc_name": "GBP/USD"},
    "JPY=X": {"isin": "", "sc_name": "USD/JPY"},
}


def basiswert_meta(ticker, asset, direction_hint=None):
    meta = BASISWERT_IDS.get(ticker, {})
    name = meta.get("sc_name") or asset
    isin = meta.get("isin") or ""
    side = direction_hint if direction_hint in {"LONG", "SHORT"} else ""
    search = " ".join(x for x in [name, side, "Knockout"] if x)
    return isin, search
