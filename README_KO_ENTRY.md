# KO Entry Analyzer installieren

Lade diese Dateien in den GitHub-Root:

- `ko_trade_profile.py`
- `integrate_ko_profile.py`

Führe einmal aus:

```bash
python integrate_ko_profile.py
```

Danach `integrate_ko_profile.py` wieder löschen oder im Repository belassen. Die Streamlit-Hauptdatei bleibt:

```text
intraday_ko_scanner_v4_1.py
```

Der Analyzer erscheint in den Setup-Details vor der Chartanalyse. Ohne konkrete WKN/Produktdaten wird niemals final grün freigegeben, weil Hebel, KO-Abstand und Spread fehlen.
