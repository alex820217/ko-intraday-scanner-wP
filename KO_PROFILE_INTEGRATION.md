# Integration in intraday_ko_scanner_v4_1.py

1. Import ergänzen:

```python
from ko_trade_profile import render_profile_sidebar, render_entry_panel, trailing_stop_pct
```

2. Nach den übrigen Sidebar-Einstellungen:

```python
ko_profile = render_profile_sidebar()
```

3. In der Detailansicht, direkt nach den Trade-Leveln:

```python
# Vor der WKN-Auswahl sind Produktdaten noch unbekannt. Der Analyzer blockiert
# dann die finale Freigabe, zeigt aber bereits Einsatz/Risiko/Ziel an.
selected_product = st.session_state.get("resolved_product")
ko_entry = render_entry_panel(r, selected_product, ko_profile)
```

4. Bei einer offenen Position kann der Trailing-Stop so angezeigt werden:

```python
active_trailing_pct = trailing_stop_pct(result["pl_pct"] or 0, ko_profile)
st.metric("Trailing Stop", f"{active_trailing_pct:+.0f} % ab Kaufpreis")
```
