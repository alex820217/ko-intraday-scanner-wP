"""One-time patcher for intraday_ko_scanner_v4_1.py.
Run locally or in GitHub Codespaces: python integrate_ko_profile.py
"""
from pathlib import Path

path = Path("intraday_ko_scanner_v4_1.py")
if not path.exists():
    raise SystemExit("intraday_ko_scanner_v4_1.py nicht gefunden")
text = path.read_text(encoding="utf-8")

import_line = "from ko_trade_profile import render_profile_sidebar, render_entry_panel, trailing_stop_pct\n"
if import_line not in text:
    anchor = "import streamlit as st\n"
    text = text.replace(anchor, anchor + import_line, 1)

if "ko_profile = render_profile_sidebar()" not in text:
    anchor = 'st.sidebar.header("Marktuniversum")\n'
    text = text.replace(anchor, anchor + "ko_profile = render_profile_sidebar()\n", 1)

if "render_entry_panel(r, selected_product, ko_profile)" not in text:
    # Insert directly before chart analysis, after technical levels/details.
    anchors = ['st.subheader("📈 Chartanalyse")', 'ticker=r.Ticker']
    selected = next((a for a in anchors if a in text), None)
    if not selected:
        raise SystemExit("Chartanalyse-Einfügepunkt nicht gefunden")
    snippet = '''# KO product validation for the selected radar candidate.\nselected_product = st.session_state.get("resolved_product")\nrender_entry_panel(r, selected_product, ko_profile)\n\n'''
    text = text.replace(selected, snippet + selected, 1)

# Optional trailing metric in open-position area when result["pl_pct"] exists.
position_anchor = 'p4.metric("Performance", "–" if result["pl_pct"] is None else f"{result[\'pl_pct\']:+.2f} %")'
if position_anchor in text and 'active_trailing_pct = trailing_stop_pct' not in text:
    addition = position_anchor + '''\n            active_trailing_pct = trailing_stop_pct(result["pl_pct"] or 0, ko_profile)\n            st.metric("Aktiver Trailing-Stop", f"{active_trailing_pct:+.0f} % ab Kaufpreis")'''
    text = text.replace(position_anchor, addition, 1)

path.write_text(text, encoding="utf-8")
print("V4.1 wurde erfolgreich um den KO Entry Analyzer ergänzt.")
