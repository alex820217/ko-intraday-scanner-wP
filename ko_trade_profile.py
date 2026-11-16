"""KO-specific entry and position management for KO Market Radar V4.1.

The module keeps the basis-value signal separate from the selected derivative.
All orders remain manual.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional
import math
import streamlit as st


@dataclass(frozen=True)
class KOProfile:
    investment_min: float = 1000.0
    investment_max: float = 2000.0
    preferred_investment: float = 1500.0
    leverage_target: float = 5.0
    leverage_min: float = 4.0
    leverage_max: float = 6.0
    ko_distance_min: float = 17.0
    ko_distance_max: float = 23.0
    standard_stop_pct: float = 5.0
    emergency_stop_pct: float = 8.0
    initial_take_profit_pct: float = 10.0
    trailing_step_pct: float = 5.0
    max_spread_pct: float = 1.5


DEFAULT_PROFILE = KOProfile()


def render_profile_sidebar() -> KOProfile:
    with st.sidebar.expander("⚙️ KO-Handelsprofil", expanded=False):
        st.caption("Persönliche Standardwerte für die KO-Entry-Prüfung")
        investment_min = st.number_input("Einsatz min. €", 100.0, 10000.0, 1000.0, 100.0)
        investment_max = st.number_input("Einsatz max. €", investment_min, 20000.0, 2000.0, 100.0)
        preferred = st.number_input("Plan-Einsatz €", investment_min, investment_max, min(1500.0, investment_max), 100.0)
        leverage_target = st.number_input("Zielhebel", 1.0, 20.0, 5.0, 0.5)
        ko_min = st.number_input("KO-Abstand min. %", 1.0, 50.0, 17.0, 1.0)
        ko_max = st.number_input("KO-Abstand max. %", ko_min, 80.0, 23.0, 1.0)
        stop = st.number_input("Standard-Stop Zertifikat %", 1.0, 20.0, 5.0, 0.5)
        max_stop = st.number_input("Absolute Verlustgrenze %", stop, 30.0, 8.0, 0.5)
        target = st.number_input("Erstes Gewinnziel %", 2.0, 50.0, 10.0, 1.0)
        trailing = st.number_input("Trailing-Schritt %-Punkte", 1.0, 20.0, 5.0, 1.0)
        max_spread = st.number_input("Max. Spread %", 0.1, 10.0, 1.5, 0.1)
    return KOProfile(
        investment_min=investment_min,
        investment_max=investment_max,
        preferred_investment=preferred,
        leverage_target=leverage_target,
        leverage_min=max(1.0, leverage_target-1.0),
        leverage_max=leverage_target+1.0,
        ko_distance_min=ko_min,
        ko_distance_max=ko_max,
        standard_stop_pct=stop,
        emergency_stop_pct=max_stop,
        initial_take_profit_pct=target,
        trailing_step_pct=trailing,
        max_spread_pct=max_spread,
    )


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def entry_analysis(row: Any, product: Optional[dict], profile: KOProfile) -> dict:
    """Combine technical setup and the selected KO product into one entry verdict."""
    signal = str(row.get("Signal", ""))
    grade = str(row.get("Setup Grade", "-"))
    entry_status = str(row.get("Entry Status", ""))
    rsi_entry = str(row.get("RSI Entry", ""))
    setup_hint = str(row.get("Setup-Hinweis", row.get("Grund", "")))
    direction = "LONG" if "LONG" in signal else "SHORT" if "SHORT" in signal else None

    leverage = _finite((product or {}).get("leverage"))
    ko_distance = _finite((product or {}).get("ko_distance_pct"))
    spread = _finite((product or {}).get("spread_pct"))
    certificate_price = _finite((product or {}).get("ask") or (product or {}).get("certificate_entry"))

    hard_blocks, cautions, positives = [], [], []
    if direction is None or not any(x in signal for x in ("EARLY", "CONFIRMED")):
        hard_blocks.append("Kein handelbares EARLY-/CONFIRMED-Signal")
    if grade not in {"A+", "A", "B"}:
        hard_blocks.append(f"Setup-Grade {grade} nicht ausreichend")
    if "SPÄT" in entry_status or "LATE" in signal:
        hard_blocks.append("Basiswert-Entry bereits zu spät")
    if "ÜBERHITZT" in rsi_entry or "HOT ENTRY" in signal:
        hard_blocks.append("Momentum für Neueinstieg überhitzt")

    if leverage is None:
        cautions.append("Hebel des konkreten Produkts fehlt")
    elif not profile.leverage_min <= leverage <= profile.leverage_max:
        hard_blocks.append(f"Hebel {leverage:.1f}x außerhalb {profile.leverage_min:.1f}–{profile.leverage_max:.1f}x")
    else:
        positives.append(f"Hebel {leverage:.1f}x passend")

    if ko_distance is None:
        cautions.append("Tatsächlicher KO-Abstand fehlt")
    elif not profile.ko_distance_min <= ko_distance <= profile.ko_distance_max:
        hard_blocks.append(f"KO-Abstand {ko_distance:.1f}% außerhalb {profile.ko_distance_min:.0f}–{profile.ko_distance_max:.0f}%")
    else:
        positives.append(f"KO-Abstand {ko_distance:.1f}% passend")

    if spread is None:
        cautions.append("Aktueller Geld-/Brief-Spread fehlt")
    elif spread > profile.max_spread_pct:
        hard_blocks.append(f"Spread {spread:.2f}% über Maximum {profile.max_spread_pct:.2f}%")
    else:
        positives.append(f"Spread {spread:.2f}% akzeptabel")

    # A technically required certificate loss above the emergency stop makes the entry unattractive.
    technical_underlying_stop = _finite(row.get("Stop"))
    underlying_entry = _finite(row.get("Entry"))
    estimated_stop_pct = None
    if technical_underlying_stop is not None and underlying_entry:
        base_move_pct = abs(technical_underlying_stop-underlying_entry)/underlying_entry*100
        effective_leverage = leverage or profile.leverage_target
        estimated_stop_pct = base_move_pct*effective_leverage
        if estimated_stop_pct > profile.emergency_stop_pct:
            hard_blocks.append(f"Technischer Stop entspräche ca. −{estimated_stop_pct:.1f}% im Zertifikat")
        elif estimated_stop_pct > profile.standard_stop_pct:
            cautions.append(f"Technischer Stop ca. −{estimated_stop_pct:.1f}%, über Standard −{profile.standard_stop_pct:.0f}%")
        else:
            positives.append(f"Technischer Stop ca. −{estimated_stop_pct:.1f}%")

    if hard_blocks:
        verdict, score = "🔴 KEIN ENTRY", 0
    elif cautions:
        verdict, score = "🟡 PRODUKT PRÜFEN / PULLBACK", 60
    else:
        verdict, score = "🟢 ENTRY FREIGEGEBEN", 85 if grade == "A" else 92 if grade == "A+" else 75

    quantity = math.floor(profile.preferred_investment/certificate_price) if certificate_price else None
    invested = quantity*certificate_price if quantity and certificate_price else profile.preferred_investment
    stop_eur = invested*profile.standard_stop_pct/100
    emergency_eur = invested*profile.emergency_stop_pct/100
    target_eur = invested*profile.initial_take_profit_pct/100
    stop_price = certificate_price*(1-profile.standard_stop_pct/100) if certificate_price else None
    emergency_price = certificate_price*(1-profile.emergency_stop_pct/100) if certificate_price else None
    target_price = certificate_price*(1+profile.initial_take_profit_pct/100) if certificate_price else None

    return {
        "verdict": verdict, "entry_score": score, "direction": direction,
        "grade": grade, "setup_hint": setup_hint, "hard_blocks": hard_blocks,
        "cautions": cautions, "positives": positives, "leverage": leverage,
        "ko_distance_pct": ko_distance, "spread_pct": spread,
        "certificate_price": certificate_price, "quantity": quantity,
        "investment": invested, "risk_standard_eur": stop_eur,
        "risk_emergency_eur": emergency_eur, "target_eur": target_eur,
        "stop_price": stop_price, "emergency_price": emergency_price,
        "target_price": target_price, "estimated_technical_stop_pct": estimated_stop_pct,
    }


def trailing_stop_pct(performance_pct: float, profile: KOProfile) -> float:
    """5%-step ladder: +5 -> BE, +10 -> +5, +15 -> +10, etc."""
    if performance_pct < profile.trailing_step_pct:
        return -profile.standard_stop_pct
    reached_step = math.floor(performance_pct/profile.trailing_step_pct)*profile.trailing_step_pct
    return reached_step-profile.trailing_step_pct


def render_entry_panel(row: Any, product: Optional[dict], profile: KOProfile) -> dict:
    a = entry_analysis(row, product, profile)
    st.subheader("🧭 KO Entry Analyzer")
    st.markdown(f"### {a['verdict']}")
    c = st.columns(5)
    c[0].metric("Entry Score", f"{a['entry_score']}/100")
    c[1].metric("Plan-Einsatz", f"{a['investment']:.0f} €")
    c[2].metric("Risiko −5%", f"−{a['risk_standard_eur']:.0f} €")
    c[3].metric("Max −8%", f"−{a['risk_emergency_eur']:.0f} €")
    c[4].metric("Ziel +10%", f"+{a['target_eur']:.0f} €")

    if a["certificate_price"]:
        p = st.columns(4)
        p[0].metric("Zertifikat Entry", f"{a['certificate_price']:.4f} €")
        p[1].metric("Stückzahl", a["quantity"] or "–")
        p[2].metric("Stop Standard", f"{a['stop_price']:.4f} €")
        p[3].metric("Ziel +10%", f"{a['target_price']:.4f} €")
    if a["hard_blocks"]:
        st.error(" · ".join(a["hard_blocks"]))
    if a["cautions"]:
        st.warning(" · ".join(a["cautions"]))
    if a["positives"]:
        st.success(" · ".join(a["positives"]))
    st.caption("Stop-Leiter: +5% → Break-even, +10% → Stop +5%, +15% → Stop +10%, +20% → Stop +15%.")
    return a
