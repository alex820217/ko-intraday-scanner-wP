from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import math
import streamlit as st


@dataclass(frozen=True)
class KOProfile:
    investment_min: float = 1000.0
    investment_max: float = 2000.0
    planned_investment: float = 1500.0
    leverage_min: float = 4.0
    leverage_max: float = 6.0
    ko_distance_min: float = 17.0
    ko_distance_max: float = 23.0
    standard_stop_pct: float = 5.0
    emergency_stop_pct: float = 8.0
    take_profit_pct: float = 10.0
    trailing_step_pct: float = 5.0
    max_spread_pct: float = 1.5


def render_profile_sidebar() -> KOProfile:
    with st.sidebar.expander("KO-Handelsprofil", expanded=False):
        investment_min = st.number_input("Einsatz min. EUR", 100.0, 10000.0, 1000.0, 100.0)
        investment_max = st.number_input("Einsatz max. EUR", investment_min, 20000.0, 2000.0, 100.0)
        planned = st.number_input("Plan-Einsatz EUR", investment_min, investment_max, min(1500.0, investment_max), 100.0)
        leverage_min = st.number_input("Hebel min.", 1.0, 20.0, 4.0, 0.5)
        leverage_max = st.number_input("Hebel max.", leverage_min, 30.0, 6.0, 0.5)
        ko_min = st.number_input("KO-Abstand min. %", 1.0, 50.0, 17.0, 1.0)
        ko_max = st.number_input("KO-Abstand max. %", ko_min, 80.0, 23.0, 1.0)
        stop = st.number_input("Standard-Stop %", 1.0, 20.0, 5.0, 0.5)
        emergency = st.number_input("Absolute Verlustgrenze %", stop, 30.0, 8.0, 0.5)
        target = st.number_input("Erstes Gewinnziel %", 2.0, 50.0, 10.0, 1.0)
        trailing = st.number_input("Trailing-Schritt %-Punkte", 1.0, 20.0, 5.0, 1.0)
        max_spread = st.number_input("Max. Spread %", 0.1, 10.0, 1.5, 0.1)
    return KOProfile(investment_min, investment_max, planned, leverage_min, leverage_max,
                     ko_min, ko_max, stop, emergency, target, trailing, max_spread)


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def trailing_stop_pct(performance_pct: float, profile: KOProfile) -> float:
    if performance_pct < profile.trailing_step_pct:
        return -profile.standard_stop_pct
    reached = math.floor(performance_pct / profile.trailing_step_pct) * profile.trailing_step_pct
    return reached - profile.trailing_step_pct


def analyze_ko_entry(row: Any, product: Optional[dict], profile: KOProfile) -> dict:
    product = product or {}
    signal = str(row.get("Signal", ""))
    grade = str(row.get("Setup Grade", "-"))
    entry_status = str(row.get("Entry Status", ""))
    rsi_status = str(row.get("RSI Entry", ""))

    blocks, warnings, positives = [], [], []
    if not any(token in signal for token in ("EARLY", "CONFIRMED")):
        blocks.append("Kein EARLY-/CONFIRMED-Signal")
    if grade not in {"A+", "A", "B"}:
        blocks.append(f"Setup-Grade {grade} nicht ausreichend")
    if "SPÄT" in entry_status or "LATE" in signal:
        blocks.append("Basiswert-Entry bereits zu spät")
    if "ÜBERHITZT" in rsi_status or "HOT" in signal:
        blocks.append("Momentum für Neueinstieg überhitzt")

    leverage = _number(product.get("leverage"))
    ko_distance = _number(product.get("ko_distance_pct"))
    spread = _number(product.get("spread_pct"))
    cert_price = _number(product.get("ask") or product.get("certificate_entry"))

    if leverage is None:
        warnings.append("Hebel des Produkts fehlt")
    elif profile.leverage_min <= leverage <= profile.leverage_max:
        positives.append(f"Hebel {leverage:.1f}x passend")
    else:
        blocks.append(f"Hebel {leverage:.1f}x außerhalb {profile.leverage_min:.1f}-{profile.leverage_max:.1f}x")

    if ko_distance is None:
        warnings.append("Tatsächlicher KO-Abstand fehlt")
    elif profile.ko_distance_min <= ko_distance <= profile.ko_distance_max:
        positives.append(f"KO-Abstand {ko_distance:.1f}% passend")
    else:
        blocks.append(f"KO-Abstand {ko_distance:.1f}% außerhalb {profile.ko_distance_min:.0f}-{profile.ko_distance_max:.0f}%")

    if spread is None:
        warnings.append("Aktueller Spread fehlt")
    elif spread <= profile.max_spread_pct:
        positives.append(f"Spread {spread:.2f}% akzeptabel")
    else:
        blocks.append(f"Spread {spread:.2f}% zu hoch")

    underlying_entry = _number(row.get("Entry"))
    underlying_stop = _number(row.get("Stop"))
    estimated_cert_stop_pct = None
    if underlying_entry and underlying_stop:
        base_stop_pct = abs(underlying_stop - underlying_entry) / underlying_entry * 100
        effective_leverage = leverage if leverage is not None else 5.0
        estimated_cert_stop_pct = base_stop_pct * effective_leverage
        if estimated_cert_stop_pct > profile.emergency_stop_pct:
            blocks.append(f"Technischer Stop entspräche ca. -{estimated_cert_stop_pct:.1f}% im Zertifikat")
        elif estimated_cert_stop_pct > profile.standard_stop_pct:
            warnings.append(f"Technischer Stop ca. -{estimated_cert_stop_pct:.1f}%, über Standard -{profile.standard_stop_pct:.0f}%")
        else:
            positives.append(f"Technischer Stop ca. -{estimated_cert_stop_pct:.1f}%")

    if blocks:
        verdict, entry_score = "KEIN ENTRY", 0
    elif warnings:
        verdict, entry_score = "PRODUKT PRÜFEN / PULLBACK", 60
    else:
        verdict = "ENTRY FREIGEGEBEN"
        entry_score = 92 if grade == "A+" else 85 if grade == "A" else 75

    quantity = math.floor(profile.planned_investment / cert_price) if cert_price else None
    investment = quantity * cert_price if quantity and cert_price else profile.planned_investment

    return {
        "verdict": verdict,
        "entry_score": entry_score,
        "blocks": blocks,
        "warnings": warnings,
        "positives": positives,
        "certificate_price": cert_price,
        "quantity": quantity,
        "investment": investment,
        "stop_price": cert_price * (1 - profile.standard_stop_pct / 100) if cert_price else None,
        "emergency_price": cert_price * (1 - profile.emergency_stop_pct / 100) if cert_price else None,
        "target_price": cert_price * (1 + profile.take_profit_pct / 100) if cert_price else None,
        "risk_standard_eur": investment * profile.standard_stop_pct / 100,
        "risk_emergency_eur": investment * profile.emergency_stop_pct / 100,
        "target_eur": investment * profile.take_profit_pct / 100,
    }


def render_entry_panel(row: Any, product: Optional[dict], profile: KOProfile) -> dict:
    result = analyze_ko_entry(row, product, profile)
    icon = "🟢" if result["verdict"] == "ENTRY FREIGEGEBEN" else "🟡" if "PRÜFEN" in result["verdict"] else "🔴"
    st.subheader("🧭 KO Entry Analyzer")
    st.markdown(f"### {icon} {result['verdict']}")

    cols = st.columns(5)
    cols[0].metric("Entry Score", f"{result['entry_score']}/100")
    cols[1].metric("Plan-Einsatz", f"{result['investment']:.0f} EUR")
    cols[2].metric(f"Risiko -{profile.standard_stop_pct:.0f}%", f"-{result['risk_standard_eur']:.0f} EUR")
    cols[3].metric(f"Max -{profile.emergency_stop_pct:.0f}%", f"-{result['risk_emergency_eur']:.0f} EUR")
    cols[4].metric(f"Ziel +{profile.take_profit_pct:.0f}%", f"+{result['target_eur']:.0f} EUR")

    if result["certificate_price"]:
        prices = st.columns(5)
        prices[0].metric("Zertifikat Entry", f"{result['certificate_price']:.4f} EUR")
        prices[1].metric("Stückzahl", result["quantity"] or "-")
        prices[2].metric("Stop", f"{result['stop_price']:.4f} EUR")
        prices[3].metric("Max Stop", f"{result['emergency_price']:.4f} EUR")
        prices[4].metric("Ziel", f"{result['target_price']:.4f} EUR")

    if result["blocks"]:
        st.error(" · ".join(result["blocks"]))
    if result["warnings"]:
        st.warning(" · ".join(result["warnings"]))
    if result["positives"]:
        st.success(" · ".join(result["positives"]))
    st.caption("Trailing: +5% -> Break-even, +10% -> Stop +5%, +15% -> Stop +10%, +20% -> Stop +15%.")
    return result
