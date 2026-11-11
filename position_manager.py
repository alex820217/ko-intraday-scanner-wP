from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

STORE = Path(__file__).with_name("positions.json")
HISTORY = Path(__file__).with_name("trade_history.json")


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write(path: Path, data: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_positions() -> list[dict[str, Any]]:
    return _read(STORE)


def save_position(position: dict[str, Any]) -> None:
    rows = load_positions()
    key = position["wkn"].strip().upper()
    position["wkn"] = key
    position["updated_at"] = datetime.now().isoformat(timespec="seconds")
    for i, row in enumerate(rows):
        if row.get("wkn", "").upper() == key:
            position.setdefault("created_at", row.get("created_at", position["updated_at"]))
            rows[i] = position
            break
    else:
        position["created_at"] = position["updated_at"]
        rows.append(position)
    _write(STORE, rows)


def delete_position(wkn: str) -> None:
    _write(STORE, [p for p in load_positions() if p.get("wkn") != wkn])


def close_position(wkn: str, exit_certificate_price: float, note: str = "") -> None:
    rows = load_positions()
    open_pos = next((p for p in rows if p.get("wkn") == wkn), None)
    if not open_pos:
        return
    closed = dict(open_pos)
    closed.update({
        "closed_at": datetime.now().isoformat(timespec="seconds"),
        "exit_certificate_price": float(exit_certificate_price),
        "close_note": note,
        "realized_pl_eur": (float(exit_certificate_price) - float(closed["certificate_entry"])) * float(closed["quantity"]),
    })
    history = _read(HISTORY)
    history.append(closed)
    _write(HISTORY, history)
    delete_position(wkn)


def load_history() -> list[dict[str, Any]]:
    return _read(HISTORY)


def evaluate_position(position: dict[str, Any], d15, d5) -> dict[str, Any]:
    r15, r5 = d15.iloc[-1], d5.iloc[-1]
    direction = position["direction"]
    current = float(r5["Close"])
    entry = float(position["underlying_entry"])
    atr = float(r15["ATR"])
    initial_stop = float(position.get("initial_stop") or (entry - 1.2 * atr if direction == "LONG" else entry + 1.2 * atr))
    initial_risk = abs(entry - initial_stop)
    signed_move = current - entry if direction == "LONG" else entry - current
    r_value = signed_move / initial_risk if initial_risk > 0 else 0.0

    ema20 = float(r15["EMA20"])
    vwap = float(r15["VWAP"])
    trend_ok = current > ema20 if direction == "LONG" else current < ema20
    vwap_ok = current > vwap if direction == "LONG" else current < vwap
    momentum_ok = float(r15["MACD_HIST"]) > 0 if direction == "LONG" else float(r15["MACD_HIST"]) < 0

    atr_trail = current - 1.2 * atr if direction == "LONG" else current + 1.2 * atr
    technical_trail = max(ema20, atr_trail) if direction == "LONG" else min(ema20, atr_trail)
    break_even = entry
    if r_value >= 1.8:
        active_stop = max(break_even, technical_trail) if direction == "LONG" else min(break_even, technical_trail)
    elif r_value >= 1.0:
        active_stop = break_even
    else:
        active_stop = initial_stop

    stop_hit = current <= active_stop if direction == "LONG" else current >= active_stop
    invalidated = stop_hit or (not trend_ok and not momentum_ok)
    if invalidated:
        action, reason = "🔴 EXIT", "Technischer Stop oder Trend/Momentum-Invalidierung"
    elif r_value >= 1.8:
        action, reason = "🟠 TEILGEWINN PRÜFEN", "TP1-Zone erreicht; Restposition mit Trailing-Stop führen"
    elif r_value >= 1.0:
        action, reason = "🟡 STOP NACHZIEHEN", "Mindestens +1R erreicht; Break-even prüfen"
    else:
        action, reason = "🟢 HALTEN", "Trend und technischer Stop sind intakt"

    ko = float(position["ko_level"])
    ko_distance_pct = abs(current - ko) / current * 100 if current else 0.0
    stop_ko_gap = abs(active_stop - ko)
    ko_wrong_side = ko >= active_stop if direction == "LONG" else ko <= active_stop
    ko_status = "🔴 KO-RISIKO" if ko_wrong_side or ko_distance_pct < 1.0 else "🟡 ENG" if ko_distance_pct < 2.0 else "🟢 AUSREICHEND"

    cert_entry = float(position["certificate_entry"])
    cert_current = float(position.get("certificate_current") or 0)
    quantity = float(position["quantity"])
    invested = cert_entry * quantity
    current_value = cert_current * quantity if cert_current > 0 else None
    pl_eur = (cert_current - cert_entry) * quantity if cert_current > 0 else None
    pl_pct = (cert_current / cert_entry - 1) * 100 if cert_current > 0 and cert_entry > 0 else None

    tp1 = entry + 1.8 * initial_risk if direction == "LONG" else entry - 1.8 * initial_risk
    tp2 = entry + 2.8 * initial_risk if direction == "LONG" else entry - 2.8 * initial_risk
    return {
        "action": action, "reason": reason, "current_underlying": current,
        "initial_stop": initial_stop, "active_stop": active_stop, "tp1": tp1, "tp2": tp2,
        "r_value": r_value, "ko_distance_pct": ko_distance_pct, "stop_ko_gap": stop_ko_gap,
        "ko_status": ko_status, "trend_ok": trend_ok, "vwap_ok": vwap_ok,
        "momentum_ok": momentum_ok, "invested": invested, "current_value": current_value,
        "pl_eur": pl_eur, "pl_pct": pl_pct,
    }
