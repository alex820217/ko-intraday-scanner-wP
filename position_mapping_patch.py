"""Drop-in helpers for KO Market Radar V4.1 position input."""
from market_universe import MARKET_GROUPS
from position_asset_mapping import resolve_underlying_to_ticker


def build_position_asset_options():
    options = {}
    for _group, assets in MARKET_GROUPS.items():
        for asset_name, ticker in assets.items():
            options[f"{asset_name} ({ticker})"] = ticker
    return dict(sorted(options.items(), key=lambda item: item[0].casefold()))


def auto_map_product(product):
    options = build_position_asset_options()
    label, ticker, confidence, reason = resolve_underlying_to_ticker(product.get("underlying"), options)
    return {
        "asset_options": options,
        "selected_label": label,
        "ticker": ticker,
        "confidence": confidence,
        "reason": reason,
    }
