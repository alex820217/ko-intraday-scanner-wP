"""UI helper for verified bid/ask data in KO Market Radar V4.1."""
def quote_card(product):
    if not product.get('quote_verified'):
        return {
            'status':'⚫ PRODUKTKURS NICHT VERIFIZIERT',
            'bid':'–','ask':'–','spread':'–',
            'bid_volume':product.get('bid_volume'),'ask_volume':product.get('ask_volume'),
            'warning':product.get('quote_warning') or 'Brokerkurs prüfen.'
        }
    bid=float(product['bid']); ask=float(product['ask'])
    return {
        'status':'✅ VERIFIZIERT','bid':bid,'ask':ask,
        'spread_abs':ask-bid,'spread_pct':(ask-bid)/bid*100,
        'bid_volume':product.get('bid_volume'),'ask_volume':product.get('ask_volume'),
        'warning':None,
    }
