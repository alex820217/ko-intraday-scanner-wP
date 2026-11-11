from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urljoin
import requests
from bs4 import BeautifulSoup

BASE = 'https://www.onvista.de'
HEADERS = {'User-Agent':'Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36','Accept-Language':'de-DE,de;q=0.9'}

@dataclass
class ProductData:
    wkn:str; isin:Optional[str]=None; product_name:Optional[str]=None; issuer:Optional[str]=None
    underlying:Optional[str]=None; direction:Optional[str]=None; ko_level:Optional[float]=None
    ratio:Optional[float]=None; leverage:Optional[float]=None; bid:Optional[float]=None; ask:Optional[float]=None
    bid_volume:Optional[int]=None; ask_volume:Optional[int]=None; spread_abs:Optional[float]=None
    spread_pct:Optional[float]=None; currency:Optional[str]=None; quote_timestamp:Optional[str]=None
    fetched_at:Optional[str]=None; source_name:str='Onvista'; source_url:Optional[str]=None
    quote_verified:bool=False; quote_warning:Optional[str]=None; error:Optional[str]=None
    def to_dict(self): return asdict(self)

def num(s):
    if not s: return None
    v=re.sub(r'[^0-9,.-]','',s)
    if not v:return None
    if ',' in v:v=v.replace('.','').replace(',','.')
    try:return float(v)
    except ValueError:return None

def integer(s):
    x=num(s); return int(x) if x is not None else None

def one(pattern,text,flags=re.I|re.S):
    m=re.search(pattern,text,flags); return m.group(1).strip() if m else None

def find_url(wkn,session):
    r=session.get(f'{BASE}/suche/?searchValue={quote(wkn)}',timeout=20); r.raise_for_status()
    soup=BeautifulSoup(r.text,'html.parser')
    for a in soup.select('a[href]'):
        href=a.get('href',''); label=a.get_text(' ',strip=True)
        if wkn.upper() in (href+' '+label).upper() and '/derivate/' in href:return urljoin(BASE,href)
    return None

def parse_quote_blocks(text):
    # Critical fix: volume belongs between Geld/Brief label and price, e.g.
    # "Geld • 22.500 Stk. 6,220 EUR". Capture volume and price separately.
    bid_pat=r'Geld\s*[•·]?\s*([0-9.]+)\s*Stk\.?\s*([0-9.,]+)\s*(EUR|USD|GBP|GBp|CHF)'
    ask_pat=r'Brief\s*[•·]?\s*([0-9.]+)\s*Stk\.?\s*([0-9.,]+)\s*(EUR|USD|GBP|GBp|CHF)'
    bm=re.search(bid_pat,text,re.I); am=re.search(ask_pat,text,re.I)
    if not (bm and am): return None
    return {'bid_volume':integer(bm.group(1)),'bid':num(bm.group(2)),'ask_volume':integer(am.group(1)),
            'ask':num(am.group(2)),'currency':bm.group(3)}

def validate_quote(q):
    if not q:return False,'Geld-/Briefblock nicht eindeutig gefunden'
    bid,ask=q['bid'],q['ask']
    if bid is None or ask is None:return False,'Geld oder Brief fehlt'
    if bid<=0 or ask<=0:return False,'Kurs muss positiv sein'
    if ask<bid:return False,'Brief liegt unter Geld'
    spread=(ask-bid)/bid*100
    if spread>20:return False,f'Spread unplausibel hoch ({spread:.2f} %)'
    if q['bid_volume'] and abs(bid-q['bid_volume'])<1e-9:return False,'Geldkurs entspricht Stückzahl'
    if q['ask_volume'] and abs(ask-q['ask_volume'])<1e-9:return False,'Briefkurs entspricht Stückzahl'
    return True,None

def resolve_wkn(wkn):
    wkn=re.sub(r'\s+','',wkn).upper(); out=ProductData(wkn=wkn,fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    if not re.fullmatch(r'[A-Z0-9]{6}',wkn):out.error='WKN muss genau 6 Zeichen haben.'; return out.to_dict()
    s=requests.Session(); s.headers.update(HEADERS)
    try:
        url=find_url(wkn,s)
        if not url:out.error='Keine öffentliche Produktseite gefunden.'; return out.to_dict()
        r=s.get(url,timeout=20); r.raise_for_status(); soup=BeautifulSoup(r.text,'html.parser'); text=' '.join(soup.stripped_strings)
        out.source_url=url; out.product_name=soup.title.get_text(' ',strip=True).split('•')[0].strip() if soup.title else None
        out.isin=one(r'ISIN\s+([A-Z]{2}[A-Z0-9]{10})',text)
        out.issuer=one(r'Emittent\s+(.+?)(?=\s+(?:Basiswert|Geld|Brief|WKN|ISIN|Hebel))',text)
        out.underlying=one(r'Basiswert\s+(.+?)(?=\s+(?:Basispreis|K\.O\.|K\.O|Hebel|Bezugsverhältnis))',text)
        blob=(out.product_name or '')+' '+text[:1200]
        out.direction='LONG' if re.search(r'\b(?:LONG|CALL)\b',blob,re.I) else 'SHORT' if re.search(r'\b(?:SHORT|PUT)\b',blob,re.I) else None
        out.ko_level=num(one(r'(?:K\.O\.-?Schwelle|K\.O\.|K\.O)\s+([0-9.,]+)',text))
        out.ratio=num(one(r'Bezugsverhältnis\s+([0-9.,]+)',text)); out.leverage=num(one(r'Hebel\s+([0-9.,]+)\s*x?',text))
        q=parse_quote_blocks(text); ok,warn=validate_quote(q); out.quote_verified=ok; out.quote_warning=warn
        if q:
            out.bid=q['bid']; out.ask=q['ask']; out.bid_volume=q['bid_volume']; out.ask_volume=q['ask_volume']; out.currency=q['currency']
            if ok:
                out.spread_abs=out.ask-out.bid; out.spread_pct=out.spread_abs/out.bid*100
            else:
                # Never let an unverified quote drive P/L.
                out.bid=out.ask=out.spread_abs=out.spread_pct=None
        out.quote_timestamp=one(r'(\d{2}\.\d{2}\.\d{4},\s*\d{2}:\d{2}:\d{2})',text) or one(r'(\d{2}\.\d{2}\.\d{2},\s*\d{2}:\d{2}:\d{2})',text)
        if not all([out.underlying,out.direction,out.ko_level is not None]):out.error='Kerndaten nicht vollständig/eindeutig. Bitte prüfen.'
    except requests.RequestException as e:out.error=f'Produktquelle nicht erreichbar: {type(e).__name__}'
    except Exception as e:out.error=f'Produktdaten nicht verarbeitbar: {type(e).__name__}'
    return out.to_dict()
