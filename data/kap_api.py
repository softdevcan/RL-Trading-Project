"""
KAP VYK API istemcisi (MKK; `docs/apispec.json`, OpenAPI 3.0.3) — Faz 9 E1 donem ici endeks olaylari.

Kimlik bilgisi YALNIZ ortam degiskeninden / `.env`'den (depoya girmez):
    KAP_API_KEY (eski ad KAP_API_TOKEN) + KAP_API_SECRET   MKK portalindaki API anahtari ve sirri.
    Kip sirasi: HTTP Basic anahtar:sir (spec'teki `basicAuth`), olmazsa /auth/generateToken
    (client_credentials) ile 24 saatlik JWT -> Bearer. Belirtec diske yazilmaz.
    KAP_API_BASE    taban adres; varsayilan spec'teki sunucu (apigwdev = GELISTIRME gecidi — uretim
                    verisi oldugu dogrulanmadan olay listesine kaynak sayilmaz, bkz. `verify_known`)
Ucretsiz plan dakikada 6 cagri: her cagri arasinda >= 10,5 sn beklenir (`MIN_INTERVAL`).
Ham yanitlar `data/universe/sources/kap_api/` altina yazilir (gitignore); turetilmis dosyalar
yalniz SHA-256 ve bildirim kimligini tasir.
"""

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = 'https://apigwdev.mkk.com.tr/api/vyk'
RAW_DIR = os.path.join(ROOT, 'data', 'universe', 'sources', 'kap_api')
MIN_INTERVAL = 10.5
_last_call = [0.0]


def _load_env() -> None:
    path = os.path.join(ROOT, '.env')
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('KAP_') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_auth_mode = [None]   # 'basic' | 'bearer' (ilk basarili kip hatirlanir)
_jwt = [None, 0.0]    # (belirtec, alinma zamani); 24 saat gecerli, 23 saatte yenilenir


def _key_secret():
    _load_env()
    key = os.environ.get('KAP_API_KEY') or os.environ.get('KAP_API_TOKEN')
    return key, os.environ.get('KAP_API_SECRET')


def _gateway_host() -> str:
    p = urllib.parse.urlsplit(base_url())
    return f'{p.scheme}://{p.netloc}'


def generate_token() -> str:
    """/auth/generateToken (OAuth istemci kimligi): client_id = API anahtari, client_secret = API sirri.
    Belirtec bellekte tutulur, diske yazilmaz."""
    if _jwt[0] and time.monotonic() - _jwt[1] < 23 * 3600:
        return _jwt[0]
    key, secret = _key_secret()
    if not (key and secret):
        raise RuntimeError('KAP: .env icinde KAP_API_KEY (ya da KAP_API_TOKEN) ve KAP_API_SECRET gerekli')
    form = urllib.parse.urlencode({'grant_type': 'client_credentials', 'client_id': key,
                                   'client_secret': secret}).encode()
    req = urllib.request.Request(_gateway_host() + '/auth/generateToken', data=form, method='POST',
                                 headers={'Content-Type': 'application/x-www-form-urlencoded',
                                          'Accept': 'application/json', 'User-Agent': 'research-client/1.0'})
    _throttle()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'KAP generateToken {e.code}: {e.read()[:300]!r}') from None
    tok = d.get('token') or d.get('access_token')
    if not tok:
        raise RuntimeError(f'KAP generateToken yanitinda belirtec yok: alanlar {list(d)}')
    _jwt[0], _jwt[1] = tok, time.monotonic()
    return tok


def _auth_header(mode: str) -> Dict[str, str]:
    key, secret = _key_secret()
    if mode == 'basic':
        if not (key and secret):
            raise RuntimeError('KAP: .env icinde KAP_API_KEY (ya da KAP_API_TOKEN) ve KAP_API_SECRET gerekli')
        return {'Authorization': 'Basic ' + base64.b64encode(f'{key}:{secret}'.encode()).decode()}
    return {'Authorization': f'Bearer {generate_token()}'}


def _throttle() -> None:
    wait = MIN_INTERVAL - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def base_url() -> str:
    _load_env()
    return os.environ.get('KAP_API_BASE', DEFAULT_BASE).rstrip('/')


def _raw_name(path: str, params: Optional[dict]) -> str:
    key = path.strip('/').replace('/', '_') + ('_' + urllib.parse.urlencode(sorted(params.items()), doseq=True)
                                               if params else '')
    safe = ''.join(c if c.isalnum() or c in '-_=.' else '_' for c in key)[:150]
    return safe + '.json'


def get(path: str, params: Optional[dict] = None, use_cache: bool = True, retries: int = 3):
    """GET + hiz siniri + ham arsiv. Onbellekte varsa ag cagrisi yapilmaz."""
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, _raw_name(path, params))
    if use_cache and os.path.exists(raw_path):
        with open(raw_path, encoding='utf-8') as fh:
            return json.load(fh)['body']
    url = base_url() + path + ('?' + urllib.parse.urlencode(params, doseq=True) if params else '')
    # Kip: KAP_AUTH (varsayilan 'basic' — test gecidinde dogrulandi 2026-11-15; 'bearer' uretim icin).
    mode = _auth_mode[0] = os.environ.get('KAP_AUTH', 'basic')
    text, last_err = None, None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={**_auth_header(mode), 'Accept': 'application/json',
                                                   'User-Agent': 'research-client/1.0'})
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode('utf-8')
            break
        except urllib.error.HTTPError as e:
            body = e.read()[:400]
            last_err = f'KAP {e.code} {path} ({mode}): {body!r}'
            if e.code == 429 and attempt < retries - 1:
                time.sleep(65)
                continue
            if e.code == 401 and b'ER006' in body and mode == 'bearer' and attempt < retries - 1:
                _jwt[0] = None   # suresi dolmus belirtec
                continue
            break
    if text is None:
        raise RuntimeError(last_err)
    body = json.loads(text)
    with open(raw_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'url_path': path, 'params': params, 'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'body': body},
                  fh, ensure_ascii=False, indent=1)
    return body


def members() -> List[dict]:
    return get('/members')


def last_disclosure_index() -> int:
    return int(get('/lastDisclosureIndex', use_cache=False)['lastDisclosureIndex'])


def disclosures(from_index: int, company_ids: Optional[List[str]] = None,
                types: Optional[str] = None, klass: Optional[str] = None) -> List[dict]:
    """from_index'ten itibaren ilk 50 bildirim (spec)."""
    p = {'disclosureIndex': str(from_index)}
    if company_ids:
        p['companyId'] = list(company_ids)
    if types:
        p['disclosureTypes'] = types
    if klass:
        p['disclosureClass'] = klass
    return get('/disclosures', p)


def detail(index: int, file_type: str = 'data') -> dict:
    return get(f'/disclosureDetail/{index}', {'fileType': file_type})
