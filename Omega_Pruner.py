# Omega_Pruner.py
import gradio as gr
import requests, time, base64, io, qrcode, json, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import warnings, logging

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# GLOBAL REQUESTS SESSION — NUCLEAR GRADE
# =========================
session = requests.Session()
session.headers.update({
    "User-Agent": "OmegaPruner-v11",
    "Accept": "application/json",
    "Connection": "keep-alive",
})

# Reuse connections, auto-retry transient failures
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=50,
    max_retries=requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
)
session.mount("https://", adapter)
session.mount("http://", adapter)


def safe_get(url: str):
    """Robust GET with 3 retries + exponential backoff"""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:  # Rate-limited
                time.sleep(1.5 ** attempt)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException):
            if attempt < 2:
                time.sleep(0.2 * (2 ** attempt))  # exponential backoff
    return None
# =========================
# Config / Constants
# =========================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"
MEMPOOL_API = "https://mempool.space/api"
BLOCKSTREAM_API = "https://blockstream.info/api/address/{addr}/utxo"
BITCOINER_API = "https://bitcoiner.live/api/address/{addr}/utxo"
MIN_KEEP_UTXOS = 1  # minimum UTXOs to keep regardless of strategy

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def sats_to_btc_str(sats: int) -> str:
    btc = sats / 100_000_000
    if btc >= 1:
        return f"{btc:,.8f}".rstrip("0").rstrip(".") + " BTC"
    return f"{int(sats):,} sats"

# -----------------------
# Minimal address helpers (used to estimate weights)
# -----------------------
def base58_decode(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + BASE58_ALPHABET.index(c)
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + n.to_bytes((n.bit_length() + 7) // 8, 'big')

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_verify_checksum(hrp, data):
    return bech32_polymod([ord(c)>>5 for c in hrp] + [0] + [ord(c)&31 for c in hrp] + data) == 1

def bech32m_verify_checksum(hrp, data):
    return bech32_polymod([ord(c)>>5 for c in hrp] + [0] + [ord(c)&31 for c in hrp] + data) == 0x2bc830a3

def convertbits(data, frombits, tobits, pad=True):
    acc = bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append(acc >> bits & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or (acc << (tobits - bits)) & maxv:
        raise ValueError("Invalid padding")
    return ret

def address_to_script_pubkey(addr: str) -> Tuple[bytes, Dict[str, Any]]:
    addr = (addr or "").strip().lower()
    if not addr:
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'unknown'}

    if addr.startswith('1'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x00:
                return b'\x76\xa9\x14' + dec[1:21] + b'\x88\xac', {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
        except: pass

    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:
                return b'\xa9\x14' + dec[1:21] + b'\x87', {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
        except: pass

    if addr.startswith('bc1q'):
        data = [CHARSET.find(c) for c in addr[4:] if c in CHARSET]
        if data and data[0] == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 20:
                return b'\x00\x14' + bytes(prog), {'input_vb': 68, 'output_vb': 31, 'type': 'P2WPKH'}
            if prog and len(prog) == 32:
                return b'\x00\x20' + bytes(prog), {'input_vb': 69, 'output_vb': 43, 'type': 'P2WSH'}

    if addr.startswith('bc1p'):
        data = [CHARSET.find(c) for c in addr[4:] if c in CHARSET]
        if data and len(data) > 1 and data[0] == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}

    return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'fallback'}

# -----------------------
# Simple TX / PSBT helpers (same approach you used before)
# -----------------------
def encode_varint(i: int) -> bytes:
    if i < 0xfd: return bytes([i])
    if i < 0x10000: return b'\xfd' + i.to_bytes(2, 'little')
    if i < 0x100000000: return b'\xfe' + i.to_bytes(4, 'little')
    return b'\xff' + i.to_bytes(8, 'little')

@dataclass
class TxIn:
    prev_tx: bytes
    prev_index: int
    script_sig: bytes = b''
    sequence: int = 0xfffffffd   # Enable RBF

    def serialize(self) -> bytes:
        return (
            self.prev_tx[::-1] +
            self.prev_index.to_bytes(4, 'little') +
            encode_varint(len(self.script_sig)) + self.script_sig +
            self.sequence.to_bytes(4, 'little')
        )

@dataclass
class TxOut:
    amount: int
    script_pubkey: bytes

    def serialize(self) -> bytes:
        return self.amount.to_bytes(8, 'little') + encode_varint(len(self.script_pubkey)) + self.script_pubkey

@dataclass
class Tx:
    version: int = 2
    tx_ins: List[TxIn] = None
    tx_outs: List[TxOut] = None
    locktime: int = 0

    def __post_init__(self):
        self.tx_ins = self.tx_ins or []
        self.tx_outs = self.tx_outs or []

def create_psbt(tx_hex: str) -> str:
    psbt = b'psbt\xff\x00' + encode_varint(len(bytes.fromhex(tx_hex))) + bytes.fromhex(tx_hex) + b'\x00\xff'
    return base64.b64encode(psbt).decode()

# -----------------------
# UTXO fetching (tries multiple public APIs)
# -----------------------
def get_utxos(addr: str, dust: int = 546) -> List[dict]:
    addr = (addr or "").strip()
    if not addr:
        return []

    apis = [
        f"{MEMPOOL_API}/address/{addr}/utxo",
        BLOCKSTREAM_API.format(addr=addr),
        BITCOINER_API.format(addr=addr),
    ]

    for url in apis:
        r = safe_get(url)
        if not r:
            continue
        try:
            data = r.json()
            if not data:
                continue

            utxos = []
            iter_u = data if isinstance(data, list) else data.get("utxos", [])
            for u in iter_u:
                val = int(u.get("value", 0))
                if val > dust:
                    utxos.append({
                        "txid": u["txid"],
                        "vout": u["vout"],
                        "value": val,
                        "address": addr,
                        "confirmed": u.get("status", {}).get("confirmed", True)
                                     if isinstance(u.get("status"), dict) else True
                    })
            if utxos:
                return utxos
        except Exception:
            continue
        time.sleep(0.05)  # be nice

    return []

# -----------------------
# xpub scan (optional hdwallet)
# -----------------------
try:
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC as HDWALLET_BTC
    HDWALLET_AVAILABLE = True
except Exception:
    HDWALLET_AVAILABLE = False

def scan_xpub(xpub: str, dust: int = 546, gap_limit: int = 20):
    if not HDWALLET_AVAILABLE:
        return [], "Install: pip install hdwallet"
    all_utxos = []
    try:
        hdw = HDWallet(symbol=HDWALLET_BTC).from_xpublic_key(xpub.strip())
        purpose = 84 if xpub.startswith(("zpub", "vpub")) else 49 if xpub.startswith(("ypub", "upub")) else 44
        for change in [0, 1]:
            empty = 0
            for i in range(200):
                path = f"m/{purpose}'/0'/{change}/{i}"
                hdw.from_path(path)
                addr = hdw.p2wpkh_address() if purpose == 84 else hdw.p2sh_p2wpkh_address() if purpose == 49 else hdw.p2pkh_address()
                utxos = get_utxos(addr, dust)
                if utxos:
                    all_utxos.extend(utxos)
                    empty = 0
                else:
                    empty += 1
                    if empty >= gap_limit:
                        break
                time.sleep(0.02)
        all_utxos.sort(key=lambda x: x["value"], reverse=True)
        return all_utxos, f"Found {len(all_utxos)} UTXOs across {len(set(u['address'] for u in all_utxos))} addresses"
    except Exception as e:
        return [], f"Error: {str(e)[:120]}"



def analyze(addr, strategy, dust_threshold, dest_addr, fee_rate_slider, dao_slider):
    fee_rate = fee_rate_slider if isinstance(fee_rate_slider, (int, float)) else fee_rate_slider.value
    dao_percent = dao_slider if isinstance(dao_slider, (int, float)) else dao_slider.value
    addr = (addr or "").strip()
    if not addr:
        return "<div style='color:#ff3366;'>Enter address or xpub</div>", [], [], "", gr.update(visible=False), gr.update(visible=False)

    is_xpub = addr[:4] in ["xpub", "ypub", "zpub", "tpub", "upub", "vpub"]
    if is_xpub:
        utxos_raw, info = scan_xpub(addr, dust_threshold)
    else:
        utxos_raw = get_utxos(addr, dust_threshold)
        info = f"Single address • {len(utxos_raw)} UTXOs" if utxos_raw else "No UTXOs"

    if not utxos_raw:
        return f"<div style='color:#ff3366;'>No UTXOs above {dust_threshold} sats</div>", [], [], "", gr.update(visible=False), gr.update(visible=False)

    utxos = sorted(utxos_raw, key=lambda x: x["value"], reverse=True)

    # === CALCULATE PRE-PRUNE SIZE ONCE (for banner) ===
    all_input_weight = sum(
        (lambda meta: meta['input_vb'] * 4)(
            address_to_script_pubkey(u['address'])[1]
        ) for u in utxos
    )
    pre_vsize = max(
        (all_input_weight + 43*4 + 31*4 + 10*4 + len(utxos)) // 4 + 10,
        (all_input_weight + 150 + len(utxos)*60) // 4
    )

    ratio = {
        "Privacy First (30% pruned)": 0.30,
        "Recommended (40% pruned)": 0.40,
        "More Savings (50% pruned)": 0.50,
        "NUCLEAR PRUNE (90% sacrificed)": 0.90
    }.get(strategy, 0.40)

    keep = max(MIN_KEEP_UTXOS, int(len(utxos) * (1 - ratio)))
    default_selected = {f"{u['txid']}:{u['vout']}" for u in utxos[keep:]}

    enriched = []
    df_rows = []

    for u in utxos:
        _, meta = address_to_script_pubkey(u['address'])
        u2 = u.copy()
        u2['input_weight'] = meta['input_vb'] * 4
        enriched.append(u2)

        key = f"{u2['txid']}:{u2['vout']}"
        is_prune = key in default_selected

        df_rows.append([
            is_prune,
            u2['txid'],
            u2['vout'],
            u2['value'],
            u2['address'],
            u2['input_weight'] // 4   # vB per input
        ])

    # === UPGRADED BANNER WITH PRE-PRUNE SIZE ===
    status = f"""
<div style='text-align:center;margin:30px 0;padding:20px;background:#000;border:3px solid #f7931a;border-radius:16px;box-shadow:0 0 60px rgba(247,147,26,0.6);'>
    <div style='color:#0f0;font-size:2.4rem;font-weight:900;letter-spacing:2px;text-shadow:0 0 30px #0f0;'>
        ANALYSIS COMPLETE
    </div>
    <div style='color:#f7931a;font-size:1.8rem;font-weight:900;margin:12px 0;letter-spacing:1px;text-shadow:0 0 25px #f7931a;'>
        FOUND <span style='color:#0f0;text-shadow:0 0 30px #0f0;'>{len(utxos):,}</span> UTXOs
    </div>
    <div style='color:#fff;font-size:1.3rem;font-weight:700;opacity:0.9;'>
        Current tx size if sent today: <span style='color:#ff6600;font-weight:800;'>{pre_vsize:,}</span> vB
    </div>
    <div style='color:#0f8;font-size:1.1rem;margin-top:8px;opacity:0.88;'>
        {info}
    </div>
</div>
"""

    summary_html = generate_summary(df_rows, enriched, fee_rate, dao_percent)

    return status, df_rows, enriched, summary_html, gr.update(visible=True), gr.update(visible=True)


def generate_summary(df_rows: List[list], enriched_state: List[dict], fee_rate: int = 15, dao_percent: float = 0.5):
    selected_utxos = []
    for row in df_rows:
        if not row or len(row) < 5 or not row[0]:
            continue
        txid = row[1]
        vout = row[2]
        for u in enriched_state:
            if u['txid'] == txid and u['vout'] == vout:
                selected_utxos.append(u)
                break

    if not selected_utxos:
        return (
            "<div style='color:#868686;font-size:0.96rem;text-align:center;"
            "text-shadow:0 0 10px rgba(0,255,0,0.4);'>"
            "No UTXOs selected for pruning."
            "</div>"
        )

    # --- CORE MATH (MUST MATCH PSBT) ---
    total_in = sum(u['value'] for u in selected_utxos)
    input_weight = sum(u['input_weight'] for u in selected_utxos)
    base_vsize = (input_weight + 43*4 + 31*4 + 10*4 + len(selected_utxos)) // 4 + 10
    vsize = max(base_vsize, (input_weight + 150 + len(selected_utxos)*60) // 4)
    fee = max(600, int(vsize * fee_rate))
    remaining = total_in - fee

    if remaining <= 0:
        return f"""
    <div style='text-align:center;margin:20px;padding:20px;background:#330000;
                border:2px solid #ff3366;border-radius:14px;
                box-shadow:0 0 40px rgba(255,51,102,0.6);'>
        <div style='color:#ff3366;font-size:1.2rem;font-weight:800;'>
            Transaction Invalid
        </div>
        <div style='color:#fff;margin-top:10px;line-height:1.6;'>
            Fee exceeds available balance after pruning.<br>
            <strong>Reduce fee rate</strong> or <strong>select more UTXOs</strong>.
        </div>
        <div style='color:#ff9900;margin-top:10px;font-size:0.95rem;'>
            Total Inputs: {sats_to_btc_str(total_in)}<br>
            Estimated Fee: {fee:,} sats
        </div>
    </div>
    """

    # === DAO (dust-proof) ===
    dao_raw = int(remaining * (dao_percent / 100.0)) if dao_percent > 0 else 0
    dao_amt = dao_raw if dao_raw >= 546 else 0
    if 0 < dao_raw < 546:
        fee += dao_raw
    remaining -= dao_amt

    # === PRE/POST PRUNE SIZE + DYNAMIC COLOR SAVINGS ===
    all_input_weight = sum(u['input_weight'] for u in enriched_state)
    pre_vsize = max(
        (all_input_weight + 43*4 + 31*4 + 10*4 + len(enriched_state)) // 4 + 10,
        (all_input_weight + 150 + len(enriched_state)*60) // 4
    )
    post_vsize = vsize
    savings_pct = round(100 * (1 - post_vsize / pre_vsize), 1) if pre_vsize > 0 else 0

    # Dynamic color: green = nuclear win, orange = decent, red = weak prune
    if savings_pct >= 70:
        savings_color = "#0f0"
        savings_label = "NUCLEAR"
    elif savings_pct >= 50:
        savings_color = "#00ff9d"
        savings_label = "EXCELLENT"
    elif savings_pct >= 30:
        savings_color = "#ff9900"
        savings_label = "GOOD"
    else:
        savings_color = "#ff3366"
        savings_label = "WEAK"

    # === CHANGE ===
    change_amt = remaining if remaining >= 546 else 0
    if 0 < change_amt < 546:
        fee += change_amt
        change_amt = 0

    if total_in - fee <= 0:
        return f"""
    <div style='text-align:center;margin:20px;padding:20px;background:#330000;
            border:2px solid #ff3366;border-radius:14px;
            box-shadow:0 0 40px rgba(255,51,102,0.6);'>
        <div style='color:#ff3366;font-size:1.2rem;font-weight:900;'>
            Transaction Invalid
        </div>
        <div style='color:#fff;margin-top:10px;line-height:1.6;'>
            Fee exceeds available balance after DAO + dust absorption.<br>
            <strong>Reduce fee rate</strong> or <strong>select more UTXOs</strong>.
        </div>
    </div>
    """

    # === FINAL NUCLEAR DISPLAY ===
    return f"""
    <div style='text-align:center;margin:10px;padding:14px;background:#111;border:2px solid #f7931a;border-radius:14px;max-width:95%;font-size:1rem;line-height:2.1;'>
      <span style='color:#fff;font-weight:600;'>Selected Inputs:</span> 
      <span style='color:#0f0;font-weight:800;'>{len(selected_utxos):,}</span><br>

      <span style='color:#fff;font-weight:600;'>Total Pruned:</span> 
      <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(total_in)}</span><br>

      <span style='color:#fff;font-weight:600;'>Est. Fee:</span> 
      <span style='color:#0f0;font-weight:800;'>{fee:,} sats</span> 
      <span style='color:#0f0;font-weight:600;'>@</span> 
      <strong style='color:#0f0;font-weight:800;'>{fee_rate} s/vB</strong><br>

      <span style='color:#fff;font-weight:600;'>Pre-prune size:</span> 
      <span style='color:#888;font-weight:700;'>{pre_vsize:,} vB</span><br>

      <span style='color:#fff;font-weight:600;'>Post-prune size:</span> 
      <span style='color:#0f0;font-weight:800;'>{post_vsize:,} vB</span>
      <span style='color:{savings_color};font-weight:900;font-size:1.05rem;text-shadow:0 0 15px {savings_color};'>
        {' ' + savings_label} (-{savings_pct}%)
      </span><br>

      <span style='color:#fff;font-weight:600;'>Change:</span> 
      <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(change_amt)}</span>
      {f" <span style='color:#ff6600;font-size:0.9rem;'>(below 546 sats — absorbed into miner’s fee)</span>" if change_amt == 0 and remaining > 0 else ""}

      {f" • <span style='color:#ff6600;'>DAO:</span> <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(dao_amt)}</span>" if dao_amt >= 546 else ""}
      {f" • <span style='color:#ff6600;'>DAO:</span> <span style='color:#ff3366;font-weight:800;'>{sats_to_btc_str(dao_raw)}</span> <span style='color:#ff3366;font-size:0.9rem;'>(dust — absorbed)</span>" if 0 < dao_raw < 546 else ""}
    </div>

    <div style='text-align:center;margin:15px 0;padding:16px;background:#220000;border:2px solid #f7931a;border-radius:12px;box-shadow:0 0 50px rgba(247,147,26,0.5);'>
      <div style='color:#ff9900;font-size:1rem;font-weight:700;letter-spacing:1.2px;line-height:1.5;'>
        <strong style='text-shadow:1px 1px 0px #000, 0 0 8px rgba(255,153,0,0.9);'>Warning: Note:</strong> Combining UTXOs lowers fees but reduces privacy.<br class="mobile-break">
        <strong style='color:#ff9900;text-shadow:1px 1px 0px #000, 0 0 8px rgba(255,102,0,0.9);'>Choose the strategy that best fits your needs.</strong>
      </div>
    </div>

    <style>
    .mobile-break {{ display: none; }}
    @media (max-width: 768px) {{ .mobile-break {{ display: inline; }} }}
    </style>
    """

def generate_psbt(
    dest_addr: str,
    fee_rate: int,
    dao_percent: float,
    df_rows: list,
    enriched_state: list
):
    if not df_rows or not enriched_state:
        return "<div style='color:#ff3366; text-align:center; padding:30px;'>Run Analyze first.</div>", gr.update(), gr.update(), ""

    # === SAFELY EXTRACT SELECTED UTXOs (6+ columns safe) ===
    selected_utxos = []
    for row in df_rows:
        if not row or len(row) < 5 or not row[0]:
            continue
        txid = row[1]
        vout = row[2]
        for u in enriched_state:
            if u["txid"] == txid and u["vout"] == vout:
                selected_utxos.append(u)
                break

    if not selected_utxos:
        return "<div style='color:#ff3366; text-align:center; padding:30px;'>No UTXOs selected for pruning!</div>", gr.update(), gr.update(), ""

    # === CORE MATH ===
    total_in = sum(u["value"] for u in selected_utxos)
    input_count = len(selected_utxos)
    input_weight = sum(u.get("input_weight", 272) for u in selected_utxos)

    base_vsize = (input_weight + 43*4 + 31*4 + 10*4 + input_count) // 4 + 10
    vsize = max(base_vsize, (input_weight + 150 + input_count*60) // 4)
    fee = max(600, int(vsize * fee_rate))
    remaining = total_in - fee

    # === PRE-PRUNE SIZE (ALL UTXOs) ===
    all_input_weight = sum(u['input_weight'] for u in enriched_state)
    pre_vsize = max(
        (all_input_weight + 43*4 + 31*4 + 10*4 + len(enriched_state)) // 4 + 10,
        (all_input_weight + 150 + len(enriched_state)*60) // 4
    )
    savings_pct = round(100 * (1 - vsize / pre_vsize), 1) if pre_vsize > 0 else 0

    # Dynamic savings badge
    if savings_pct >= 70:
        savings_color = "#0f0"
        savings_label = "NUCLEAR"
    elif savings_pct >= 50:
        savings_color = "#00ff9d"
        savings_label = "EXCELLENT"
    elif savings_pct >= 30:
        savings_color = "#ff9900"
        savings_label = "GOOD"
    else:
        savings_color = "#ff3366"
        savings_label = "WEAK"

    # === FAILSTOPS ===
    if remaining <= 0:
        return (f"""
        <div style='text-align:center;margin:20px;padding:20px;background:#330000;
                    border:2px solid #ff3366;border-radius:14px;
                    box-shadow:0 0 40px rgba(255,51,102,0.6);'>
            <div style='color:#ff3366;font-size:1.2rem;font-weight:900;'>Transaction Invalid</div>
            <div style='color:#fff;margin-top:10px;line-height:1.6;'>
                Fee exceeds available balance after pruning.<br>
                <strong>Reduce fee rate</strong> or <strong>select more UTXOs</strong>.
            </div>
        </div>
        """, gr.update(visible=False), gr.update(visible=False), "")

    # === DAO + CHANGE (dust-proof) ===
    dao_raw = int(remaining * (dao_percent / 100.0)) if dao_percent > 0 else 0
    dao_amt = dao_raw if dao_raw >= 546 else 0
    if 0 < dao_raw < 546:
        fee += dao_raw
    remaining -= dao_amt

    dest = (dest_addr or "").strip() or selected_utxos[0]["address"]
    if not dest.startswith(("1","3","bc1")):
        return "<div style='color:#ff3366;text-align:center;'>Invalid destination address.</div>", gr.update(), gr.update(), ""

    dest_spk, _ = address_to_script_pubkey(dest)
    dao_spk, _  = address_to_script_pubkey(DEFAULT_DAO_ADDR)

    change_amt = remaining if remaining >= 546 else 0
    if 0 < change_amt < 546:
        fee += change_amt
        change_amt = 0

    if total_in - fee <= 0:
        return (f"""
        <div style='text-align:center;margin:20px;padding:20px;background:#330000;
                border:2px solid #ff3366;border-radius:14px;
                box-shadow:0 0 40px rgba(255,51,102,0.6);'>
            <div style='color:#ff3366;font-size:1.2rem;font-weight:900;'>Transaction Invalid</div>
            <div style='color:#fff;margin-top:10px;line-height:1.6;'>
                Fee exceeds balance after DAO + dust absorption.
            </div>
        </div>
        """, gr.update(visible=False), gr.update(visible=False), "")

    # === BUILD TX ===
    tx = Tx()
    for u in selected_utxos:
        tx.tx_ins.append(TxIn(bytes.fromhex(u["txid"]), int(u["vout"])))

    if dao_amt >= 546:
        tx.tx_outs.append(TxOut(dao_amt, dao_spk))
    if change_amt >= 546:
        tx.tx_outs.append(TxOut(change_amt, dest_spk))

    raw_tx = (
        tx.version.to_bytes(4, 'little') +
        b'\x00\x01' +
        encode_varint(len(tx.tx_ins)) +
        b''.join(i.serialize() for i in tx.tx_ins) +
        encode_varint(len(tx.tx_outs)) +
        b''.join(o.serialize() for o in tx.tx_outs) +
        b'\x00' * len(tx.tx_ins) +
        tx.locktime.to_bytes(4, 'little')
    )

    psbt_b64 = create_psbt(raw_tx.hex())

    # === QR ===
    box_size = 6
    error_correction = qrcode.constants.ERROR_CORRECT_L if len(psbt_b64) > 2800 else qrcode.constants.ERROR_CORRECT_M
    qr = qrcode.QRCode(version=None, error_correction=error_correction, box_size=box_size, border=4)
    qr.add_data(f"bitcoin:?psbt={psbt_b64}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#f7931a", back_color="#000000")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    qr_img_html = f'<img src="{qr_uri}" style="width:100%;height:auto;display:block;image-rendering:crisp-edges;border-radius:12px;" alt="NUCLEAR QR"/>'

    qr_warning_html = ""
    if len(psbt_b64) > 2800:
        qr_warning_html = """
        <div style="margin-top:12px;padding:10px 14px;border-radius:8px;background:#331a00;color:#ffb347;
                     font-family:monospace;font-size:0.95rem;line-height:1.4;border:1px solid #ff9900aa;">
            Large PSBT detected.<br>
            If QR fails → tap <strong>COPY PSBT</strong> and paste directly.
        </div>
        """

    # === FINAL NUCLEAR RESULT PAGE ===
    return f"""
   <div style="text-align:center;margin:80px auto;max-width:960px;">
    <div style="display:inline-block;padding:55px;background:#000;border:14px solid #f7931a;border-radius:36px;
                box-shadow:0 0 140px rgba(247,147,26,0.95);
                background:radial-gradient(circle at center,#0a0a0a 0%,#000 100%);">

        <div style="margin:40px 0 60px;">
            <button disabled style="padding:30px 100px;font-size:2.4rem;font-weight:800;
                                    background:#000;color:#0f0;letter-spacing:8px;
                                    border:8px solid #0f0;border-radius:34px;
                                    box-shadow:0 0 100px #0f0;text-shadow:0 0 20px #0f0;cursor:not-allowed;">
                RAW UNSIGNED TRANSACTION
            </button>
        </div>

        <div style="margin:0 auto 40px;width:520px;max-width:96vw;padding:20px;background:#000;
                    border:8px solid #0f0;border-radius:24px;box-shadow:0 0 60px #0f0,inset 0 0 40px #0f0;">
            {qr_img_html}
        </div>

        {qr_warning_html}

        <!-- NUCLEAR SUMMARY BLOCK -->
        <div style='text-align:center;margin:40px 0;padding:18px;background:#111;border:2px solid #f7931a;
                    border-radius:14px;max-width:95%;font-size:1.4rem;line-height:2.1;'>
            <span style='color:#fff;font-weight:600;'>Inputs:</span> 
            <span style='color:#0f0;font-weight:800;'>{input_count:,}</span><br>

            <span style='color:#fff;font-weight:600;'>Total Pruned:</span> 
            <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(total_in)}</span><br>

            <span style='color:#fff;font-weight:600;'>Input Weight:</span> 
            <span style='color:#0f0;font-weight:800;'>{input_weight:,} wu → {input_weight//4:,} vB</span><br>

            <span style='color:#fff;font-weight:600;'>Pre-prune size:</span> 
            <span style='color:#888;font-weight:700;'>{pre_vsize:,} vB</span><br>

            <span style='color:#fff;font-weight:600;'>Post-prune size:</span> 
            <span style='color:#0f0;font-weight:800;'>{vsize:,} vB</span>
            <span style='color:{savings_color};font-weight:900;font-size:1.15rem;text-shadow:0 0 18px {savings_color};'>
                {' ' + savings_label} (-{savings_pct}%)
            </span><br>

            <span style='color:#fff;font-weight:600;'>Final Fee:</span> 
            <span style='color:#0f0;font-weight:800;'>{fee:,} sats</span> 
            <span style='color:#0f0;font-weight:600;'>@</span> 
            <strong style='color:#0f0;font-weight:800;'>{fee_rate} s/vB</strong><br>

            <span style='color:#fff;font-weight:600;'>Change:</span> 
            <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(change_amt)}</span>
            {f" <span style='color:#ff6600;'>(dust absorbed)</span>" if change_amt == 0 and remaining > 0 else ""}

            {f" • <span style='color:#ff6600;'>DAO:</span> <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(dao_amt)}</span>" if dao_amt >= 546 else ""}
            {f" • <span style='color:#ff6600;'>DAO:</span> <span style='color:#ff3366;font-weight:800;'>{sats_to_btc_str(dao_raw)}</span> <span style='color:#ff3366;'>(dust absorbed)</span>" if 0 < dao_raw < 546 else ""}
        </div>

        <!-- PSBT + COPY -->
        <div style="margin:60px auto 20px;width:92%;max-width:880px;">
            <div style="position:relative;background:#000;border:6px solid #f7931a;border-radius:18px;
                        box-shadow:0 0 40px #0f0;overflow:hidden;">
                <textarea id="psbt-output" readonly 
                    style="width:100%;height:180px;background:#000;color:#0f0;font-size:1rem;
                           padding:24px;padding-right:140px;border:none;outline:none;resize:none;
                           font-family:monospace;font-weight:700;box-sizing:border-box;">
                    {psbt_b64}</textarea>
                <button onclick="navigator.clipboard.writeText(document.getElementById('psbt-output').value).then(() => {{ 
                    this.innerText='COPIED'; setTimeout(() => this.innerText='COPY PSBT', 1500); 
                }})"
                    style="position:absolute;top:14px;right:14px;padding:12px 28px;background:#f7931a;
                           color:#000;border:none;border-radius:14px;font-weight:900;cursor:pointer;
                           font-size:1.1rem;z-index:10;">
                    COPY PSBT
                </button>
            </div>
            <div style="text-align:center;margin-top:12px;">
                <span style="color:#00f0ff;font-weight:700;text-shadow:0 0 10px #0f0;">RBF enabled</span>
                <span style="color:#888;"> • Raw PSBT • Tap COPY to clipboard</span>
            </div>
        </div>

        <div style='color:#ff9900;font-size:1rem;text-align:center;margin:40px 0 20px;padding:16px;
                    background:#220000;border:2px solid #f7931a;border-radius:12px;
                    box-shadow:0 0 40px rgba(247,147,26,0.4);'>
            <div style='color:#fff;font-weight:800;text-shadow:0 0 12px #f7931a;'>
                Important: Wallet must support <strong style='color:#0f0;text-shadow:0 0 15px #0f0;'>PSBT</strong>
            </div>
            <div style='color:#0f8;margin-top:8px;opacity:0.9;'>
                Sparrow • BlueWallet • Electrum • UniSat • OK
            </div>
        </div>

    </div>
    </div>
    """, gr.update(visible=False), gr.update(visible=False), ""
# --------------------------
# Gradio UI
# --------------------------

with gr.Blocks(
    title="Ωmega Pruner v10.2.1 — NUCLEAR EDITION"
) as demo:
    # NUCLEAR SOCIAL PREVIEW — THIS IS ALL YOU NEED NOW
    gr.HTML("""
    <meta property="og:title" content="Ωmega Pruner v10.2.1 — NUCLEAR EDITION">
    <meta property="og:description" content="The cleanest open-source UTXO consolidator. Zero custody. Full coin-control. RBF. Taproot.">
    <meta property="og:image" content="docs/omega_thumbnail.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:url" content="https://omega-pruner.onrender.com">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    """, visible=False)
    gr.HTML("""
<!-- Ω Background (unchanged, working, with gradient-scale pulse) -->
<div id="omega-bg" style="
    position: fixed !important;
    inset: 0 !important;
    top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    pointer-events: none !important;
    z-index: -1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: transparent;
">
    <span class="omega-wrap" style="
        display:inline-block;
        animation: 
            gradient-pulse 18s infinite ease-in-out,
            omega-spin 120s linear infinite;
    ">
        <span class="omega-symbol" style="
            font-size: 100vh !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, rgba(247,147,26,0.28), rgba(247,147,26,0.15)) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            color: transparent !important;
            text-shadow: 0 0 220px rgba(247,147,26,0.72) !important;
            animation: omega-breath 28s infinite ease-in-out !important;
            user-select: none !important;
            line-height: 1 !important;
            opacity: 0.96 !important;
        ">Ω</span>
    </span>
</div>

<!-- Floating Banner -->
<div id="banner-glow" style="text-align:center;margin:40px 0; position:relative; z-index:1;">
    <h1 id="headline" style="
        color:#f7931a;
        font-size:3rem;
        font-weight:900;
        letter-spacing:2px;
        text-shadow: 0 0 20px #f7931a;
        animation: headline-glow 2s infinite alternate;
        display:inline-block;
    ">
        N U C L E A R  &nbsp; C O I N  &nbsp; C O N T R O L
    </h1>
    <p id="subtitle" style="
        color:#0f0;
        font-size:1.3rem;
        font-weight:900;
        letter-spacing:0.6px;
        text-shadow:
            0 0 12px #0f0,                 /* neon core */
            0 0 24px #0f0,                 /* outer glow */
            0 0 40px #0f0,                 /* massive halo */
            0 2px 6px #000,                /* sharp black drop */
            0 6px 16px #000000cc,          /* deep black shadow */
            0 12px 32px #000000e6;         /* nuclear black halo */
        animation: subtitle-glow 2s infinite alternate;
        margin-top:12px;
    ">
        Prune dust. Consolidate UTXOs. Save money.
    </p>
</div>

<style>
@keyframes omega-breath {
    0%, 100% { opacity: 0.76; transform: scale(0.95) rotate(0deg); }
    50%      { opacity: 1.0;  transform: scale(1.05) rotate(180deg); }
}

/* NEW — subtle scale pulse on gradient wrapper */
@keyframes gradient-pulse {
    0%, 100% { transform: scale(0.97); }
    50%      { transform: scale(1.03); }
}

/* Banner text pulse glow - stronger intensity */
@keyframes headline-glow {
    0% { text-shadow: 0 0 20px #f7931a, 0 0 40px #ffaa00, 0 0 60px #f7931a; }
    100% { text-shadow: 0 0 50px #f7931a, 0 0 100px #ffaa00, 0 0 150px #f7931a; }
}
@keyframes subtitle-glow {
    0% { text-shadow: 0 0 12px black, 0 0 24px #00ff9d, 0 0 36px #0f0; }
    100% { text-shadow: 0 0 30px black, 0 0 60px #00ff9d, 0 0 90px #0f0; }
}

@keyframes omega-spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.gradio-container { 
    position: relative !important; 
    z-index: 0 !important; 
    background: transparent !important;
    overflow-y: auto !important;
}
body { overflow-y: auto !important; }

#omega-bg { 
    isolation: isolate !important; 
    will-change: transform, opacity !important; 
}

.omega-symbol, .omega-wrap { 
    will-change: transform, opacity !important; 
}
</style>
""", elem_id="omega-banner-intense")

    # Input rows
    with gr.Row():
        addr = gr.Textbox(
            label="Address or xpub • No private keys ever entered • 100% Non-custodial",
            placeholder="bc1q... | zpub... | xpub...",
            lines=2,
            scale=3
        )
        strategy = gr.Dropdown(
            choices=[
                "Privacy First (30% pruned)",
                "Recommended (40% pruned)",
                "More Savings (50% pruned)",
                "NUCLEAR PRUNE (90% sacrificed)"
            ],
            value="Recommended (40% pruned)",
            label="Strategy"
        )

    with gr.Row():
        dust = gr.Slider(0, 5000, 546, step=1, label="Dust Threshold (sats)")
        dest = gr.Textbox(
            label="Change Address (optional)",
            placeholder="Leave blank → reuse first input"
        )
        fee_rate = gr.Slider(1, 300, 15, step=1, label="Fee Rate (sat/vB)")
        thank_you = gr.Slider(0, 5, 0.5, step=0.1, label="Thank-You / DAO Donation (%)")

    # Analyze button at top
    analyze_btn = gr.Button("1. ANALYZE & LOAD UTXOs", variant="primary")
    html_out = gr.HTML() 
    summary = gr.HTML()
    enriched_state = gr.State()
    # Generate button between summary and table

    
    with gr.Row(visible=False) as generate_row:
        gen_btn = gr.Button(
            "2. GENERATE NUCLEAR PSBT",
            variant="primary",
            visible=False
        )
    df = gr.DataFrame(
        headers=["PRUNE","TXID","vout (output index)","Value (sats)","Address", "Weight (vB)"],
        datatype=["bool","str","number","number","str", "number"],
        type="array",
        interactive=True,
        wrap=True,
        label="CHECK TO PRUNE • Uncheck to keep • Pre-checked UTXOs start at bottom of table",
    )



    # Start over button below table
    reset_btn = gr.Button("START OVER", variant="secondary")

    # --------------------------
    # BUTTON / TABLE WIRING — FINAL, BULLETPROOF
    # --------------------------

    # 1. ANALYZE — inline gr.State() for reliability
    analyze_btn.click(
        fn=analyze,
        inputs=[addr, strategy, dust, dest, fee_rate, thank_you],
        outputs=[html_out, df, enriched_state, summary, gen_btn, generate_row]
    )

    # 2. GENERATE PSBT — outputs only to html
    gen_btn.click(
            fn=generate_psbt,
        inputs=[dest, fee_rate, thank_you, df, enriched_state],
        outputs=[html_out, gen_btn, generate_row , summary]
        
    ).then(
        lambda _ : gr.update(interactive=False),
        outputs=df
    )
        

    reset_btn.click(
        fn=lambda: (
            "",     # html_out
            gr.update(   # df (table) FULL RESET
                value=[],
                row_count=1,
                col_count=6,
                datatype=["bool", "str", "number", "number", "str", "number"]),     # df  ✅ clears table
            [],     # enriched_state
            "",     # summary
            "",     # addr
            "Recommended (40% pruned)",  # strategy
            546,    # dust
            "",     # dest
            0.5,    # thank_you
            15,     # fee_rate
            gr.update(visible=False),   # gen_btn
            gr.update(visible=False)    # generate_row
        ),
        outputs=[
            html_out,
            df,             # ✅ table cleared normally
            enriched_state,
            summary,
            addr,
            strategy,
            dust,
            dest,
            thank_you,
            fee_rate,
            gen_btn,
            generate_row
        ]
    )


    df.change(lambda _ : gr.update(interactive=True), outputs=df)
    
    # 4. LIVE SUMMARY UPDATES — safe and fast
    df.change(
        fn=generate_summary,
        inputs=[df, enriched_state, fee_rate, thank_you],
        outputs=summary
    )
    fee_rate.change(
        fn=generate_summary,
        inputs=[df, enriched_state, fee_rate, thank_you],
        outputs=summary
    )
    thank_you.change(
        fn=generate_summary,
        inputs=[df, enriched_state, fee_rate, thank_you],
        outputs=summary
    )
    # 5. FOOTER
    gr.HTML(
    '<div style="'
    'margin: 60px auto 6px auto !important; '
    'padding: 16px 0 12px 0 !important; '
    'text-align: center; '
    'font-size: 0.94rem; '
    'color: #888; '
    'opacity: 0.96; '
    'max-width: 720px; '
    'line-height: 1.7;'
    '">'

    # VERSION
    '<strong style="color:#f7931a; font-size:1.08rem; letter-spacing:0.5px; '
    'text-shadow:0 0 12px rgba(247,147,26,0.65);">'
    'Ωmega Pruner v10.2.1 — NUCLEAR EDITION'
    '</strong><br>'

    # GITHUB LINK
    '<a href="https://github.com/babyblueviper1/Viper-Stack-Omega" '
    'target="_blank" rel="noopener" '
    'style="color: #f7931a; text-decoration: none; font-weight:600; '
    'text-shadow:0 0 10px rgba(247,147,26,0.55);">'
    'GitHub • Open Source • Apache 2.0'
    '</a><br><br>'

    # CUSTOM BUILDS SECTION — NEON Ω-GREEN
    '<div style="margin: 6px auto 10px auto; max-width:650px;">'
    '<a href="https://www.babyblueviper.com/p/omega-pruner-custom-builds" '
    'target="_blank" '
    'style="font-size:0.96rem; color:##dcdcdc; font-weight:700; '
    'text-decoration:none; letter-spacing:0.3px; '
    'text-shadow:0 0 18px rgba(0,255,0,0.85);">'
    '<strong>This build is engineered for speed and clarity.</strong><br>'
    'For extended capabilities or tailored integrations, custom versions can be commissioned.'
    '</a>'
    '</div><br>'

    # TAGLINE — NUCLEAR BLACK SHADOW, READABLE ON ANY DISPLAY
    '<span style="'
    'color:#0f0; '
    'font-size:0.88rem; '
    'font-weight:800; '
    'letter-spacing:0.6px; '
    'text-shadow: '
        '0 0 12px #0f0, '                    # neon glow
        '0 0 24px #0f0, '                    # outer glow
        '0 0  6px #000, '                    # sharp black drop
        '0  4px 10px #000, '                 # deep shadow
        '0  8px 20px #000000e6; '            # massive black halo
    '">'
    'Prune today. Win forever. • Ω'
    '</span>'
    '</div>',
    elem_id="omega_footer"
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False, debug=False)
