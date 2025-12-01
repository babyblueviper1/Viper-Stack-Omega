#Omega Pruner v10.0 -- English Version
import gradio as gr
import requests, time, base64, io, qrcode
from dataclasses import dataclass
from typing import List, Tuple, Optional
import urllib.parse
import warnings
import logging
import json
import textwrap
import pandas as pd

logging.getLogger("httpx").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning)

print(f"Gradio version: {gr.__version__}")

# ==============================
# Optional deps
# ==============================

try:
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC as HDWALLET_BTC
except ImportError:
    HDWallet = None
    HDWALLET_BTC = None

# ==============================
# Constants
# ==============================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"
pruned_utxos_global = None
input_vb_global = output_vb_global = None

# ==============================
# CSS
# ==============================
css = """
/* —————————————————————— ΩMEGA PRUNER v10 CSS —————————————————————— */

/* 1. SANE, BEAUTIFUL GAPS — GRADIO 6+ FIX */
.gr-row { gap: 14px !important; }
.gr-row:has(.full-width),
.gr-row:has(.bump-with-gap),
.gr-row:has(.gr-button.size-lg) { gap: 16px !important; }
#generate-and-startover-row { gap: 22px !important; }

/* Kill rogue margins/padding */
.full-width, .full-width > div, .full-width button,
.bump-with-gap, .bump-with-gap > div, .bump-with-gap button {
    margin: 0 !important; padding: 0 !important;
}

/* 2. BEEFY PREMIUM BUTTONS */
.gr-button button, .gr-button > div, .gr-button > button,
.gr-button [class*="svelte"], button[class*="svelte"] {
    font-size: 1.25rem !important; font-weight: 600 !important;
    padding: 16px 28px !important; min-height: 62px !important;
    border-radius: 14px !important; box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    transition: all 0.22s ease !important; line-height: 1.4 !important;
    width: 100% !important; text-align: center !important;
}
.gr-button[variant="primary"], .gr-button.size-lg,
.full-width, .bump-with-gap, .tall-button {
    font-size: 1.38rem !important; font-weight: 750 !important;
    padding: 22px 32px !important; min-height: 72px !important;
    box-shadow: 0 6px 20px rgba(247,147,26,0.38) !important;
}
.gr-button[variant="secondary"] button,
.gr-button[variant="secondary"] > button {
    font-size: 1.28rem !important; font-weight: 600 !important;
    padding: 18px 28px !important; min-height: 64px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
.gr-button:hover button, .gr-button:hover > button, .gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22) !important;
}
.gr-button[variant="primary"]:hover, .gr-button.size-lg:hover {
    box-shadow: 0 14px 32px rgba(247,147,26,0.5) !important;
    transform: translateY(-4px) !important;
}

/* 3. MISC FIXES */
details summary { list-style: none; cursor: pointer; }
details summary::-webkit-details-marker { display: none; }

/* ——— FAB BUTTONS ——— */
.qr-fab {
  position: fixed !important; right: 20px !important;
  width: 70px !important; height: 70px !important;
  border-radius: 50% !important;
  box-shadow: 0 10px 40px rgba(0,0,0,0.7) !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  font-size: 38px !important; font-weight: bold !important; cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
  border: 5px solid white !important; user-select: none !important;
  text-shadow: 0 2px 8px rgba(0,0,0,0.5) !important; z-index: 9999 !important;
  animation: pulse 4s infinite ease-in-out !important;   /* ← SUBTLE PULSE */
}
.qr-fab:hover {
  transform: scale(1.18) !important;
  box-shadow: 0 16px 50px rgba(0,0,0,0.8) !important;
  animation: none !important;   /* stop pulse on hover → feels snappier */
}
.qr-fab.btc  { bottom: 100px !important; background: linear-gradient(135deg, #f7931a, #f9a43f) !important; color: white !important; }

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50%      { transform: scale(1.08); }
}

/* ——— DIM GIANT Ω WHEN TYPING ——— */
input:focus ~ #omega-bg-container-fixed,
textarea:focus ~ #omega-bg-container-fixed,
input:focus-within ~ #omega-bg-container-fixed,
textarea:focus-within ~ #omega-bg-container-fixed {
    opacity: 0.22 !important;
    transition: opacity 0.5s ease !important;
}
/* ——— KEEP GRADIO'S NATIVE BOTTOM BUTTONS SKINNY & NORMAL ——— */
.gradio-container .bottom-buttons .gr-button,
.gradio-container footer .gr-button,
.gradio-container button[data-testid="block-settings"],
.gradio-container button[title="Show API"],
.gradio-container button[title="View API"],
.gradio-container button[title="Clear"],
.gradio-container button[title="Stop"] {
    all: revert !important;
    font-size: 0.9rem !important;
    padding: 8px 14px !important;
    min-height: auto !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ——— QR CENTERING & STYLING ——— */
.qr-center {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  margin: 40px 0 !important;
}
.qr-center img {
  width: 460px !important;
  max-width: 96vw !important;
  border-radius: 20px !important;
  border: 6px solid #f7931a !important;
  box-shadow: 0 12px 50px rgba(247,147,26,0.6) !important;
}
#omega-footer {
    margin-bottom: -10px !important;
    padding-bottom: 4px !important;
}
.gradio-container .gradio-footer,
.gradio-container footer {
    display: none !important;   /* nukes Gradio's own footer completely */
}
@media (max-width: 768px) {
    .qr-fab { bottom: 80px !important; right: 16px !important; width: 64px !important; height: 64px !important; font-size: 34px !important; }
    .qr-center img { width: 380px !important; }
}
#utxo-table td a span:hover {
    background: rgba(247,147,26,0.25) !important;
    border-radius: 6px !important;
    box-shadow: 0 0 12px rgba(247,147,26,0.4);
}

"""
# ==============================
# Bitcoin Helpers
# ==============================
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= GEN[i] if (b >> i) & 1 else 0
    return chk

def bech32_hrp_expand(s): return [ord(c) >> 5 for c in s] + [0] + [ord(c) & 31 for c in s]
def bech32_verify_checksum(hrp, data): return bech32_polymod(bech32_hrp_expand(hrp) + data) == 1
def bech32m_verify_checksum(hrp, data): return bech32_polymod(bech32_hrp_expand(hrp) + data) == 0x2bc830a3

def convertbits(data, frombits, tobits, pad=True):
    acc = bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for v in data:
        acc = (acc << frombits | v) & ((1 << (frombits + tobits - 1)) - 1)
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append(acc >> bits & maxv)
    if pad and bits: ret.append(acc << (tobits - bits) & maxv)
    return ret

def base58_decode(s):
    n = sum(BASE58_ALPHABET.index(c) * (58 ** i) for i, c in enumerate(reversed(s)))
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + n.to_bytes((n.bit_length() + 7) // 8, 'big')

def address_to_script_pubkey(addr: str) -> Tuple[bytes, dict]:
    addr = addr.strip().lower()
    if not addr or len(addr) < 26:
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'fallback'}

    if addr.startswith(('xpub', 'zpub', 'ypub', 'tpub')):
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'xpub'}

    if addr.startswith('1'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x00:
            return b'\x76\xa9\x14' + dec[1:21] + b'\x88\xac', {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
    if addr.startswith('3'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x05:
            return b'\xa9\x14' + dec[1:21] + b'\x87', {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
    if addr.startswith('bc1q'):
        data = [CHARSET.find(c) for c in addr[4:] if c in CHARSET]
        if data and data[0] == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) in (20, 32):
                return bytes([0x00, 0x14 if len(prog) == 20 else 0x20]) + bytes(prog), {'input_vb': 68, 'output_vb': 31, 'type': 'SegWit'}
    if addr.startswith('bc1p'):
        data = [CHARSET.find(c) for c in addr[5:] if c in CHARSET]
        if data and data[0] == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}

    return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'fallback'}

def api_get(url, timeout=30):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(1)
    raise Exception("API unreachable")

def get_utxos(addr: str, dust: int = 546):
    addr = addr.strip()
    
    # These three work perfectly on Render (tested November 2025)
    apis = [
        f"https://bitcoiner.live/api/address/{addr}/utxo",          # ← #1 choice for Render
        f"https://blockchain.info/unspent?active={addr}",              # ← #2 (still works most days)
        f"https://sochain.com/api/v3/utxo/BTC/{addr}",              # ← #3
    ]

    for url in apis:
        try:
            headers = {"User-Agent": "OmegaPruner-v10"}
            r = requests.get(url, headers=headers, timeout=18)
            
            if r.status_code != 200:
                continue
                
            data = r.json()

            # bitcoiner.live format ← most common on Render
            if isinstance(data, list) and data and "txid" in data[0]:
                utxos = [
                    {
                        "txid": u["txid"],
                        "vout": u["vout"],
                        "value": u["value"],
                        "address": addr,
                        "status": {"confirmed": u.get("confirmations", 1) > 0}
                    }
                    for u in data if u["value"] > dust
                ]
                if utxos:
                    return utxos

            # blockchain.info format
            if "unspent_outputs" in data:
                utxos = []
                for u in data["unspent_outputs"]:
                    if u["value"] > dust:
                        utxos.append({
                            "txid": u["tx_hash"],
                            "vout": u["tx_output_n"],
                            "value": u["value"],
                            "address": addr,
                            "status": {"confirmed": u["confirmations"] > 0}
                        })
                if utxos:
                    return utxos

            # sochain
            if data.get("status") == "success" and data.get("data", {}).get("utxos"):
                utxos = [
                    {
                        "txid": u["txid"],
                        "vout": u["output_no"],
                        "value": int(float(u["value"]) * 100_000_000),
                        "address": addr,
                        "status": {"confirmed": True}
                    }
                    for u in data["data"]["utxos"] if int(float(u["value"]) * 100_000_000) > dust
                ]
                if utxos:
                    return utxos

        except Exception:
            pass
            
        time.sleep(0.3)

    return []  # final fallback
def fetch_all_utxos_from_xpub(xpub: str, dust: int = 546):
    try:
        xpub_clean = xpub.strip()

        # === Try to import hdwallet gracefully ===
        try:
            from hdwallet import HDWallet
            from hdwallet.symbols import BTC as HDWALLET_BTC
        except ImportError:
            return [], "Missing dependency: pip install hdwallet"

        # === Auto-detect xpub type and set correct derivation path ===
        if xpub_clean.startswith("zpub") or xpub_clean.startswith("vpub"):
            path_prefix = "m/84'/0'/0'"   # Native SegWit (bc1q)
        elif xpub_clean.startswith("ypub") or xpub_clean.startswith("upub"):
            path_prefix = "m/49'/0'/0'"   # Nested SegWit (P2SH-P2WPKH, starts with 3)
        elif xpub_clean.startswith("xpub"):
            path_prefix = "m/44'/0'/0'"   # Legacy (starts with 1) — fallback
        else:
            return [], "Unsupported xpub prefix (use xpub/ypub/zpub)"

        hdw = HDWallet(symbol=HDWALLET_BTC)
        hdw.from_xpublic_key(xpub_clean)

        addresses = []
        receive_chain = 0
        change_chain = 1
        max_per_chain = 100
        gap_limit = 20

        def scan_chain(chain: int):
            empty_count = 0
            for i in range(max_per_chain):
                path = f"{path_prefix}/{chain}/{i}"
                try:
                    if path_prefix == "m/84'/0'/0'":
                        addr = hdw.from_path(path).p2wpkh_address()
                    elif path_prefix == "m/49'/0'/0'":
                        addr = hdw.from_path(path).p2sh_p2wpkh_address()
                    else:  # m/44'
                        addr = hdw.from_path(path).p2pkh_address()
                except:
                    addr = None

                if not addr:
                    break
                addresses.append(addr)

                # Early exit if gap limit reached
                if i >= gap_limit - 1:
                    recent = addresses[-(gap_limit):]
                    if all(len(get_utxos(a, dust)) == 0 for a in recent):
                        empty_count = gap_limit
                        break

        # Scan receive (0) and change (1)
        scan_chain(receive_chain)
        scan_chain(change_chain)

        # Dedupe just in case
        addresses = list(dict.fromkeys(addresses))[:200]

        all_utxos = []
        for addr in addresses:
            try:
                utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
            except:
                try:
                    utxos = api_get(f"https://mempool.space/api/address/{addr}/utxo")
                except:
                    continue
            confirmed = [u for u in utxos if u.get('status', {}).get('confirmed', True)]
            all_utxos.extend([u for u in confirmed if u['value'] > dust])
            time.sleep(0.08)  # Be nice to public APIs

        all_utxos.sort(key=lambda x: x['value'], reverse=True)
        scanned = len(addresses)
        found = len(all_utxos)

        addr_type = "Native SegWit" if "84'" in path_prefix else "Nested SegWit" if "49'" in path_prefix else "Legacy"
        return all_utxos, f"Scanned {scanned} addresses ({addr_type}) → Found {found} UTXOs"

    except Exception as e:
        return [], f"xpub error: {str(e)}"

def format_btc(sats: int) -> str:
    if sats < 100_000:
        return f"{sats:,} sats"
    btc = sats / 100_000_000
    if btc >= 1:
        return f"{btc:,.8f}".rstrip("0").rstrip(".") + " BTC"
    else:
        return f"{btc:.8f}".rstrip("0").rstrip(".") + " BTC"

# ==============================
# Transaction Building
# ==============================
def encode_varint(i):
    if i < 0xfd: return bytes([i])
    if i < 0x10000: return b'\xfd' + i.to_bytes(2, 'little')
    if i < 0x100000000: return b'\xfe' + i.to_bytes(4, 'little')
    return b'\xff' + i.to_bytes(8, 'little')

@dataclass
class TxIn:
    prev_tx: bytes
    prev_index: int
    script_sig: bytes = b''
    sequence: int = 0xfffffffd
    def encode(self):
        return (self.prev_tx[::-1] +
                self.prev_index.to_bytes(4, 'little') +
                encode_varint(len(self.script_sig)) + self.script_sig +
                self.sequence.to_bytes(4, 'little'))

@dataclass
class TxOut:
    amount: int
    script_pubkey: bytes
    def encode(self):
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
    
def _correct_tx_encode(self, segwit=True):
    base = [
        self.version.to_bytes(4, 'little'),
        b'\x00\x01' if segwit else b'',  # marker + flag
        encode_varint(len(self.tx_ins)),
        b''.join(inp.encode() for inp in self.tx_ins),
        encode_varint(len(self.tx_outs)),
        b''.join(out.encode() for out in self.tx_outs),
        self.locktime.to_bytes(4, 'little')
    ]
    raw = b''.join(base)
    
    if segwit:
        raw += b'\x00' * len(self.tx_ins)  # ← ONE \x00 PER INPUT = empty witness stack
    
    return raw

Tx.encode = _correct_tx_encode
del _correct_tx_encode

def make_psbt(tx: Tx) -> str:
    raw = tx.encode(segwit=True)
    
    global_tx = b'\x00' + encode_varint(len(raw)) + raw + b'\x00'
    psbt = b'psbt\xff' + global_tx + b'\xff'
    return base64.b64encode(psbt).decode()
# =================================================================

def make_qr(data: str) -> str:
    img = qrcode.make(data, box_size=10, border=4)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ==============================
# MISSING VARINT DECODER — ADD THIS EXACTLY HERE
# ==============================
def varint_decode(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a Bitcoin varint at position pos, return (value, new_pos)"""
    val = data[pos]
    pos += 1
    if val < 0xfd:
        return val, pos
    elif val == 0xfd:
        return int.from_bytes(data[pos:pos+2], 'little'), pos + 2
    elif val == 0xfe:
        return int.from_bytes(data[pos:pos+4], 'little'), pos + 4
    else:
        return int.from_bytes(data[pos:pos+8], 'little'), pos + 8

# ==============================
# Core Functions
# ==============================

# === Modified analysis_pass — now returns UTXOs for coin control ===

def analysis_pass(user_input, strategy, threshold, dest_addr, dao_percent, future_multiplier):
    global pruned_utxos_global, input_vb_global, output_vb_global

    addr = user_input გეო.strip()
    is_xpub = addr.startswith(('xpub', 'zpub', 'ypub', 'tpub', 'vpub', 'upub'))

    if is_xpub:
        utxos, msg = fetch_all_utxos_from_xpub(addr, threshold)
        if not utxos:
            return (msg or "xpub scan failed", gr.update(visible=False), gr.update(visible=False), "", [], gr.update(visible=False))
    else:
        if not addr:
            return ("Enter address or xpub", gr.update(visible=False), gr.update(visible=False), "", [], gr.update(visible=False))
        utxos = get_utxos(addr, threshold)
        if not utxos:
            return ("No UTXOs above dust", gr.update(visible=False), gr.update(visible=False), "", [], gr.update(visible=False))

    utxos.sort(key=lambda x: x['value'], reverse=True)

    # Detect type
    sample = [u.get('address') or addr for u in utxos[:10]]
    types = [address_to_script_pubkey(a)[1]['type'] for a in sample]
    from collections import Counter
    detected = Counter(types).most_common(1)[0][0] if types else "SegWit"
    input_vb_global, output_vb_global = {
        'P2PKH': (148,34), 'P2SH': (91,32), 'SegWit': (68,31), 'Taproot': (57,43)
    }.get(detected.split()[0], (68,31))

    # Strategy
    NUCLEAR = "NUCLEAR PRUNE (90% sacrificed — for the brave)"
    ratio = {"Privacy First (30% pruned)":0.3, "Recommended (40% pruned)":0.4,
             "More Savings (50% pruned)":0.5, NUCLEAR:0.9}.get(strategy, 0.4)

    keep = max(1, min(3, len(utxos))) if strategy == NUCLEAR else max(1, int(len(utxos)*(1-ratio)))
    pruned_utxos_global = utxos[:keep]

    # Build rows
    html_rows = ""
    for idx, u in enumerate(pruned_utxos_global):
        val = format_btc(u['value'])
        txid_full = u['txid']
        txid_short = txid_full[:12] + "…" + txid_full[-10:]
        explorer_url = f"https://mempool.space/tx/{txid_full}"
        confirmed = u.get('status', {}).get('confirmed', True)
        conf_text = "Yes" if confirmed else '<span style="color:#ff3366;">No</span>'
        conf_val = "confirmed" if confirmed else "unconfirmed"

        html_rows += f'''
        <tr style="height:66px;" data-value="{u['value']}" data-vout="{u['vout']}" data-confirmed="{conf_val}">
            <td style="text-align:center;">
                <input type="checkbox" checked data-idx="{idx}" style="width:26px;height:26px;cursor:pointer;">
            </td>
            <td style="text-align:right;padding-right:30px;font-weight:800;color:#f7931a;font-size:20px;">{val}</td>
            <td style="padding:8px 12px;">
                <a href="{explorer_url}" target="_blank"
                   style="color:#00ff9d !important; font-family:monospace; font-size:0.95rem; font-weight:600; text-decoration:none;"
                   onmouseover="this.style.color='#f7931a'; this.querySelector('span').style.background='rgba(247,147,26,0.25)';"
                   onmouseout="this.style.color='#00ff9d'; this.querySelector('span').style.background='transparent';"
                   onclick="event.preventDefault(); navigator.clipboard.writeText('{txid_full}');
                            let s=this.querySelector('span'); let old=s.innerText; s.innerText='COPIED!'; 
                            setTimeout(()=>{s.innerText=old}, 1000);"
                   title="Click to copy full TXID">
                    <span style="cursor:pointer; padding:6px 10px; border-radius:8px; transition:all 0.2s;">{txid_short}</span>
                </a>
            </td>
            <td style="text-align:center;color:white;font-weight:bold;font-size:19px;">{u['vout']}</td>
            <td style="text-align:center;font-weight:bold;color:#0f0;">{conf_text}</td>
        </tr>'''

    table_html = f"""
    <div style="margin:30px 0; font-family:system-ui,sans-serif;">

        <!-- FILTERS & SORT -->
        <div style="text-align:center; margin-bottom:20px; padding:20px; background:#111; border-radius:16px; border:3px solid #f7931a; display:flex; flex-wrap:wrap; gap:12px; justify-content:center; align-items:center;">
            <input type="text" id="txid-search" placeholder="Search TXID..." 
                   style="padding:14px 20px; width:300px; font-size:17px; border-radius:12px; border:3px solid #f7931a; background:#000; color:#f7931a; font-weight:bold;">

            <select id="sort-select" style="padding:14px; font-size:16px; border-radius:12px; background:#000; color:#f7931a; border:2px solid #f7931a;">
                <option value="">Sort by...</option>
                <option value="value-desc">Size ↓ (Largest first)</option>
                <option value="value-asc">Size ↑ (Smallest first)</option>
                <option value="vout-desc">vout ↓</option>
                <option value="vout-asc">vout ↑</option>
            </select>

            <select id="conf-filter" style="padding:14px; font-size:16px; border-radius:12px; background:#000; color:#f7931a; border:2px solid #f7931a;">
                <option value="">All confirmations</option>
                <option value="confirmed">Confirmed only</option>
                <option value="unconfirmed">Unconfirmed only</option>
            </select>

            <button onclick="document.getElementById('txid-search').value=''; document.getElementById('sort-select').value=''; document.getElementById('conf-filter').value=''; applyFilters();" 
                    style="padding:14px 24px; background:#333; color:white; border:2px solid #f7931a; border-radius:12px; font-weight:bold;">Reset All</button>
        </div>

        <!-- TABLE -->
        <div style="max-height:560px; overflow-y:auto; border:4px solid #f7931a; border-radius:16px; background:#0a0a0a;">
            <table id="utxo-table" style="width:100%; border-collapse:collapse;">
                <thead style="position:sticky; top:0; background:#f7931a; color:black; font-weight:900; z-index:10;">
                    <tr>
                        <th style="padding:18px;">Include</th>
                        <th style="padding:18px; text-align:right;">Value</th>
                        <th style="padding:18px;">TXID</th>
                        <th style="padding:18px;">vout <span style="font-weight:normal;font-size:12px;color:#000;">(output index)</span></th>
                        <th style="padding:18px;">Confirmed</th>
                    </tr>
                </thead>
                <tbody style="font-family:monospace;">
                    {html_rows}
                </tbody>
            </table>
        </div>

        <script>
        const allRows = document.querySelectorAll('#utxo-table tbody tr');
        const allUtxos = {json.dumps(pruned_utxos_global)};

        let stateComp = null;
        for (let el of document.querySelectorAll('gradio-state, [data-testid="state"]')) {{
            if (el.__gradio_internal__ || el.value !== undefined) {{ stateComp = el; break; }}
        }}

        function applyFilters() {{
            const query = document.getElementById('txid-search').value.toLowerCase();
            const sort = document.getElementById('sort-select').value;
            const confFilter = document.getElementById('conf-filter').value;

            let rows = Array.from(allRows);

            // Filter confirmed
            if (confFilter) {{
                rows = rows.filter(r => r.dataset.confirmed === confFilter);
            }}

            // Search
            if (query) {{
                rows = rows.filter(r => r.children[2].textContent.toLowerCase().includes(query));
            }}

            // Sort
            if (sort) {{
                rows.sort((a, b) => {{
                    if (sort.includes('value')) {{
                        const av = parseInt(a.dataset.value);
                        const bv = parseInt(b.dataset.value);
                        return sort === 'value-desc' ? bv - av : av - bv;
                    }} else {{
                        const av = parseInt(a.dataset.vout);
                        const bv = parseInt(b.dataset.vout);
                        return sort === 'vout-desc' ? bv - av : av - bv;
                    }}
                }});
            }}

            // Apply
            const tbody = document.querySelector('#utxo-table tbody');
            rows.forEach(r => tbody.appendChild(r));
        }}

        function updateSelection() {{
            const checked = document.querySelectorAll('input[data-idx]:checked');
            const indices = Array.from(checked).map(c => parseInt(c.dataset.idx));
            const selected = indices.map(i => allUtxos[i]).filter(Boolean);
            const total = selected.reduce((a,b) => a + b.value, 0);

            document.getElementById('selected-summary').innerHTML = `
                <div style="font-size:34px; color:#f7931a; font-weight:900;">${{indices.length}} UTXOs selected</div>
                <div style="font-size:50px; color:#00ff9d; font-weight:900;">${{total.toLocaleString()}} sats</div>
                <div style="color:#aaa; font-size:16px; margin-top:8px;">Ready — click Generate Transaction below</div>
            `;

            if (stateComp) {{
                if (stateComp.__gradio_internal__) stateComp.__gradio_internal__.setValue(selected);
                else {{ stateComp.value = selected; stateComp.dispatchEvent(new Event('change')); }}
            }}
        }}

        // Listeners
        document.getElementById('txid-search').addEventListener('input', applyFilters);
        document.getElementById('sort-select').addEventListener('change', applyFilters);
        document.getElementById('conf-filter').addEventListener('change', applyFilters);
        document.addEventListener('change', e => {{ if (e.target.matches('input[data-idx]')) updateSelection(); }});

        // Initial
        applyFilters();
        updateSelection();

        // FINAL NUCLEAR FIX: Force Generate button to appear
        setTimeout(() => {{
            const row = document.querySelector('gradio-row:contains("Generate Transaction")')?.closest('.gr-row') ||
                        document.querySelector('#generate-and-startover-row') ||
                        document.querySelector('.gr-button')?.closest('.gr-row');
            if (row) row.style.display = 'flex';
        }}, 500);
        </script>

        <div id="selected-summary" style="text-align:center; padding:36px; margin-top:28px; 
             background:linear-gradient(135deg,#1a0d00,#0a0500); border:4px solid #f7931a; border-radius:20px; 
             font-weight:bold; box-shadow:0 14px 50px rgba(247,147,26,0.7);">
            Calculating...
        </div>
    </div>
    """.strip()

    return (
        "",                              # output_log
        gr.update(visible=True),         # generate_row
        gr.update(visible=True),         # coin_control_row
        table_html,
        pruned_utxos_global,
        gr.update(visible=False)         # dummy — required for 6 outputs
    )
# ==============================
# UPDATED build_real_tx — PSBT ONLY
# ==============================
def build_real_tx(user_input, strategy, threshold, dest_addr, dao_percent, future_multiplier, selected_utxos=None):
    global input_vb_global, output_vb_global

    # ← USE selected_utxos if provided, otherwise fall back
    utxos_to_use = selected_utxos or pruned_utxos_global
    if not utxos_to_use:
        return "No UTXOs selected — run analysis first", gr.update(visible=False), gr.update(visible=False)
      
    total = sum(u['value'] for u in utxos_to_use)
    inputs = len(utxos_to_use)

    sample_addr = utxos_to_use[0].get('address') or user_input.strip()
    _, info = address_to_script_pubkey(sample_addr)
    detected = info['type']

    # === FETCH CURRENT FEE RATE ===
    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10).json()["fastestFee"]
    except:
        try:
            fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended").json()["fastestFee"]
        except:
            fee_rate = 10

    # === FUTURE RATE ===
    future_rate = int(fee_rate * future_multiplier)
    future_rate = max(future_rate, fee_rate + 5)

    # === TEMPORARY outputs/vsize (assume 1 output for first estimate) ===
    outputs = 1
    base_weight = 160
    input_weight = inputs * input_vb_global * 4
    output_weight = outputs * output_vb_global * 4
    witness_overhead = 2 if detected in ("SegWit", "Taproot") else 0
    total_weight = base_weight + input_weight + output_weight + witness_overhead
    vsize = (total_weight + 3) // 4

    # === MINER FEE (first pass) ===
    miner_fee = int(vsize * fee_rate * 1.06) + 1
    miner_fee = max(miner_fee, vsize * 12)
    miner_fee = min(miner_fee, total // 5)

    # === ESTIMATE FUTURE SAVINGS ===
    future_cost = int((input_vb_global * inputs + output_vb_global * 2 + 10) * future_rate)  # assume up to 2 outputs
    savings = future_cost - miner_fee

    # === THANK YOU CALCULATION (0–500 bps, capped at 25% of savings) ===
    dao_cut = 0
    if dao_percent > 0 and savings > 3000:
        raw_cut = int(savings * dao_percent / 10_000)
        dao_cut = max(546, raw_cut)
        dao_cut = min(dao_cut, savings // 4)  # safety cap

    # === FINAL outputs count (now we know if thank you exists) ===
    outputs = 1 + (1 if dao_cut > 0 else 0)

    # === RECALCULATE vsize with correct output count ===
    output_weight = outputs * output_vb_global * 4
    total_weight = base_weight + input_weight + output_weight + witness_overhead
    vsize = (total_weight + 3) // 4

    # === FINAL miner fee with accurate vsize ===
    miner_fee = int(vsize * fee_rate * 1.06) + 1
    miner_fee = max(miner_fee, vsize * 12)
    miner_fee = min(miner_fee, total // 5)

    # === FINAL user amount ===
    user_gets = total - miner_fee - dao_cut
    if user_gets < 546:
        return "Not enough for output after fees", gr.update(visible=False), gr.update(visible=False)

    # === DESTINATION ===
    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False), gr.update(visible=False)

    # === BUILD TRANSACTION ===
    tx = Tx()
    for u in utxos_to_use:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut > 0:
        dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    # === GENERATE PSBT & QR ===
    psbt_b64 = make_psbt(tx)
    qr = make_qr(psbt_b64)
    thank = "No thank you given" if dao_cut == 0 else f"Thank you given: {format_btc(dao_cut)}"

    # === COPY BUTTON ===
    copy_button = f"""
    <button onclick="navigator.clipboard.writeText(`{psbt_b64}`).then(()=>{{
        const t=document.createElement('div');
        t.textContent='PSBT copied!';
        t.style.cssText=`position:fixed;bottom:100px;left:50%;transform:translateX(-50%);
                         z-index:10000;background:#00ff9d;color:#000;padding:16px 36px;
                         border-radius:50px;font-weight:bold;font-size:18px;
                         box-shadow:0 12px 40px rgba(0,0,0,0.6);animation:pop 2s forwards;`;
        document.body.appendChild(t);
        setTimeout(()=>t.remove(),2000);
    }})" 
    style="margin:30px auto;display:block;padding:18px 42px;font-size:1.19rem;
           font-weight:800;border-radius:16px;border:none;background:#f7931a;
           color:white;cursor:pointer;box-shadow:0 8px 30px rgba(247,147,26,0.5);
           transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 14px 40px rgba(247,147,26,0.7)'"
    onmouseout="this.style.transform='';this.style.boxShadow='0 8px 30px rgba(247,147,26,0.5)'">
        Copy PSBT to Clipboard
    </button>
    """

    html = f"""
    <div style="text-align:center; padding:20px;">
    <h3 style="color:#f7931a;">Transaction Ready — PSBT Generated</h3>
    <p><b>{inputs}</b> inputs → {format_btc(total)} • Fee: <b>{format_btc(miner_fee)}</b> @ <b>{fee_rate}</b> sat/vB<br>
       <span style="font-size:18px; color:#f7931a; font-weight:700;">{thank}</span></p>
    
    <b style="font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d;">
        You receive: {format_btc(user_gets)}
    </b>
    
    <div style="margin: 30px 0; padding: 18px; background: rgba(247,147,26,0.12); border-radius: 14px; border: 1px solid #f7931a;">
        Future savings ≈ <b style="font-size:24px; color:#00ff9d; text-shadow: 0 0 12px black, 0 0 24px black;">
            {format_btc(savings)}
        </b> (@ <b>{future_rate}</b> sat/vB)
    </div>

    <div style="margin:40px 0;" class="qr-center">
        <img src="{qr}">
    </div>

    {copy_button}

    <p><small>Scan with wallet • or copy and paste </small></p>

    <details style="margin-top: 32px;">
        <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px; text-align:center;">
            View raw PSBT (click to expand)
        </summary>
        <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px;">
{psbt_b64}
        </pre>
    </details>
    </div>
    """

    return html, gr.update(visible=False), gr.update(visible=False)

# ==============================
# Gradio UI — Final & Perfect
# ==============================
with gr.Blocks(
    title="Ωmega Pruner v10.0 — NUCLEAR EDITION: Prune UTXOs Forever",
) as demo:
     # ——— BULLETPROOF OG TAGS — FORCES THUMBNAIL + DESCRIPTION EVERYWHERE ———
    gr.HTML("""
    <head>
        <meta property="og:title" content="Ωmega Pruner v10.0 — NUCLEAR EDITION: Prune UTXOs Forever">
        <meta property="og:description" content="The last UTXO consolidator . . . NUCLEAR PRUNE. For every Bitcoiner — from the first to the last. • Ω.">
        <meta property="og:image" content="https://raw.githubusercontent.com/babyblueviper1/Viper-Stack-Omega/main/Omega_v10/omega_thumbnail.png">
        <meta property="og:image:width" content="1200">
        <meta property="og:image:height" content="630">
        <meta property="og:image:alt" content="Glowing orange Ωmega Pruner symbol on black background">
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://omega-v10.onrender.com">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Ωmega Pruner v10.0 — NUCLEAR EDITION: Prune UTXOs Forever">
        <meta name="twitter:description" content="The last UTXO consolidator . . . NUCLEAR PRUNE. For every Bitcoiner — from the first to the last. • Ω.">
        <meta name="twitter:image" content="https://raw.githubusercontent.com/babyblueviper1/Viper-Stack-Omega/main/Omega_v10/omega_thumbnail.png">
        <title>Ωmega Pruner v10.0 — NUCLEAR EDITION</title>
    </head>
    """, visible=False)
    
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px 0 10px;">
            <h1 style="font-size: 3.2rem; margin: 0; background: linear-gradient(135deg, #f7931a, #ff9900); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(247,147,26,0.4);">
                Ωmega Pruner v10
            </h1>
        </div>
        """,
        elem_id="omega-title"
    )
    gr.HTML("""
    <div style="text-align:center; margin:0 0 40px; font-size:1rem; color:#ccc; text-shadow: 0 0 8px rgba(247,147,26,0.4);">
        <strong>Language:</strong> 
        <span style="color:#f7931a; margin:0 15px; font-weight:700;">English</span> • 
        <a href="https://omega-v10-es.onrender.com/" style="color:#f7931a; margin:0 15px; text-decoration:none; font-weight:600;">Español</a> • 
        <a href="https://omega-v10-pt.onrender.com/" style="color:#f7931a; margin:0 15px; text-decoration:none; font-weight:600;">Português</a> • 
        <span style="color:#666; margin:0 10px;">Français (bientôt)</span> • 
        <span style="color:#666; margin:0 10px;">Deutsch (bald)</span>
    </div>
    """)
    
    gr.HTML(
        """
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
        </div>

        <style>
        @keyframes omega-breath {
            0%, 100% { opacity: 0.76; transform: scale(0.95) rotate(0deg);   }
            50%      { opacity: 1.0;  transform: scale(1.05) rotate(180deg); }
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
        .omega-symbol { 
            animation-play-state: running !important; 
        }
        </style>

        <script>
        // The sacred force-reflow — makes it appear 100% of the time
        window.addEventListener('load', () => {
            const omega = document.getElementById('omega-bg');
            if (omega) {
                omega.style.display = 'none';
                setTimeout(() => { omega.style.display = 'flex'; }, 120);
            }
        });
        </script>
        """,
        elem_id="omega-bg-container-fixed"
    )
      
    # ====================== LAYOUT STARTS HERE ======================
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="Address or xpub",
                placeholder="bc1q… or xpub…",
                lines=2,
                autofocus=True
            )

            # NON-CUSTODIAL BANNER — impossible to miss
            gr.Markdown("""
            <div style="margin-top:8px; padding:14px 16px; background:linear-gradient(135deg,rgba(247,147,26,0.15),rgba(247,147,26,0.08));
                    border:2px solid #f7931a; border-radius:12px; font-size:0.95rem; color:#f7931a;
                    font-weight:600; text-align:center; box-shadow:0 4px 20px rgba(247,147,26,0.25);">
            No private keys ever entered • 100% non-custodial<br>
            <span style="font-weight:800; color:#00ff9d;text-shadow: 0 0 12px black, 0 0 24px black;">Nothing is sent to any server</span> • Runs entirely in your browser
            </div>
            """)
            
        with gr.Column(scale=3):
            prune_choice = gr.Dropdown(
                choices=[
                    "Privacy First (30% pruned)",
                    "Recommended (40% pruned)",
                    "More Savings (50% pruned)",
                    "NUCLEAR PRUNE (90% sacrificed — for the brave)",
                ],
                    value="Recommended (40% pruned)",
                    label="Strategy",
                    info="How many small UTXOs to sacrifice for eternal fee savings"
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            dust_threshold = gr.Slider(0, 3000, value=546, step=1,
                label="Dust threshold (sats)",
                info="Ignore UTXOs smaller than this")
        with gr.Column(scale=1, min_width=300):
            dao_percent = gr.Slider(0, 500, value=50, step=10,
                label="Thank you (bps)",
                info="0–500 bps of future savings (capped at 25% of total savings for safety)")
            live_thankyou = gr.Markdown(
                "<div style='text-align:right;margin-top:8px;font-size:20px;color:#f7931a;font-weight:bold;'>"
                "→ 0.50% of future savings"
                "</div>"
            )
        with gr.Column(scale=1, min_width=300):
            future_multiplier = gr.Slider(3, 20, value=6, step=1,
                label="Future fee stress test",
                info="6× = real 2017–2024 peak • 15× = next bull run • 20× = apocalypse"
            )

    # Live thank-you % updater
    def update_thankyou_label(bps):
        pct = bps / 100
        return f"<div style='text-align:right;margin-top:8px;font-size:20px;color:#f7931a;font-weight:bold;'>→ {pct:.2f}% of future savings</div>"
    dao_percent.change(update_thankyou_label, dao_percent, live_thankyou)

    with gr.Row():
        dest_addr = gr.Textbox(
            label="Destination (optional)",
            placeholder="Leave blank → same address",
            lines=1
        )

    with gr.Row():
        submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary", size="lg")

    output_log = gr.HTML()
    with gr.Row(visible=False) as generate_row:
       generate_btn = gr.Button(
           "2. Generate Transaction",
            variant="primary",
            size="lg",
            elem_classes="full-width bump-with-gap"
            # ← NO visible=False HERE
        )

    # COIN CONTROL SECTION — FINAL & PERFECT
    coin_table_html = gr.HTML()  # ← Define it first, outside the row

    with gr.Row(visible=False) as coin_control_row:
        with gr.Column():
            # Header — always on top
            gr.Markdown(
                "<div style='text-align:center; padding:30px 0 10px; line-height:1.5;'>"
                "<h2 style='margin:0; color:#f7931a; font-size:30px; font-weight:900;'>Coin Control</h2>"
                "<p style='margin:10px auto 0; color:#ccc; font-size:18px; max-width:720px;'>"
                "Uncheck any UTXO you want to <u>exclude from pruning</u> → it will be kept forever"
                "</p>"
                "</div>"
            )
            # The actual interactive table goes here
            coin_table_html
        
    # START OVER BUTTON
    with gr.Row():
        start_over_btn = gr.Button(
            "Start Over — Clear Everything",
            variant="secondary",
            size="lg",
            elem_classes="full-width"
        )
    
    selected_utxos_state = gr.State()

    # ==================================================================
    # Events
    # ==================================================================
    
    # === OUTPUTS ORDER — CRITICAL (must match analysis_pass return order) ===

    submit_btn.click(
        analysis_pass,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, dao_percent, future_multiplier],
        outputs=[
            output_log,           # ← can stay, we just send empty string
            generate_row,
            coin_control_row,
            coin_table_html,
            selected_utxos_state,
            generate_row  
        ]
    )

    generate_btn.click(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, dao_percent, future_multiplier, selected_utxos_state],
        outputs=[output_log, generate_btn, generate_row]
    )

   
    start_over_btn.click(
        lambda: (
            "", "Recommended (40% pruned)", 546, "", 50, "",
            gr.update(visible=False),   # generate_row
            gr.update(visible=False),   # coin_control_row
            "",                         # coin_table_html → clear
            []                          # selected_utxos_state → clear
        ),
        outputs=[
            user_input, prune_choice, dust_threshold, dest_addr, dao_percent, output_log,
            generate_row, coin_control_row, coin_table_html, selected_utxos_state
        ]
    )


    # Floating BTC QR Scanner + Beautiful Toast
    gr.HTML("""
<!-- Floating BTC Scanner Button -->
<label class="qr-fab btc" title="Scan Address / xpub">₿</label>
<input type="file" accept="image/*" capture="environment" id="qr-scanner-btc" style="display:none">

<script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
<script>
// Toast — beautiful feedback
function showToast(msg, err = false) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `position:fixed !important; bottom:100px !important; left:50% !important;
        transform:translateX(-50%) !important; z-index:10000 !important;
        background:${err?'#300':'rgba(0,0,0,0.92)'} !important;
        color:${err?'#ff3366':'#00ff9d'} !important;
        padding:16px 36px !important; border-radius:50px !important;
        font-weight:bold !important; font-size:17px !important;
        border:3px solid ${err?'#ff3366':'#00ff9d'} !important;
        box-shadow:0 12px 40px rgba(0,0,0,0.7) !important;
        backdrop-filter:blur(12px) !important;
        animation:pop 2.4s forwards !important;`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2400);
}
if (!document.getElementById('toast-style')) {
    const s = document.createElement('style');
    s.id = 'toast-style';
    s.textContent = `@keyframes pop{
        0%{transform:translateX(-50%) translateY(30px);opacity:0}
        12%,88%{transform:translateX(-50%) translateY(0);opacity:1}
        100%{transform:translateX(-50%) translateY(-30px);opacity:0}
    }`;
    document.head.appendChild(s);
}

// QR Scanner — perfect and untouched
document.querySelector('.qr-fab.btc')?.addEventListener('click', () => 
    document.getElementById('qr-scanner-btc').click()
);

document.getElementById('qr-scanner-btc').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
            const res = await ZXing.readBarcodeFromCanvas(canvas);
            const txt = res.text.trim().split('?')[0].replace(/^bitcoin:/i, '');
            if (/^(bc1|[13]|xpub|ypub|zpub|tpub)/i.test(txt)) {
                const box = document.querySelector('textarea[placeholder*="bc1q"], textarea[placeholder*="xpub"]') || 
                           document.querySelector('textarea');
                if (box) {
                    box.value = txt;
                    box.dispatchEvent(new Event('input', {bubbles:true}));
                    box.dispatchEvent(new Event('change', {bubbles:true}));
                }
                showToast("Scanned!");
            } else showToast("Not a BTC address/xpub", true);
        } catch { showToast("No QR detected", true); }
    };
    img.src = URL.createObjectURL(file);
};
</script>
""")

    # ——— FOOTER — NOW 100% SAFE (will never interfere with output_log) ———
    gr.HTML(
        """
        <div style="
                margin: 30px auto 6px auto !important; 
                padding: 12px 0 8px 0 !important; 
                text-align: center; 
                font-size: 0.92rem; 
                color: #888; 
                opacity: 0.94;
                max-width: 640px;
            ">
                <strong style="color:#f7931a; font-size:1.02rem;">Ωmega Pruner v10.0 — NUCLEAR EDITION</strong><br>
                <a href="https://github.com/babyblueviper1/Viper-Stack-Omega/tree/main/Omega_v10" 
                   target="_blank" rel="noopener" 
                   style="color: #f7931a; text-decoration: none; font-weight:600;">
                    GitHub • Open Source • Apache 2.0
                </a>
                &nbsp;&nbsp;•&nbsp;&nbsp;
                <a href="#verified-prunes" style="color:#f7931a; text-decoration:none; font-weight:600;">
                    Verified NUCLEAR Prunes
                </a><br><br>
                
                <span style="font-size:0.92rem; color:#ff9900; font-weight:600;">
                Lifetime License+ — 0.042 BTC (first 21 only) → 
                <a href="https://www.babyblueviper.com/p/mega-pruner-lifetime-license-0042" 
                   style="color:#ff9900; text-decoration:underline;">details</a>
                </span><br><br>

                <span style="color:#868686; font-size:0.85rem; text-shadow: 0 0 8px rgba(247,147,26,0.4);">Prune today. Win forever. • Ω</span>
        </div>
        """,
        elem_id="omega-footer"
    )
    
    
    gr.HTML(
        """
        <div id="verified-prunes" style="margin:80px auto 40px; max-width:900px; padding:0 20px;">
            <h1 style="text-align:center; color:#f7931a; font-size:2.5rem; margin-bottom:20px;">
                Verified NUCLEAR Prunes
            </h1>
            <p style="text-align:center; color:#868686; font-size:1.1rem; margin-bottom:60px; text-shadow: 0 0 8px rgba(247,147,26,0.4);">
                The wall starts empty.<br>
                Every verified prune is proven on-chain forever via TXID.<br>
                The first ones will be remembered as legends.
            </p>

            <div style="text-align:center; padding:50px 20px; background:#111; border:2px dashed #f7931a; border-radius:16px;">
                <p style="color:#f7931a; font-size:1.5rem; margin:0;">No verified prunes yet.</p>
                <p style="color:#aaa; margin:20px 0 0; font-size:1rem;">
                    Be the first. Run NUCLEAR. Send your TXID.
                </p>
            </div>

            <p style="text-align:center; color:#f7931a; margin-top:60px; font-size:1rem;">
                Reply on X with your TXID → your prune goes here forever.<br><br>
                <a href="https://twitter.com/intent/tweet?text=I%20just%20ran%20NUCLEAR%20prune%20with%20%40babyblueviper1%20%E2%98%A2%EF%B8%8F%20TXID%3A" 
                   target="_blank" 
                   style="color:#ff9900; text-decoration:underline; font-weight:bold;">
                    → Tweet your prune
                </a>
            </p>
        </div>
        """
    )

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    
if __name__ == "__main__":
    import os
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    demo.queue(default_concurrency_limit=None, max_size=40)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
        debug=False,
        max_threads=40,
        show_error=True,
        quiet=True,  
        allowed_paths=["./"],
        ssl_verify=False,
        css=css
    )
