# app.py — Omega Pruner v10.0 — Infinite Edition
import gradio as gr
import requests, time, base64, io, qrcode
from dataclasses import dataclass
from typing import List, Tuple, Optional
import urllib.parse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Gradio version: {gr.__version__}")

# ==============================
# Optional deps
# ==============================
try:
    from bolt11 import decode as bolt11_decode
except ImportError:
    bolt11_decode = None

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
/* Full-width buttons */
.full-width, .full-width > button { width: 100% !important; margin: 20px 0 !important; }
.tall-button { height: 100% !important; }
.tall-button > button { height: 100% !important; padding: 20px !important; font-size: 18px !important; }
details summary { list-style: none; cursor: pointer; }
details summary::-webkit-details-marker { display: none; }

/* Floating QR Buttons */
.qr-fab {
  position: fixed !important; right: 20px; z-index: 9999; width: 70px; height: 70px;
  border-radius: 50%; box-shadow: 0 10px 40px rgba(0,0,0,0.7); display: flex;
  align-items: center; justify-content: center; font-size: 38px; cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1); border: 5px solid white;
  font-weight: bold; user-select: none; text-shadow: 0 2px 8px rgba(0,0,0,0.5);
}
.qr-fab:hover { transform: scale(1.18); box-shadow: 0 16px 50px rgba(0,0,0,0.8); }
.qr-fab.btc { bottom: 100px; background: linear-gradient(135deg, #f7931a, #f9a43f); color: white; }
.qr-fab.ln  { bottom: 20px;  background: linear-gradient(135deg, #00ff9d, #33ffc7); color: #000; font-size: 42px; }

/* === RBF SECTION == */
.rbf-copy-btn,
.rbf-clear-btn,
.rbf-bump-btn {
    width: 100% !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

.rbf-clear-btn {
    margin-top: 16px !important;     /* ← THIS IS THE GAP */
}

.rbf-bump-btn {
    margin-top: 24px !important;     /* ← BIG GAP BEFORE BUMP */
}

/* Desktop: side-by-side if you want (optional) */
@media (min-width: 769px) {
    .rbf-copy-btn,
    .rbf-clear-btn {
        display: inline-block !important;
        width: 48% !important;
        margin-right: 4% !important;
    }
    .rbf-clear-btn {
        margin-right: 0 !important;
        margin-top: 0 !important;
    }
    .rbf-bump-btn {
        margin-top: 32px !important;
    }
}
.qr-center {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 40px 0 !important;
    padding: 10px 0 !important;
}

/* Extra safety — force image centering */
.qr-center img {
    max-width: 96vw !important;
    width: 460px !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
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

def get_utxos(addr, dust=546):
    try:
        api_get("https://blockstream.info/api/blocks/tip/height")
        utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
    except:
        api_get("https://mempool.space/api/blocks/tip/height")
        utxos = api_get(f"https://mempool.space/api/address/{addr}/utxo")
    confirmed = [u for u in utxos if u.get('status', {}).get('confirmed', True)]
    return [u for u in confirmed if u['value'] > dust]

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
# Transaction Building & RBF (fixed)
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
    parts = [
        self.version.to_bytes(4, 'little'),
        encode_varint(len(self.tx_ins)),
        b''.join(inp.encode() for inp in self.tx_ins),
        encode_varint(len(self.tx_outs)),
        b''.join(out.encode() for out in self.tx_outs),
        self.locktime.to_bytes(4, 'little')
    ]
    if segwit:
        parts.insert(1, b'\x00\x01')
        if not any(inp.script_sig for inp in self.tx_ins):  # unsigned
            parts.append(b'\x00\x00\x00\x00')  # witness placeholder
    return b''.join(parts)

Tx.encode = _correct_tx_encode

def make_psbt(tx: Tx) -> str:
    raw = tx.encode(segwit=True)
    if raw.endswith(b'\x00\x00\x00\x00'):
        raw = raw[:-4]
    psbt = b'psbt\xff' + b'\x00' + encode_varint(len(raw)) + raw + b'\x00'
    return base64.b64encode(psbt).decode()

def varint_decode(data: bytes, pos: int) -> tuple[int, int]:
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
def make_qr(data: str) -> str:
    img = qrcode.make(data, box_size=10, border=4)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ==============================
# BULLETPROOF RBF BUMP (Fixed!)
# ==============================
def rbf_bump(raw_hex: str):
    raw_hex = raw_hex.strip()
    if not raw_hex:
        return raw_hex, "<div style='color:#f7931a;'>Paste a raw transaction hex first</div>"

    try:
        data = bytes.fromhex(raw_hex)
    except:
        return raw_hex, "Invalid hex"

    pos = 0
    version = int.from_bytes(data[pos:pos+4], 'little')
    pos += 4

    is_segwit = data[pos:pos+2] == b'\x00\x01'
    if is_segwit:
        pos += 2

    vin_len, pos = varint_decode(data, pos)
    inputs = []
    for _ in range(vin_len):
        txid = data[pos:pos+32][::-1].hex()
        vout = int.from_bytes(data[pos+32:pos+36], 'little')
        pos += 36
        slen, pos = varint_decode(data, pos)
        pos += slen
        seq = int.from_bytes(data[pos:pos+4], 'little')
        inputs.append({'sequence': seq, 'seq_pos': pos-4})
        pos += 4

    vout_len, pos = varint_decode(data, pos)
    if vout_len == 0:
        return raw_hex, "No outputs — cannot bump"

    change_pos = pos
    change_amount = int.from_bytes(data[pos:pos+8], 'little')
    pos += 8
    slen, pos = varint_decode(data, pos)
    pos += slen

    for _ in range(1, vout_len):
        pos += 8
        slen, pos = varint_decode(data, pos)
        pos += slen

    if is_segwit:
        for _ in range(vin_len):
            items, pos = varint_decode(data, pos)
            for _ in range(items):
                ilen, pos = varint_decode(data, pos)
                pos += ilen

    vsize = (len(data) + 3) // 4
    extra = int(vsize * 50)
    if change_amount <= extra + 546:
        return raw_hex, f"<div style='color:#ff3333; padding:16px; background:#300; border-radius:12px;'><b>Cannot bump</b><br>Not enough change output</div>"

    tx = bytearray(data)
    new_amount = change_amount - extra
    tx[change_pos:change_pos+8] = new_amount.to_bytes(8, 'little')

    # Enable RBF if all sequences are max
    if all(i['sequence'] == 0xffffffff for i in inputs):
        for i in inputs:
            tx[i['seq_pos']:i['seq_pos']+4] = (0xfffffffd).to_bytes(4, 'little')

    bumped = tx.hex()

    try:
        rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=5).json()["fastestFee"]
    except:
        rate = "??"

    info = f"""
    <div style="background:#00ff9d20; padding:18px; border-radius:14px; border:3px solid #00ff9d; text-align:center; margin:20px 0;">
        <b style="font-size:24px; color:#00ff9d;">BUMP SUCCESSFUL</b><br><br>
        +50 sat/vB → +{extra:,} sats to miners<br>
        Change: {format_btc(change_amount)} → <b>{format_btc(new_amount)}</b><br>
        <small>Current fastest: ~{rate} sat/vB</small>
    </div>
    """
    return bumped, info
# ==============================
# Core Functions
# ==============================
def analysis_pass(user_input, strategy, threshold, dest_addr, selfish_mode, dao_percent, dao_addr):
    global pruned_utxos_global, input_vb_global, output_vb_global

    addr = user_input.strip()
    is_xpub = addr.startswith(('xpub', 'zpub', 'ypub', 'tpub'))

    if is_xpub:
        utxos, msg = fetch_all_utxos_from_xpub(addr, threshold)
        if not utxos:
            return msg or "Failed to scan xpub", gr.update(visible=False)
    else:
        if not addr:
            return "Enter address or xpub", gr.update(visible=False)
        utxos = get_utxos(addr, threshold)
        if not utxos:
            return "No UTXOs above dust threshold", gr.update(visible=False)

    utxos.sort(key=lambda x: x['value'], reverse=True)

    # Detect script type
    sample = [u.get('address') or addr for u in utxos[:10]]
    types = [address_to_script_pubkey(a)[1]['type'] for a in sample]
    from collections import Counter
    detected = Counter(types).most_common(1)[0][0] if types else "SegWit"

    vb_map = {
        'P2PKH': (148, 34), 'P2SH': (91, 32), 'SegWit': (68, 31), 'Taproot': (57, 43)
    }
    input_vb_global, output_vb_global = vb_map.get(detected.split()[0], (68, 31))

    # Apply pruning strategy
    ratio = {
        "Privacy First (30% pruned)": 0.3,
        "Recommended (40% pruned)": 0.4,
        "More Savings (50% pruned)": 0.5
    }.get(strategy, 0.4)

    strategy_name = {
        "Privacy First (30% pruned)": "Privacy First",
        "Recommended (40% pruned)": "Recommended",
        "More Savings (50% pruned)": "More Savings"
    }.get(strategy, strategy.split(" (")[0])

    keep = max(1, int(len(utxos) * (1 - ratio)))
    pruned_utxos_global = utxos[:keep]

    return (
        f"""
        <div style="text-align:center; padding:20px;">
            <b style="font-size:22px; color:#f7931a;">Analysis Complete</b><br><br>
            Found <b>{len(utxos):,}</b> UTXOs • Keeping <b>{keep}</b> largest<br>
            <b style="color:#f7931a;">Strategy:</b> <b>{strategy_name}</b> • Format: <b>{detected}</b><br><br>
            Click <b>Generate Transaction</b> to continue
        </div>
        """,
        gr.update(visible=True)
    )

# ==============================
# UPDATED build_real_tx — PSBT ONLY
# ==============================
def build_real_tx(user_input, strategy, threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice):
    global pruned_utxos_global, input_vb_global, output_vb_global

    if not pruned_utxos_global:
        return "Run analysis first", gr.update(visible=False), gr.update(visible=False), "", "", ""

    sample_addr = pruned_utxos_global[0].get('address') or user_input.strip()
    _, info = address_to_script_pubkey(sample_addr)
    detected = info['type']

    strategy_name = {
        "Privacy First (30% pruned)": "Privacy First",
        "Recommended (40% pruned)": "Recommended", 
        "More Savings (50% pruned)": "More Savings"
    }.get(strategy, strategy.split(" (")[0])

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)
    outputs = 1 + (1 if not selfish_mode and dao_percent > 0 else 0)
    weight = 40 + inputs * input_vb_global*4 + outputs * output_vb_global*4 + 4
    if detected in ("SegWit", "Taproot"):
        weight += 2
    vsize = (weight + 3) // 4   # ceiling division

    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        fee_rate = 2

    future_rate = max(fee_rate * 6, 100)
    future_cost = int((input_vb_global * inputs + output_vb_global) * future_rate)
    miner_fee = max(1000, int(vsize * fee_rate * 1.2))
    savings = future_cost - miner_fee
    dao_cut = max(546, int(savings * dao_percent / 10_000)) if not selfish_mode and dao_percent > 0 and savings > 2000 else 0
    user_gets = total - miner_fee - dao_cut

    if user_gets < 546:
        return "Not enough after fees", gr.update(visible=False), gr.update(visible=False), "", "", ""

    if ln_invoice and ln_invoice.strip().lower().startswith("lnbc"):
            return lightning_sweep_flow(pruned_utxos_global, ln_invoice.strip(), miner_fee, dao_cut, selfish_mode, detected)

    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Invalid destination", gr.update(visible=False), gr.update(visible=False), "", "", ""

    # Build the transaction
    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    # Generate BOTH outputs
    psbt_b64 = make_psbt(tx)

    # Generate clean, unsigned, RBF-ready raw hex — 100% compatible with ALL wallets
    unsigned_tx = tx.encode(segwit=True)

    # CRITICAL: Strip the empty witness placeholder (4 zero bytes) when unsigned
    if unsigned_tx.endswith(b'\x00\x00\x00\x00'):
        unsigned_tx = unsigned_tx[:-4]

    raw_hex = unsigned_tx.hex()

    qr = make_qr(psbt_b64)

    thank = "No thank-you" if dao_cut == 0 else f"Thank-you: {format_btc(dao_cut)}"

    details_section = f"""
<details style="margin-top: 40px;">
    <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px;">
        View PSBT (click to expand)
    </summary>
    <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px;">
{psbt_b64}
    </pre>
</details>
"""

    html = f"""
    <div style="text-align:center; padding:20px;">
        <h3 style="color:#f7931a;">Transaction Ready — PSBT Generated</h3>
        <p><b>{inputs}</b> inputs → {format_btc(total)} • Fee: {format_btc(miner_fee)} @ {fee_rate} sat/vB • {thank}</p>
        <b style="font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d;">You receive: {format_btc(user_gets)}</b>
        <div style="margin: 30px 0; padding: 18px; background: rgba(247,147,26,0.12); border-radius: 14px; border: 1px solid #f7931a; text-align: center;">
            Future savings ≈ <b style="font-size: 28px; color: #00ff9d; font-weight: 900; text-shadow: 0 2px 8px rgba(0,255,157,0.6); letter-spacing: 0.5px;">{format_btc(savings)}</b>
            <span style="color: #cccccc; font-size: 16px;"> (@ {future_rate} sat/vB)</span>
        </div>

        <div style="margin:40px 0;">
            <div class="qr-center">
                <img src="{qr}" style="width:460px; max-width:96vw; border-radius:20px; border:6px solid #f7931a; box-shadow:0 12px 50px rgba(247,147,26,0.6);">
            </div>
        </div>

        <p><small>Scan with Sparrow • Nunchuk • BlueWallet • Electrum</small></p>

        <details style="margin-top: 32px;">
            <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px; text-align:center; padding:12px 0;">
                View PSBT (click to expand)
            </summary>
            <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px; text-align:left;">
{psbt_b64}
            </pre>
        </details>

        <p style="margin:36px 0 20px; color:#f7931a; font-weight:bold; font-size:18px; line-height:1.4;">
            RBF ready — click "Bump +50 sat/vB" anytime (survives refresh)
        </p>
    </div>
    """

    return (
        html,
        gr.update(visible=False),   # generate_btn
        gr.update(visible=True),    # ln_invoice_row
        "",                         # ln_invoice_state (cleared)
        raw_hex                     # saved for infinite RBF
    )
# ==============================
# LIGHTNING SWEEP — NOW PSBT TOO (Optional: Raw Hex OK here)
# ==============================
def lightning_sweep_flow(utxos, invoice, miner_fee, dao_cut, selfish_mode, detected="SegWit"):
    if not bolt11_decode:
        return "bolt11 library missing — Lightning disabled", gr.update(visible=False), gr.update(visible=False), "", ""

    try:
        decoded = bolt11_decode(invoice)
        total = sum(u['value'] for u in utxos)
        user_gets = total - miner_fee - (0 if selfish_mode else dao_cut)

        if abs(user_gets * 1000 - (decoded.amount_msat or 0)) > 5_000_000:
            raise ValueError("Invoice amount mismatch (±5k sats)")

        if not getattr(decoded.payment_address):
            raise ValueError("Invoice must support on-chain fallback (payment_address)")

        dest_script, _ = address_to_script_pubkey(decoded.payment_address)
        tx = Tx()
        for u in utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
        tx.tx_outs.append(TxOut(user_gets, dest_script))
        if dao_cut and not selfish_mode:
            dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
            tx.tx_outs.append(TxOut(dao_cut, dao_script))

        raw_hex = tx.encode(segwit=True)
        if raw_hex.endswith(b'\x00\x00\x00\x00'):
            raw_hex = raw_hex[:-4]
        raw_hex = raw_hex.hex()
        qr = make_qr(raw_hex)

        html = f"""
        <div style="text-align:center; padding:20px; color:#00ff9d;">
            <h3>Lightning Sweep Ready</h3>
            <b style="font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d;">
                {format_btc(user_gets)} to Lightning Instantly
            </b>
            <div style="margin:40px 0;">
                <div class="qr-center"><img src="{qr}" style="width:460px; max-width:96vw; border-radius:20px; border:6px solid #00ff9d; box-shadow:0 12px 50px rgba(0,255,157,0.5);"></div>
            </div>
            <p><small>Scan with Phoenix • Breez • Zeus • Blink • Muun</small></p>
        </div>
        """

        return html, gr.update(visible=False), gr.update(visible=False), invoice, raw_hex

    except Exception as e:
        required = total - miner_fee - (0 if selfish_mode else dao_cut)
        return f"""
        <div style="text-align:center; color:#ff3333; padding:30px; background:#300; border-radius:16px;">
            <b style="font-size:22px;">Lightning Failed</b><br><br>{str(e)}<br><br>
            Invoice must be for ~{format_btc(required)} (±5k sats)
        </div>
        """, gr.update(visible=False), gr.update(visible=True), invoice, ""


# ==============================
# Gradio UI — Final & Perfect
# ==============================
with gr.Blocks(
    title="Omega v10 — Infinite Edition",
) as demo:
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
    gr.HTML("""
    <div style="
        text-align: center;
        margin: 24px 0 32px 0;
        padding: 12px;
        pointer-events: none;   /* prevents accidental clicks */
    ">
        <h1 style="
            font-size: 38px !important;
            font-weight: 900 !important;
            color: #f7931a !important;
            margin: 0 0 12px 0 !important;
            text-shadow: 0 4px 12px rgba(247,147,26,0.4);
            line-height: 1.2;
        ">Omega Pruner v10.0 — Infinite Edition</h1>
        
        <p style="
            font-size: 18px !important;
            color: #aaa !important;
            margin: 8px 0 !important;
            line-height: 1.5;
        ">
            Zero custody • Infinite one-click RBF • Lightning sweep • Survives refresh<br>
            The last UTXO consolidator you'll ever need
        </p>
        
        <p style="
            font-size: 15px !important;
            color: #f7931a !important;
            margin: 16px 0 0 0 !important;
        ">
            Source: <a href="https://github.com/babyblueviper1/Viper-Stack-Omega" target="_blank" style="color:#f7931a; text-decoration: underline;">GitHub</a> – Apache 2.0 • No logs • Runs in your browser
        </p>
    </div>
      
    # ====================== LAYOUT STARTS HERE ======================
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="Address or xpub",
                placeholder="bc1q… or xpub…",
                lines=2
            )
        with gr.Column(scale=3):
            prune_choice = gr.Dropdown(
                choices=[
                    "Privacy First (30% pruned)",
                    "Recommended (40% pruned)",
                    "More Savings (50% pruned)"
                ],
                value="Recommended (40% pruned)",
                label="Strategy"
            )

    with gr.Row():
        selfish_mode = gr.Checkbox(label="Selfish mode – keep 100%", value=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            dust_threshold = gr.Slider(
                minimum=0,
                maximum=3000,
                value=546,
                step=1,
                label="Dust threshold (sats)",
                info="UTXOs below this value are ignored"
            )
        with gr.Column(scale=1, min_width=300):
            dao_percent = gr.Slider(
                minimum=0,
                maximum=500,
                value=50,
                step=10,
                label="Thank-you to Ω author (basis points)",
                info="0 bps = keep 100% • 500 bps = 5%"
            )
            live_thankyou = gr.Markdown(
                "<div style='text-align: right; margin-top: 8px; font-size: 20px; color: #f7931a; font-weight: bold;'>"
                "→ 0.50% of future savings"
                "</div>"
            )

    def update_thankyou_label(bps):
        pct = bps / 100
        return f"<div style='text-align: right; margin-top: 8px; font-size: 20px; color: #f7931a; font-weight: bold;'>→ {pct:.2f}% of future savings</div>"

    dao_percent.change(update_thankyou_label, dao_percent, live_thankyou)

    with gr.Row():
        with gr.Column(scale=4):
            dest_addr = gr.Textbox(
                label="Destination (optional)",
                placeholder="Leave blank = same address"
            )
        with gr.Column(scale=3):
            dao_addr = gr.Textbox(
                label="Thank-you address (optional)",
                value=DEFAULT_DAO_ADDR,
                placeholder="Leave blank to support the Ω author"
            )

    with gr.Row():
        submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary")

    output_log = gr.HTML()

    with gr.Row():
        generate_btn = gr.Button(
            "2. Generate Transaction",
            visible=False,
            variant="primary",
            size="lg",
            elem_classes="full-width"
        )

    ln_invoice_state = gr.State("")

    with gr.Row(visible=False) as ln_invoice_row:
        with gr.Column(scale=7):
            ln_invoice = gr.Textbox(
                label="Lightning Invoice → paste lnbc… to sweep instantly",
                placeholder="Paste your invoice here",
                lines=4
            )
        with gr.Column(scale=3):
            submit_ln_btn = gr.Button(
                "Generate Lightning Sweep",
                variant="primary",
                size="lg",
                elem_classes="tall-button"
            )

    with gr.Row():
        start_over_btn = gr.Button(
            "Start Over — Clear Everything",
            variant="secondary",
            size="lg",
            elem_classes="full-width"
        )
    # =================== INFINITE RBF SECTION ===================
    gr.Markdown("### Infinite RBF Bump Zone")

    with gr.Row():
        with gr.Column(scale=8):
            rbf_in = gr.Textbox(
                label="Raw hex (auto-saved from last tx)",
                lines=6,
                elem_classes="rbf-textbox"
            )

        # RIGHT COLUMN — THE FINAL SOLUTION
        with gr.Column(scale=4):
            # Copy button — standalone
            copy_btn = gr.Button(
                "Copy raw hex",
                size="sm",
                elem_classes="rbf-copy-btn"
            ).click(
                None, None, None,
                js="""
                () => {
                    const t = document.querySelector('textarea[label*="Raw hex"]');
                    if (t && t.value) {
                        navigator.clipboard.writeText(t.value);
                        alert("Copied!");
                    }
                }
                """
            )

            # Clear button — standalone, with margin
            clear_btn = gr.Button(
                "Clear saved",
                size="sm",
                elem_classes="rbf-clear-btn"
            ).click(
                None, None, None,
                js="""
                () => {
                    localStorage.removeItem('omega_rbf_hex');
                    alert('Cleared!');
                    location.reload();
                }
                """
            )

            # Bump button — full width, big gap above
            rbf_btn = gr.Button(
                "Bump +50 sat/vB to Miners",
                variant="primary",
                size="lg",
                elem_classes="rbf-bump-btn"
            )
    # ==================================================================
    # Events
    # ==================================================================
    submit_btn.click(
        analysis_pass,
        [user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr],
        [output_log, generate_btn]
    )

    generate_btn.click(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice_state],
        outputs=[output_log, generate_btn, ln_invoice_row, ln_invoice_state, rbf_in]
    )

    ln_invoice.change(lambda x: x, ln_invoice, ln_invoice_state)

    submit_ln_btn.click(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice_state],
        outputs=[output_log, generate_btn, ln_invoice_row, ln_invoice_state]
    )

    ln_invoice.submit(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice_state],
        outputs=[output_log, generate_btn, ln_invoice_row, ln_invoice_state]
    )

    start_over_btn.click(
        lambda: (
            "", "Recommended (40% pruned)", 546, "", False, 50, DEFAULT_DAO_ADDR,
            "", gr.update(visible=False), gr.update(visible=False),
            "", "", ""
        ),
        outputs=[
            user_input, prune_choice, dust_threshold, dest_addr,
            selfish_mode, dao_percent, dao_addr,
            output_log, generate_btn, ln_invoice_row,
            ln_invoice, ln_invoice_state, rbf_in
        ]
    )

    rbf_btn.click(
        fn=rbf_bump,
        inputs=rbf_in,
        outputs=[rbf_in, output_log],
        js="""
        (hex) => {
            if (hex && typeof hex === 'string') {
                try { localStorage.setItem('omega_rbf_hex', hex.trim()); }
                catch(e) { console.warn('localStorage full'); }
            }
        }
        """
    )

    # Floating QR Scanners + Styles
     gr.HTML("""
     <label class="qr-fab btc" title="Scan Address / xpub">B</label>
    <label class="qr-fab ln" title="Scan Lightning Invoice">&#9889;</label>

    <input type="file" accept="image/*" capture="environment" id="qr-scanner-btc" style="display:none">
    <input type="file" accept="image/*" capture="environment" id="qr-scanner-ln" style="display:none">

    <script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
    <script>
    const btcBtn = document.querySelector('.qr-fab.btc');
    const lnBtn = document.querySelector('.qr-fab.ln');
    const btcInput = document.getElementById('qr-scanner-btc');
    const lnInput = document.getElementById('qr-scanner-ln');
    btcBtn.onclick = () => btcInput.click();
    lnBtn.onclick = () => lnInput.click();

    async function scan(file, isLightning = false) {
      if (!file) return;
      const img = new Image();
      img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
          const result = await ZXing.readBarcodeFromCanvas(canvas);
          const text = result.text.trim();
          if (isLightning && text.toLowerCase().startsWith('lnbc')) {
            const box = document.querySelector('textarea[placeholder*="lnbc"]') || document.querySelector('textarea');
            if (box) { box.value = text; box.dispatchEvent(new Event('input')); box.dispatchEvent(new Event('change')); }
            alert("Lightning invoice scanned!");
          } else if (!isLightning && /^(bc1|[13]|xpub|ypub|zpub|tpub)/i.test(text.split('?')[0].replace(/^bitcoin:/i, '').trim())) {
            const cleaned = text.split('?')[0].replace(/^bitcoin:/i, '').trim();
            const box = document.querySelector('textarea[placeholder*="bc1q"], textarea[placeholder*="xpub"]') || document.querySelector('textarea');
            if (box) { box.value = cleaned; box.dispatchEvent(new Event('input')); box.dispatchEvent(new Event('change')); }
            alert("Address/xpub scanned!");
          } else alert("Not recognized");
        } catch (e) { alert("No QR code detected"); }
      };
      img.src = URL.createObjectURL(file);
    }
    btcInput.onchange = e => scan(e.target.files[0], false);
    lnInput.onchange = e => scan(e.target.files[0], true);

    const saved = localStorage.getItem('omega_rbf_hex');
    if (saved) document.querySelector('textarea[label*="Raw hex"]')?.value = saved;
    </script>

    <style>
      .qr-fab {
        position: fixed !important;
        right: 20px !important;
        width: 70px !important;
        height: 70px !important;
        border-radius: 50% !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.7) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 38px !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        border: 5px solid white !important;
        font-weight: bold !important;
        user-select: none !important;
        z-index: 9999 !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
      }
      .qr-fab:hover { transform: scale(1.18) !important; box-shadow: 0 16px 50px rgba(0,0,0,0.8) !important; }
      .qr-fab.btc { bottom: 100px !important; background: linear-gradient(135deg,#f7931a,#f9a43f) !important; color: white !important; }
      .qr-fab.ln  { bottom: 20px !important;  background: linear-gradient(135deg,#00ff9d,#33ffc7) !important; color: #000 !important; font-size: 42px !important; }
    </style>
    """)
    
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
    )
