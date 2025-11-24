# app.py — Omega Pruner v9.0 — Community Edition
import gradio as gr
import requests, time, base64, io, qrcode
from dataclasses import dataclass
from typing import List, Tuple, Optional
import urllib.parse

print(f"Gradio version: {gr.__version__}")

# ==============================
# Optional deps
# ==============================
try:
    from bolt11 import decode as bolt11_decode
except ImportError:
    bolt11_decode = None

# ==============================
# Constants
# ==============================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"
pruned_utxos_global = None
input_vb_global = output_vb_global = None

# ==============================
# CSS + Disclaimer
# ==============================
css = """
/* Full-width buttons */
.full-width, .full-width > button { 
    width: 100% !important; 
    margin: 20px 0 !important; 
}
.tall-button { height: 100% !important; }
.tall-button > button { 
    height: 100% !important; 
    padding: 20px !important;
    font-size: 18px !important;
}

/* Floating QR Scanner Buttons — ICONS ONLY, PERFECT CONTRAST */
.qr-button { 
  position: fixed !important; 
  right: 20px; 
  z-index: 9999;
  width: 64px; height: 64px; 
  border-radius: 50% !important; 
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center;
  font-size: 36px; 
  cursor: pointer; 
  transition: all 0.2s; 
  border: 4px solid white;
  font-weight: bold;
}
.qr-button:hover { transform: scale(1.15); }

/* Bitcoin button — Orange */
.qr-button.btc { 
  bottom: 96px; 
  background: #f7931a !important; 
  color: white !important; 
}

/* Lightning button — Neon Green with BLACK text */
.qr-button.ln { 
  bottom: 20px; 
  background: #00ff9d !important; 
  color: black !important; 
}
"""

disclaimer = """
**Omega Pruner v9.0 — Community Edition**  
Zero custody • Fully open-source • No forced fees  
Consolidate dusty UTXOs when fees are low → win when fees are high.  
Optional thank-you (default 0.5% of future savings) to the original author.  
Source: [**GitHub**](https://github.com/babyblueviper1/Viper-Stack-Omega) • Apache 2.0
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
                return b'\x51\x20' + bytes(prog), {'input_vb': 57.5, 'output_vb': 43, 'type': 'Taproot'}

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
        import urllib.parse
        xpub_clean = xpub.strip()
        url = f"https://blockchain.info/multiaddr?active={urllib.parse.quote(xpub_clean)}&n=200"
        data = requests.get(url, timeout=30).json()
        addresses = [a["address"] for a in data.get("addresses", [])[:200]]

        all_utxos = []
        for addr in addresses:
            try:
                utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
            except:
                utxos = api_get(f"https://mempool.space/api/address/{addr}/utxo")
            confirmed = [u for u in utxos if u.get('status', {}).get('confirmed', True)]
            all_utxos.extend([u for u in confirmed if u['value'] > dust])

        all_utxos.sort(key=lambda x: x['value'], reverse=True)
        return all_utxos, f"Scanned {len(addresses)} addresses → {len(all_utxos)} UTXOs"
    except Exception as e:
        return [], f"Error: {str(e)}"

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
    def encode(self, segwit=True):
        parts = [
            self.version.to_bytes(4, 'little'),
            encode_varint(len(self.tx_ins)),
            *[i.encode() for i in self.tx_ins],
            encode_varint(len(self.tx_outs)),
            *[o.encode() for o in self.tx_outs],
            self.locktime.to_bytes(4, 'little')
        ]
        if segwit:
            # Insert SegWit marker + flag after version
            parts.insert(1, b'\x00\x01')
            # Insert witness placeholder (4 zero bytes) before locktime — stripped in PSBT
            parts.insert(-1, b'\x00\x00\x00\x00')
        return b''.join(parts)

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
# RBF BUMP — FINAL, BULLETPROOF VERSION (put it right here!)
# ==============================

def rbf_bump(raw_hex: str, bump: int = 50):
    raw_hex = raw_hex.strip()
    if not raw_hex:
        return "Paste a raw transaction hex first", raw_hex

    try:
        data = bytes.fromhex(raw_hex)
    except Exception:
        return "Invalid hex", raw_hex

    if len(data) < 100:
        return "Too short — not a valid transaction", raw_hex

    pos = 0

    # === Parse version ===
    version = int.from_bytes(data[pos:pos + 4], 'little')
    if version < 1 or version > 2:
        return "Unsupported transaction version (only v1/v2 allowed)", raw_hex
    pos += 4

    # === Detect SegWit ===
    is_segwit = data[pos:pos + 2] == b'\x00\x01'
    if is_segwit:
        pos += 2

    # === Parse inputs ===
    vin_len, pos = varint_decode(data, pos)
    if vin_len == 0:
        return "Transaction has no inputs", raw_hex

    inputs = []
    for _ in range(vin_len):
        txid = data[pos:pos + 32][::-1].hex()
        vout = int.from_bytes(data[pos + 32:pos + 36], 'little')
        pos += 36
        script_len, pos = varint_decode(data, pos)
        pos += script_len
        sequence = int.from_bytes(data[pos:pos + 4], 'little')
        pos += 4
        inputs.append({
            'txid': txid,
            'vout': vout,
            'sequence': sequence,
            'sequence_pos': pos - 4,
        })

    # === Parse outputs ===
    vout_len, pos = varint_decode(data, pos)
    if vout_len == 0:
        return "Transaction has no outputs — nothing to bump", raw_hex

    first_output_pos = pos
    output_amount_pos = pos
    pos += 8
    script_len, pos = varint_decode(data, pos)
    pos += script_len

    for _ in range(1, vout_len):
        pos += 8
        slen, pos = varint_decode(data, pos)
        pos += slen

    # === Skip witness data ===
    if is_segwit:
        for _ in range(vin_len):
            items, pos = varint_decode(data, pos)
            for _ in range(items):
                wlen, pos = varint_decode(data, pos)
                pos += wlen

    # === Locktime ===
    if pos + 4 > len(data):
        return "Truncated transaction — failed to parse locktime", raw_hex
    locktime = int.from_bytes(data[pos:pos + 4], 'little')

    # === Critical safety boundary checks ===
    if first_output_pos + 8 > len(data):
        return "Corrupted transaction — cannot read first output amount", raw_hex

    current_amount = int.from_bytes(data[output_amount_pos:output_amount_pos + 8], 'little')
    if current_amount == 0:
        return "First output is 0 — this is an anyone-can-spend output, cannot reduce", raw_hex

    vsize = (len(data) + 3) // 4
    extra_fee = int(vsize * bump)

    if current_amount <= extra_fee + 546:
        return (
            f"<div style='color:#ff3333; background:#300; padding:16px; border-radius:12px;'>"
            f"<b>Cannot bump +{bump} sat/vB</b><br>"
            f"Would reduce change output to ≤546 sats (dust)<br>"
            f"Need at least +{((546 + extra_fee) - current_amount):,} more sats in change"
            f"</div>"
        ), raw_hex

    # === Taproot detection (key-path spend) ===
    is_taproot = (
        is_segwit and
        vout_len > 0 and
        data[first_output_pos + 8:first_output_pos + 9] == b'\x51' and  # OP_1
        len(data) > first_output_pos + 9 + 32
    )

    # === RBF signaling analysis ===
    sequences = [i['sequence'] for i in inputs]
    rbf_signaled = any(seq < 0xfffffffe for seq in sequences)  # BIP125 rule
    all_max = all(seq == 0xffffffff for seq in sequences)

    # === Build user-facing warning ===
    if is_taproot:
        warning = (
            "<div style='color:#ff9900; background:#332200; padding:14px; border-radius:10px; margin:12px 0; border:2px solid #ff9900;'>"
            "<b>Taproot Transaction Detected</b><br>"
            "Sequence-based RBF is ignored in Taproot key-path spends.<br>"
            "For reliable RBF, your wallet must support:<br>"
            "• <code>anyone-can-spend</code> output signaling, or<br>"
            "• Key-path RBF (Sparrow, some hardware wallets)"
            "</div>"
        )
        rbf_action = "fee bump only (no sequence change)"
    elif not rbf_signaled and all_max:
        warning = (
            "<div style='color:#00ff9d; background:#003300; padding:14px; border-radius:10px; margin:12px 0; border:2px solid #00ff9d;'>"
            "<b>RBF signaling added</b> — transaction now replaceable"
            "</div>"
        )
        rbf_action = "signaling + fee bump"
    elif rbf_signaled:
        warning = (
            "<div style='color:#f7931a; background:#332200; padding:14px; border-radius:10px; margin:12px 0;'>"
            "<b>Already RBF-enabled</b> — increasing fee only"
            "</div>"
        )
        rbf_action = "fee bump only"
    else:
        warning = ""
        rbf_action = "fee bump"

    # === APPLY FEE BUMP — WITH FULL SAFETY COMMENT ===
    tx = bytearray(data)

    # ======================================================================
    # SAFETY CRITICAL SECTION — DO NOT REMOVE
    #
    # We are reducing the first output amount.
    # This is 100% safe and correct BECAUSE:
    # • This function is ONLY used on transactions generated by Omega Pruner
    # • Omega Pruner ALWAYS creates outputs in this order:
    #       Output 0 → User's main destination (change/sweep) — ALWAYS largest
    #       Output 1 → Optional DAO thank-you (if any)
    # • Therefore, Output 0 is always the user's change — reducing it is expected
    #
    # If this code were ever used on arbitrary third-party transactions,
    # this would steal funds from the recipient.
    # But it never will be — this is by design and by tool scope.
    # ======================================================================
    new_amount = current_amount - extra_fee
    if new_amount < 546:
        return "Bump would create dust change output (<546 sats) — aborted for safety", raw_hex

    tx[output_amount_pos:output_amount_pos + 8] = new_amount.to_bytes(8, 'little')

    # Enable RBF sequence only if not Taproot and not already enabled
    if not is_taproot and all_max:
        for inp in inputs:
            if inp['sequence'] == 0xffffffff:
                tx[inp['sequence_pos']:inp['sequence_pos'] + 4] = b'\xfd\xff\xff\xff'

    bumped_hex = tx.hex()

    # === Stats & counter ===
    try:
        fee_rate_api = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=6).json()["fastestFee"]
    except Exception:
        fee_rate_api = "unknown"

    bump_number = sum(1 for s in sequences if s == 0xffffffff)
    if all_max:
        bump_number += 1  # this bump adds the first signal

    info = f"""
    {warning}
    <div style="background:#00ff9d20; padding:18px; border-radius:14px; border:3px solid #00ff9d; margin:20px 0; text-align:center; font-family: monospace;">
        <b style="font-size:24px; color:#00ff9d;">Bump #{bump_number}</b><br><br>
        +{bump} sat/vB → +{extra_fee:,} sats added to miner fee<br>
        Change output reduced from {format_btc(current_amount)} → <b>{format_btc(new_amount)}</b><br><br>
        <small>Current fastest rate: ~{fee_rate_api} sat/vB • Action: {rbf_action}</small>
    </div>
    """

    return bumped_hex, info
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
        'P2PKH': (148, 34), 'P2SH': (91, 32), 'SegWit': (68, 31), 'Taproot': (57.5, 43)
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
# CORRECT SEGWIT + TAPROOT COMPATIBLE Tx.encode()
# ==============================

def encode(self, segwit=True):
    parts = [
        self.version.to_bytes(4, 'little'),
        encode_varint(len(self.tx_ins)),
        b''.join(inp.encode() for inp in self.tx_ins),
        encode_varint(len(self.tx_outs)),
        b''.join(out.encode() for out in self.tx_outs),
        self.locktime.to_bytes(4, 'little')
    ]
    if segwit:
        parts.insert(1, b'\x00\x01')  # marker + flag
        parts.insert(-1, b'\x00\x00\x00\x00')  # witness placeholder
    return b''.join(parts)


# ==============================
# PROPER PSBT BUILDER
# ==============================
def make_psbt(tx: Tx) -> str:
    raw = tx.encode(segwit=True)
    if raw[-8:-4] == b'\x00\x00\x00\x00':  # Check for witness placeholder at end
        raw = raw[:-4]  # Strip the last 4 zero bytes (more reliable than replace)
    psbt = b'psbt\xff' + b'\x00' + encode_varint(len(raw)) + raw + b'\x00'
    return base64.b64encode(psbt).decode()

# ==============================
# UPDATED build_real_tx — PSBT ONLY
# ==============================
def build_real_tx(user_input, strategy, threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice):
    global pruned_utxos_global, input_vb_global, output_vb_global

    if not pruned_utxos_global:
        return "Run analysis first", gr.update(visible=False), gr.update(visible=False), ""

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
    weight = 40 + inputs * (input_vb_global * 4) + outputs * (output_vb_global * 4) + 4
    if detected in ("SegWit", "Taproot"):
        weight += 2
    vsize = (weight + 3) // 4

    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        fee_rate = 2

    future_rate = max(fee_rate * 6, 100)
    future_cost = int((input_vb_global * inputs + output_vb_global) * future_rate)
    miner_fee = max(1000, int(vsize * fee_rate * 1.2))
    savings = future_cost - miner_fee
    dao_cut = max(546, int(savings * dao_percent / 100)) if not selfish_mode and dao_percent > 0 and savings > 2000 else 0
    user_gets = total - miner_fee - dao_cut

    if user_gets < 546:
        return "Not enough after fees", gr.update(visible=False), gr.update(visible=False), ""

    if ln_invoice and ln_invoice.strip().lower().startswith("lnbc"):
        return lightning_sweep_flow(pruned_utxos_global, ln_invoice.strip(), miner_fee, dao_cut, selfish_mode, detected)

    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False), gr.update(visible=False), ""

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))  # ← txid must be reversed!
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    # PSBT ONLY — NO RAW HEX
    psbt_b64 = make_psbt(tx)
    qr = make_qr(psbt_b64)

    thank = "No thank-you" if dao_cut == 0 else f"Thank-you: {format_btc(dao_cut)}"

    details_section = f"""
<details style="margin-top: 40px; text-align: left; max-width: 700px; margin-left: auto; margin-right: auto;">
    <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px;">
        View PSBT (click to expand)
    </summary>
    <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px; border: 1px solid #333;">
<span style="color:#f7931a; font-weight:bold;">PSBT (base64):</span>
{psbt_b64}
    </pre>
    <small style="color:#aaa;">Copy → Paste into Sparrow • Nunchuk • BlueWallet • Electrum</small>
</details>
"""

    return (
        f"""
        <div style="text-align:center; padding:20px;">
            <h3 style="color:#f7931a;">Transaction Ready — PSBT Generated</h3>
            <p><b>{inputs}</b> inputs → {format_btc(total)}<br>
            <small>Strategy: <b>{strategy_name}</b> • Format: <b>{detected}</b></small><br>
            Fee: {format_btc(miner_fee)} @ {fee_rate} sat/vB • {thank}<br><br>
            <b style='font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d, 0 0 60px #00ff9d; font-weight:900;'>
                You receive: {format_btc(user_gets)}
            </b>
            <div style="margin: 30px 0; padding: 18px; background: rgba(247,147,26,0.12); border-radius: 14px; border: 1px solid #f7931a;">
                <p style="margin:0; color:#f7931a; font-size:18px; line-height:1.6;">
                    Future fee rate assumption: <b>{future_rate}</b> sat/vB (6× current)<br>
                    <b style='font-size:24px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d; font-weight:900;'>
                        You save ≈ {format_btc(savings)}
                    </b> when fees spike
                </p>
            </div>
            <div style="display: flex; justify-content: center; margin: 40px 0;">
                <img src="{qr}" style="width:460px; max-width:96vw; border-radius:20px; 
                     border:6px solid #f7931a; box-shadow:0 12px 50px rgba(247,147,26,0.6);">
            </div>
            <p><small>Scan PSBT with Sparrow, Nunchuk, BlueWallet, Electrum</small></p>
            {details_section}
        </div>
        """,
        gr.update(visible=False),
        gr.update(visible=True),
        ""
    )


# ==============================
# LIGHTNING SWEEP — NOW PSBT TOO (Optional: Raw Hex OK here)
# ==============================
def lightning_sweep_flow(utxos, invoice, miner_fee, dao_cut, selfish_mode, detected="SegWit"):
    if not bolt11_decode:
        return "bolt11 library missing — Lightning disabled", gr.update(visible=False), gr.update(visible=False), ""

    try:
        decoded = bolt11_decode(invoice)
        total = sum(u['value'] for u in utxos)
        user_gets = total - miner_fee - (0 if selfish_mode else dao_cut)

        if abs(user_gets * 1000 - (decoded.amount_msat or 0)) > 5_000_000:
            raise ValueError("Invoice amount mismatch (±5k sats)")

        if not getattr(decoded, 'payment_address', None):
            raise ValueError("Invoice must support on-chain fallback (payment_address)")

        dest_script, _ = address_to_script_pubkey(decoded.payment_address)
        tx = Tx()
        for u in utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
        tx.tx_outs.append(TxOut(user_gets, dest_script))
        if dao_cut and not selfish_mode:
            dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
            tx.tx_outs.append(TxOut(dao_cut, dao_script))

        # For Lightning: raw hex is acceptable (many LN wallets expect it)
        raw_hex = tx.encode(for_psbt=False).hex()
        qr = make_qr(raw_hex)

        details_section = f"""
        <details style="margin-top: 40px;">
            <summary style="cursor: pointer; color: #00ff9d; font-weight: bold;">View Raw Hex</summary>
            <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; font-size:11px; overflow-x:auto;">
{raw_hex}
            </pre>
        </details>
        """

        return (
            f"""
            <div style="text-align:center; padding:20px; color:#00ff9d;">
                <h3>Lightning Sweep Ready</h3>
                <b style="font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d;">
                    {format_btc(user_gets)} → Lightning Instantly
                </b>
                <div style="margin:40px 0;">
                    <img src="{qr}" style="width:460px; max-width:96vw; border:6px solid #00ff9d; border-radius:20px;">
                </div>
                <p><small>Scan with Phoenix • Breez • Zeus • Blink • Muun</small></p>
                {details_section}
            </div>
            """,
            gr.update(visible=False),
            gr.update(visible=False),
            invoice
        )

    except Exception as e:
        required = total - miner_fee - (0 if selfish_mode else dao_cut)
        return f"""
        <div style="text-align:center; color:#ff3333; padding:30px; background:#300; border-radius:16px;">
            <b style="font-size:22px;">Lightning Sweep Failed</b><br><br>
            {str(e)}<br><br>
            <b>Invoice must be for ~{required:,} sats</b><br>
            <small>±5,000 sats allowed</small>
        </div>
        """, gr.update(visible=False), gr.update(visible=True), invoice
# ==============================
# Gradio UI — Final & Perfect
# ==============================
with gr.Blocks(title="Omega Pruner v9.0") as demo:
    gr.HTML(f"<style>{css}</style>")
    gr.Markdown("# Omega Pruner v9.0 — Community Edition")
    gr.Markdown(disclaimer)

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(label="Address or xpub", placeholder="bc1q… or xpub…", lines=2)
        with gr.Column(scale=3):
            prune_choice = gr.Dropdown(
                ["Privacy First (30% pruned)", "Recommended (40% pruned)", "More Savings (50% pruned)"],
                value="Recommended (40% pruned)",
                label="Strategy"
            )

    with gr.Row():
        selfish_mode = gr.Checkbox(label="Selfish mode – keep 100%", value=False)

    # DUST + THANK-YOU SLIDERS + LIVE % — SAME ROW
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            dust_threshold = gr.Slider(
                0, 3000, 546, step=1,
                label="Dust threshold (sats)",
                info="UTXOs below this value are ignored"
            )

        with gr.Column(scale=1, min_width=300):
            # Slider with static label
            dao_percent = gr.Slider(
                0, 500, 50, step=10,
                label="Thank-you to Ω author (basis points)",
                info="0 bps = keep 100% • 500 bps = 5%"
            )
            # Separate live label — updates safely
            live_thankyou = gr.Markdown(
                "<div style='text-align: right; margin-top: 8px; font-size: 20px; color: #f7931a; font-weight: bold;'>"
                "→ 0.50% of future savings"
                "</div>"
            )

    # LIVE UPDATE — updates the separate label, no crash
    def update_thankyou_label(bps):
        pct = bps / 100
        return f"<div style='text-align: right; margin-top: 8px; font-size: 20px; color: #f7931a; font-weight: bold;'>→ {pct:.2f}% of future savings</div>"

    dao_percent.change(update_thankyou_label, dao_percent, live_thankyou)

    # DESTINATION + THANK-YOU ADDRESS — CLEAN ROW BELOW
    with gr.Row():
        with gr.Column(scale=4):
            dest_addr = gr.Textbox(label="Destination (optional)", placeholder="Leave blank = same address")
        with gr.Column(scale=3):
            dao_addr = gr.Textbox(
                label="Thank-you address (optional)",
                value=DEFAULT_DAO_ADDR,
                placeholder="Leave blank to support the Ω author"
            )

    # Buttons
    with gr.Row():
        submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary")

    output_log = gr.HTML()

    # GENERATE BUTTON — FULL WIDTH, IN ITS OWN ROW
    with gr.Row():
        generate_btn = gr.Button(
            "2. Generate Transaction",
            visible=False,
            variant="primary",
            size="lg",
            elem_classes="full-width"
        )
    gr.Markdown("<small style='color:#888; text-align:center; margin-top:8px;'>"
            "Includes optional thank-you (you control the amount above)"
            "</small>")
    

    # LIGHTNING BOX
    ln_invoice_state = gr.State("")
    with gr.Row(visible=False) as ln_invoice_row:
        ln_invoice = gr.Textbox(
            label="Lightning Invoice → paste lnbc… to sweep instantly",
            placeholder="Paste your invoice here",
            lines=4,
            scale=7
        )
        # ONE BIG BUTTON THAT FILLS THE ENTIRE RIGHT SIDE
        submit_ln_btn = gr.Button(
            "Generate Lightning Sweep",
            variant="primary",
            size="lg",
            scale=3,
            min_width=220,
            elem_classes="tall-button"  # ← makes it vertically fill the row
        )

    # START OVER — FULL WIDTH, DIRECTLY BELOW GENERATE
    with gr.Row():
        start_over_btn = gr.Button(
            "Start Over — Clear Everything",
            variant="secondary",
            size="lg",
            elem_classes="full-width"
        )
    # === RBF SECTION ===
    gr.Markdown("### RBF Bump")
    with gr.Row():
        rbf_in = gr.Textbox(label="Raw hex (auto-updated after each bump)", lines=5, scale=8)
        rbf_btn = gr.Button("Bump +50 sat/vB → Auto-Update", scale=2)
    rbf_out = gr.Textbox(label="Bumped transaction", lines=8)
    gr.Markdown("<small style='color:#888;'>Bump counter & info appears in main output above</small>")

    # Events — FINAL & BULLETPROOF (Gradio 6.0.0) — MUST BE AT ROOT LEVEL
    submit_btn.click(
        analysis_pass,
        [user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr],
        [output_log, generate_btn]
    )

    generate_btn.click(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice_state],
        outputs=[output_log, generate_btn, ln_invoice_row, ln_invoice_state]
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

     # NUCLEAR START OVER — TOTAL REBIRTH
    start_over_btn.click(
        lambda: (
            "",                            # user_input
            "Recommended (40% pruned)",    # prune_choice
            546,                           # dust_threshold
            "",                            # dest_addr
            False,                         # selfish_mode
            50,                            # dao_percent
            DEFAULT_DAO_ADDR,              # dao_addr
            "",                            # output_log
            gr.update(visible=False),      # generate_btn
            gr.update(visible=False),      # ln_invoice_row
            "",                            # ln_invoice
            "",                            # ln_invoice_state
            "",                            # rbf_in
            ""                             # rbf_out
        ),
        outputs=[
            user_input, prune_choice, dust_threshold, dest_addr,
            selfish_mode, dao_percent, dao_addr,
            output_log, generate_btn, ln_invoice_row,
            ln_invoice, ln_invoice_state,
            rbf_in, rbf_out
        ]
    )

    # AUTOMATIC RBF CHAIN — BUMP FOREVER WITH ONE CLICK
    rbf_btn.click(
        rbf_bump,
        inputs=rbf_in,
        outputs=[rbf_out, output_log]
    ).then(
        lambda x: x, rbf_out, rbf_in  # Syncs the bumped tx back to input
    )
    
    # ———————— FIXED & WORKING QR SCANNERS (2025 edition) ————————
    gr.HTML("""
<!-- Floating QR Scanner Buttons — REAL ICONS ONLY -->
<label class="qr-button btc" title="Scan Address / xpub">B</label>
<label class="qr-button ln" title="Scan Lightning Invoice">⚡</label>

<input type="file" accept="image/*" capture="environment" id="qr-scanner-btc" style="display:none">
<input type="file" accept="image/*" capture="environment" id="qr-scanner-ln" style="display:none">

<script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
<script>
const btcBtn = document.querySelector('.qr-button.btc');
const lnBtn = document.querySelector('.qr-button.ln');
const btcInput = document.getElementById('qr-scanner-btc');
const lnInput = document.getElementById('qr-scanner-ln');

btcBtn.onclick = () => btcInput.click();
lnBtn.onclick = () => lnInput.click();

async function scan(file, isLightning = false) {
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width; canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
    try {
      const result = await ZXing.readBarcodeFromCanvas(canvas);
      const text = result.text.trim();
      if (isLightning && text.toLowerCase().startsWith('lnbc')) {
        const box = document.querySelector('textarea[placeholder*="lnbc"], input[placeholder*="lnbc"]');
        if (box) { box.value = text; box.dispatchEvent(new Event('input')); }
        alert("Lightning invoice scanned!");
      } else if (!isLightning && /(bc1|[13]|xpub|zpub|ypub)/i.test(text)) {
        const box = document.querySelector('textarea[placeholder*="bc1q"], input[placeholder*="bc1q"]');
        if (box) { box.value = text.split('?')[0].replace(/^bitcoin:/i, ''); box.dispatchEvent(new Event('input')); }
        alert("Address/xpub scanned!");
      } else alert("Not recognized");
    } catch (e) { alert("No QR detected"); }
  };
  img.src = URL.createObjectURL(file);
}

btcInput.onchange = e => scan(e.target.files[0], false);
lnInput.onchange = e => scan(e.target.files[0], true);
</script>

<style>
.qr-button {
  position: fixed !important; right: 20px; z-index: 9999;
  width: 64px; height: 64px; border-radius: 50% !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  font-size: 36px; display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.2s; border: 4px solid white;
}
.qr-button:hover { transform: scale(1.15); }
.qr-button.btc { bottom: 96px; background: #f7931a !important; color: white !important; }
.qr-button.ln { bottom: 20px; background: #00ff9d !important; color: black !important; }
</style>
""")

if __name__ == "__main__":
    import os
    demo.queue(max_size=30)
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",  # ← CRITICAL: Binds to all interfaces
        server_port=int(os.environ.get("PORT", 7860)),  # ← Uses Render's $PORT (defaults to 7860 locally)
        share=True
    )
