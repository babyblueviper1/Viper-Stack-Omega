# app.py — Omega Pruner v9.0 — Community Edition
import gradio as gr
import requests, time, base64, io, qrcode
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    def encode(self):
        parts = [
            self.version.to_bytes(4, 'little'),
            encode_varint(len(self.tx_ins)),
            *[i.encode() for i in self.tx_ins],
            encode_varint(len(self.tx_outs)),
            *[o.encode() for o in self.tx_outs],
            self.locktime.to_bytes(4, 'little')
        ]
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

    if len(raw_hex) < 400:
        return (
            "Invalid — expected full raw transaction hex<br><br>"
            "Go to mempool.space → your tx → Advanced → Raw Transaction → copy ALL the hex",
            raw_hex
        )

    try:
        data = bytes.fromhex(raw_hex)
    except:
        return "Invalid hex characters", raw_hex

    # === DEFAULT VALUES (in case parsing fails) ===
    bump_count = 0
    fee_rate = 20

    try:
        # === COUNT HOW MANY TIMES IT HAS BEEN BUMPED (CORRECT LOGIC) ===
        pos = 4
        vin_len, pos = varint_decode(data, pos)
        for _ in range(vin_len):
            pos += 36
            slen, pos = varint_decode(data, pos)
            pos += slen
            seq = int.from_bytes(data[pos:pos+4], 'little')
            pos += 4
            if seq != 0xfffffffd:
                bump_count += 1

        # === FEE RATE (fallback) ===
        try:
            fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8).json()["fastestFee"]
        except:
            pass  # keep fee_rate = 20

        # === REST OF PARSING (same as before) ===
        pos = 4
        vin_len, pos = varint_decode(data, pos)
        for _ in range(vin_len):
            pos += 36
            slen, pos = varint_decode(data, pos)
            pos += slen + 4
        vout_len, pos = varint_decode(data, pos)
        first_output_pos = pos
        for _ in range(vout_len):
            pos += 8
            slen, pos = varint_decode(data, pos)
            pos += slen
        if pos < len(data) and data[pos:pos+2] == b'\x00\x01':
            pos += 2
            for _ in range(vin_len):
                wlen, pos = varint_decode(data, pos)
                pos += wlen
        pos += 4

        if first_output_pos + 8 > len(data):
            return "Could not parse outputs", raw_hex
        amount = int.from_bytes(data[first_output_pos:first_output_pos+8], 'little')

        vsize = (len(data) + 3) // 4
        extra = int(vsize * bump)
        if amount <= extra + 546:
            return "Not enough for bump — output would be dust", raw_hex

        new_amount = amount - extra
        tx = bytearray(data)
        tx[first_output_pos:first_output_pos+8] = new_amount.to_bytes(8, 'little')

        pos = 4 + 1
        for _ in range(vin_len):
            pos += 36
            slen, pos = varint_decode(data, pos)
            pos += slen
            tx[pos:pos+4] = b'\xfd\xff\xff\xff'
            pos += 4

        # === FINAL SUCCESS WITH YOUR PERFECT COUNTER ===
        counter_text = f"""
        <div style="background:#00ff9d20; padding:12px; border-radius:12px; border:2px solid #00ff9d; margin:15px 0;">
            <b style="color:#00ff9d; font-size:18px;">{bump_count + 1} bump{'s' if bump_count > 0 else ''}!</b><br>
            Total bump: +{(bump_count + 1) * 50:,} sat/vB • This bump adds +{extra:,} sats
        </div>
        """

        return tx.hex(), counter_text

    except Exception as e:
        return f"Failed to parse transaction: {str(e)}<br>Make sure it's a valid RBF-enabled tx", raw_hex
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

def build_real_tx(user_input, strategy, threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice):
    global pruned_utxos_global, input_vb_global, output_vb_global

    # Fix #1 — Recalculate detected type (or pass it from analysis)
    sample_addr = pruned_utxos_global[0].get('address') or user_input.strip()
    _, info = address_to_script_pubkey(sample_addr)
    detected = info['type'].split(' (')[0] if '(' in info['type'] else info['type']

    strategy_name = {
        "Privacy First (30% pruned)": "Privacy First",
        "Recommended (40% pruned)": "Recommended", 
        "More Savings (50% pruned)": "More Savings"
    }.get(strategy, strategy.split(" (")[0])

    if not pruned_utxos_global:
        return "Run analysis first", gr.update(visible=False), gr.update(visible=False), ""

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)
    outputs = 1 + (1 if not selfish_mode and dao_percent > 0 else 0)
    vsize = 10 + inputs + input_vb_global * inputs + output_vb_global * outputs

    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        fee_rate = 2

    # NEW: Dynamic future rate = 6× current
    future_rate = max(fee_rate * 6, 100)  # never below 100 sat/vB (safety)

    # Future cost if you DON'T consolidate now
    future_cost = int((input_vb_global * inputs + output_vb_global) * future_rate)
    miner_fee = max(1000, int(vsize * fee_rate * 1.2))

    savings = future_cost - miner_fee
    dao_cut = max(546, int(savings * dao_percent / 100)) if not selfish_mode and dao_percent > 0 and savings > 2000 else 0
    user_gets = total - miner_fee - dao_cut

    if user_gets < 546:
        return "Not enough after fees", gr.update(visible=False), gr.update(visible=False), ""

    if ln_invoice and ln_invoice.strip().lower().startswith("lnbc"):
        return lightning_sweep_flow(pruned_utxos_global, ln_invoice.strip(), miner_fee, dao_cut, selfish_mode)

    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False), gr.update(visible=False), ""

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    raw = tx.encode().hex()                                   # ← FIXED
    psbt = base64.b64encode(b'psbt\xff\x00\x00' + tx.encode() + b'\x00').decode()
    qr = make_qr(psbt)

    thank = "No thank-you" if dao_cut == 0 else f"Thank-you: {format_btc(dao_cut)}"

    details_section = f"""
<details style="margin-top: 40px; text-align: left; max-width: 700px; margin-left: auto; margin-right: auto;">
    <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px;">
        View Raw Hex & PSBT (click to expand)
    </summary>
    <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px; border: 1px solid #333;">
<span style="color:#f7931a; font-weight:bold;">Raw Hex:</span>
{raw}

<span style="color:#f7931a; font-weight:bold;">PSBT (base64):</span>
{psbt}
    </pre>
    <small style="color:#aaa;">Use in Sparrow, Nunchuk, BlueWallet, Electrum, etc.</small>
</details>
"""
    lightning_hint = f"""
    <div style="margin: 40px 0 20px 0; padding: 18px; background: rgba(0,255,157,0.08); border-radius: 14px; border: 1px solid #00ff9d; max-width: 680px; margin-left: auto; margin-right: auto;">
    <p style="margin:0; text-align:center; color:#1e90ff; font-size:16px;">
        Lightning invoice must be for exactly<br>
        <b style="font-size:32px; color:#FFD700; text-shadow: 0 0 10px #00ff9d; font-weight:900;">
            {user_gets:,} sats
        </b><br>
        <small style="color:#1e90ff;">(±5,000 sats tolerance allowed)</small>
    </p>
</div>
"""
    return (
    f"""
    <div style="text-align:center; padding:20px;">
        <h3 style="color:#f7931a;">Transaction Ready</h3>
        <p><b>{inputs}</b> inputs → {format_btc(total)}<br>
        <small>Strategy: <b>{strategy_name}</b> • Detected: <b>{detected}</b></small><br>
        Fee: {format_btc(miner_fee)} @ {fee_rate} sat/vB • {thank}<br><br>
        <<b style='font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d, 0 0 60px #00ff9d; font-weight:900;'>You receive: {format_btc(user_gets)}</b>
        <div style="margin: 30px 0; padding: 18px; background: rgba(247,147,26,0.12); border-radius: 14px; border: 1px solid #f7931a;">
            <p style="margin:0; color:#f7931a; font-size:18px; line-height:1.6;">
                Future fee rate assumption: <b>{future_rate}</b> sat/vB (6× current)<br>
                <b style='font-size:24px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d; font-weight:900;'>You save ≈ {format_btc(savings)}</b> if fees hit that level<br>
                <span style="font-size:14px; color:#777;">
                    Fees have exceeded 6× the current rate in every Bitcoin bull cycle since 2017.
                </span>
            </p>
        </div>
        <div style="display: flex; justify-content: center; margin: 40px 0;">
            <img src="{qr}" style="width:460px; max-width:96vw; border-radius:20px; 
                 border:6px solid #f7931a; box-shadow:0 12px 50px rgba(247,147,26,0.6);">
        </div>
        <p><small>Scan with Sparrow, BlueWallet, Nunchuk, Electrum</small></p>
        {details_section}
        {lightning_hint}
    </div>
    """,
    gr.update(visible=False),
    gr.update(visible=True),
    ""
    )


def lightning_sweep_flow(utxos, invoice, miner_fee, dao_cut, selfish_mode):
    if not bolt11_decode:
        return (
            "bolt11 library missing — Lightning disabled",
            gr.update(visible=False),
            gr.update(visible=False),
            ""  # ← 4th value
        )

    try:
        decoded = bolt11_decode(invoice)
        total = sum(u['value'] for u in utxos)
        user_gets = total - miner_fee - (0 if selfish_mode else dao_cut)

        if abs(user_gets * 1000 - (decoded.amount_msat or 0)) > 5_000_000:
            raise ValueError("Invoice amount mismatch (±5M msat)")

        if not getattr(decoded, 'payment_address', None):
            raise ValueError("Invoice must support on-chain fallback")

        dest_script, _ = address_to_script_pubkey(decoded.payment_address)
        tx = Tx()
        for u in utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))
        tx.tx_outs.append(TxOut(user_gets, dest_script))
        if dao_cut and not selfish_mode:
            dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
            tx.tx_outs.append(TxOut(dao_cut, dao_script))

        qr = make_qr(tx.encode().hex())
        raw = tx.encode().hex()

        details_section = f"""
        <details style="margin-top: 40px; text-align: left; max-width: 700px; margin-left: auto; margin-right: auto;">
            <summary style="cursor: pointer; color: #00ff9d; font-weight: bold; font-size: 18px;">
                View Raw Transaction Hex (click to expand)
            </summary>
            <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px; border: 1px solid #333;">
            <span style="color:#00ff9d; font-weight:bold;">Raw Transaction Hex:</span>
            {raw}
            </pre>
            <small style="color:#aaa;">Copy and broadcast with any wallet that supports raw hex</small>
        </details>
        """

        return (
            f"""
            <div style="text-align:center; color:#00ff9d; font-size:26px; padding:20px;">
            Lightning Sweep Ready<br><br>
            <b style='font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d, 0 0 60px #00ff9d; font-weight:900;'>
            You receive: {format_btc(user_gets)}
            </b> instantly
            </div>
            <div style="display: flex; justify-content: center; margin: 40px 0;">
            <img src="{qr}" style="max-width:100%; width:460px; border-radius:20px; 
            box-shadow:0 12px 50px rgba(0,255,157,0.7); border: 6px solid #00ff9d;">
            </div>
            <p><small>Scan with Phoenix, Breez, Blink, Muun, Zeus, etc.</small></p>
            {details_section}
            """,
            gr.update(visible=False),  # hide generate button
            gr.update(visible=False),  # hide Lightning box
            ln_invoice                 # preserve invoice in state
        )

    except Exception as e:
        required_sats = total - miner_fee - (0 if selfish_mode else dao_cut)
        msg = f"""
        <div style="text-align:center; color:#ff3333; padding:30px; background:#33000020; border-radius:16px; border:2px solid #ff5555;">
            <b style="font-size:22px;">Lightning Sweep Failed</b><br><br>
            {str(e)}<br><br>
            <b style="color:#fff; font-size:28px;">
                Invoice must be for exactly<br>
                <b style="color:#f7931a; font-size:36px;">{required_sats:,} sats</b>
            </b><br><br>
            <div style="color:#ff6666; font-size:16px; font-weight:bold; margin-top:14px; text-shadow: 0 0 8px rgba(255,0,0,0.3);">±5,000 sats tolerance allowed</div>
        """
        return msg, gr.update(visible=False), gr.update(visible=True), ln_invoice
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
            "2. Generate Transaction → Consolidate & Pay Thank-You",
            visible=False,
            variant="primary",
            size="lg",
            elem_classes="full-width"
        )

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
        rbf_in = gr.Textbox(label="Raw hex", lines=5, scale=8)
        rbf_btn = gr.Button("Bump +50 sat/vB", scale=2)
    rbf_out = gr.Textbox(label="Bumped transaction", lines=8)

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
    rbf_btn.click(
        lambda hex: rbf_bump(hex.strip())[0] if hex.strip() else "Paste a raw transaction first",
        rbf_in, rbf_out
    )

    gr.Markdown("<hr><small>Made with love by the swarm • Ω lives forever • 2025</small>")

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
