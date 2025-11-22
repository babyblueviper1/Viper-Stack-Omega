# app.py — Omega Pruner v9.0 — Community Edition
import gradio as gr
print(f"Gradio version: {gr.__version__}")
import requests, time, base64, qrcode, io
from dataclasses import dataclass
from typing import List

# ==============================
# Optional deps (still graceful)
# ==============================
try:
    from bolt11 import decode as bolt11_decode
except:
    bolt11_decode = None

# ==============================
# Constants
# ==============================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"   # kept only for grateful users
pruned_utxos_global = None
input_vb_global = 68
output_vb_global = 31

# ==============================
# Clean, non-cult CSS + disclaimer
# ==============================
css = """
.qr-button { position: fixed !important; bottom: 24px; right: 24px; z-index: 9999;
    width: 64px; height: 64px; border-radius: 50% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); cursor: pointer; display: flex;
    align-items: center; justify-content: center; font-size: 38px; }
.qr-button.camera { bottom: 96px !important; background: #f7931a !important; }
.qr-button.lightning { bottom: 24px !important; background: #1188ff !important; }
.big-fuel-button button { height: 80px !important; font-size: 18px !important; border-radius: 16px !important; }
"""

disclaimer = """
**Omega Pruner v9.0 — Community Edition**  
Open-source • Zero custody • No forced fees  
Consolidate dusty UTXOs when fees are low → win when fees are high.  
Optional small thank-you to the original author (default 0.5% of future savings).  
Source: github.com/omega-pruner/v9 (Apache license)
"""

# ==============================
# All the Bitcoin helpers
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
    for value in data:
        acc = ((acc << frombits) | value) & ((1 << (frombits + tobits - 1)) - 1)
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append(acc >> bits & maxv)
    if pad and bits: ret.append(acc << (tobits - bits) & maxv)
    return ret

def base58_decode(s):
    n = 0
    for c in s: n = n * 58 + BASE58_ALPHABET.index(c)
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + n.to_bytes((n.bit_length() + 7) // 8, 'big')

def address_to_script_pubkey(addr: str):
    if not addr or len(addr.strip()) < 26:  # Too short for any valid addr
        return b'\x00\x14' + b'\x00' * 20, {'input_vb': 68, 'output_vb': 31, 'type': 'P2WPKH (fallback)'}
    
    addr = addr.strip().lower()
    # Reject obvious non-addresses like xpubs
    if addr.startswith('xpub') or addr.startswith('zpub') or addr.startswith('ypub') or addr.startswith('tpub'):
        return b'\x00\x14' + b'\x00' * 20, {'input_vb': 68, 'output_vb': 31, 'type': 'SegWit (xpub fallback)'}
    
    # Your existing logic (unchanged, just indented)
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
                script = bytes([0x00, 0x14 if len(prog) == 20 else 0x20]) + bytes(prog)
                return script, {'input_vb': 68, 'output_vb': 31, 'type': 'SegWit'}
    if addr.startswith('bc1p'):
        data = [CHARSET.find(c) for c in addr[5:] if c in CHARSET]
        if data and data[0] == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {'input_vb': 57.5, 'output_vb': 43, 'type': 'Taproot'}
    
    # NEW: Graceful fallback for invalids (e.g., xpub as dest)
    return b'\x00\x14' + b'\x00' * 20, {'input_vb': 68, 'output_vb': 31, 'type': 'SegWit (fallback - provide valid dest)'}

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
    confirmed = [u for u in utxos if u.get('status', {}).get('confirmed')]
    return [u for u in confirmed if u['value'] > dust]

def fetch_all_utxos_from_xpub(xpub: str, dust: int = 546):
    """
    Pure-API xpub scanner — No bip32 lib needed. Uses Blockchain.info multiaddr + Blockstream/Mempool.
    Scans up to 200 addresses (100 receive + 100 change). Sorted descending by value.
    """
    try:
        import urllib.parse  # Built-in, zero deps
        xpub_clean = xpub.strip()
        
        # Step 1: Get addresses via Blockchain.info multiaddr (fast, handles derivation paths)
        multi_url = f"https://blockchain.info/multiaddr?active={urllib.parse.quote(xpub_clean)}&n=200&index=0"
        multi_resp = requests.get(multi_url, timeout=30)
        multi_resp.raise_for_status()
        data = multi_resp.json()
        
        if not data.get("addresses"):
            raise ValueError("No addresses derived from xpub — invalid format?")
        
        all_utxos = []
        scanned = 0
        max_scan = 200  # Safety cap
        
        for addr_info in data["addresses"]:
            if scanned >= max_scan:
                break
            addr = addr_info["address"]
            scanned += 1
            
            # Step 2: Fetch UTXOs for this addr using your existing api_get (with fallbacks)
            try:
                utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
            except:
                try:
                    utxos = api_get(f"https://mempool.space/api/address/{addr}/utxo")
                except Exception as e:
                    print(f"API fail for {addr}: {e}")  # Silent log, don't crash
                    continue
            
            confirmed = [u for u in utxos if u.get('status', {}).get('confirmed', True)]
            all_utxos.extend([u for u in confirmed if u['value'] > dust])
        
        # CRITICAL: Sort by value descending (your existing logic)
        all_utxos = sorted(all_utxos, key=lambda x: x['value'], reverse=True)
        
        return all_utxos, f"API-scanned xpub → {len(all_utxos):,} UTXOs across {scanned} addresses"
    
    except requests.exceptions.RequestException as e:
        return [], f"API scan failed (network?): {str(e)}<br>Tip: Check xpub format or try single address."
    except Exception as e:
        return [], f"xpub parse error: {str(e)}<br>Fallback: Enter a single BTC address instead."

def format_btc(sats: int) -> str:
    if sats == 0:
        return "0 sats"
    if sats < 100_000:
        return f"{sats:,} sats"
    btc = sats / 100_000_000
    if sats >= 100_000_000:  # 1 BTC+
        return f"{btc:,.4f} BTC".rstrip("0").rstrip(".")
    else:
        return f"{btc:.8f} BTC".rstrip("0").rstrip(".")
# ==============================
# Transaction Primitives
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
    sequence: int = 0xfffffffd  # RBF enabled by default
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
        return (self.amount.to_bytes(8, 'little') +
                encode_varint(len(self.script_pubkey)) + self.script_pubkey)

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
        out = [self.version.to_bytes(4, 'little'), encode_varint(len(self.tx_ins))]
        out += [i.encode() for i in self.tx_ins]
        out += [encode_varint(len(self.tx_outs))]
        out += [o.encode() for o in self.tx_outs]
        out += [self.locktime.to_bytes(4, 'little')]
        return b''.join(out)

def tx_to_psbt(tx: Tx) -> str:
    return base64.b64encode(b'psbt\xff\x00\x00' + tx.encode() + b'\x00').decode()

# ==============================
# RBF Bump
# ==============================
def varint_decode(data: bytes, pos: int):
    first = data[pos]
    pos += 1
    if first < 0xfd: return first, pos
    elif first == 0xfd: return int.from_bytes(data[pos:pos+2], 'little'), pos + 2
    elif first == 0xfe: return int.from_bytes(data[pos:pos+4], 'little'), pos + 4
    else: return int.from_bytes(data[pos:pos+8], 'little'), pos + 8

def rbf_bump(raw_hex: str, bump: int = 50):
    try:
        data = bytes.fromhex(raw_hex.strip())
        pos = 4
        vin_len, pos = varint_decode(data, pos)
        for _ in range(vin_len):
            pos += 36
            slen, pos = varint_decode(data, pos)
            pos += slen + 4
        vout_len, pos = varint_decode(data, pos)
        amount = int.from_bytes(data[pos:pos+8], 'little')
        vsize = (len(data) + 3) // 4
        extra = int(vsize * bump)
        if amount <= extra + 546:
            return "Not enough for bump", raw_hex
        new_amount = amount - extra
        tx = bytearray(data)
        tx[pos:pos+8] = new_amount.to_bytes(8, 'little')
        ipos = 5
        for _ in range(vin_len):
            ipos += 36
            slen, ipos = varint_decode(data, ipos-1 if ipos > len(data) else ipos)
            ipos += slen + 4
            tx[ipos-4:ipos] = b'\xfd\xff\xff\xff'
        return tx.hex(), f"Bumped +{bump} sat/vB (+{extra:,} sats fee)"
    except Exception as e:
        return f"Error: {e}", raw_hex

# ==============================
# NEW: Local QR code generation (privacy!)
# ==============================
def make_qr(data: str) -> str:
    img = qrcode.make(data, box_size=10, border=4)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ==============================
# Core logic — now with all requested upgrades
# ==============================
def analysis_pass(addr, strategy, threshold, dest, sweep, invoice, xpub):
    global pruned_utxos_global, input_vb_global, output_vb_global
    pruned_utxos_global = None
    input_vb_global = 68
    output_vb_global = 31

    if xpub and xpub.strip():
        utxos, info = fetch_all_utxos_from_xpub(xpub.strip(), threshold)
        if isinstance(info, str) and ("error" in info.lower() or "failed" in info.lower()):
            return info.replace("\n", "<br>"), gr.update(visible=True)
    else:
        if not addr or not addr.strip():
            return "Enter a Bitcoin address or xpub", gr.update(visible=True)
        utxos = get_utxos(addr.strip(), threshold)

    if not utxos:
        return "No confirmed UTXOs above dust threshold.<br>Lower threshold or wait for confirmations.", gr.update(visible=True)

    # SORT BY VALUE DESCENDING — CRITICAL FIX
    utxos = sorted(utxos, key=lambda x: x['value'], reverse=True)

    # ──────────────────────── NEW DETECTION LOGIC (xpub-proof) ────────────────────────
    sample_addrs = [u.get('address') or addr.strip() for u in utxos[:10]]  # sample up to 10
    script_types = []
    for s in sample_addrs:
        _, info = address_to_script_pubkey(s)
        if info:
            script_types.append(info['type'])

    if script_types:
        from collections import Counter
        detected_type = Counter(script_types).most_common(1)[0][0]
    else:
        detected_type = "SegWit (safe fallback)"

    # Apply correct vBytes based on detected type
    type_to_vb = {
        'P2PKH': {'input_vb': 148, 'output_vb': 34},
        'P2SH':  {'input_vb': 91,  'output_vb': 32},
        'SegWit': {'input_vb': 68, 'output_vb': 31},
        'Taproot': {'input_vb': 57.5, 'output_vb': 43},
    }
    vb = type_to_vb.get(detected_type.split(' ')[0], {'input_vb': 68, 'output_vb': 31})
    input_vb_global = vb['input_vb']
    output_vb_global = vb['output_vb']
    # ─────────────────────────────────────────────────────────────────────────────────────

    ratio = {"Privacy First (30% pruned)": 0.3, "Recommended (40% pruned)": 0.4, "More Savings (50% pruned)": 0.5}.get(strategy, 0.4)
    keep_count = max(1, int(len(utxos) * (1 - ratio)))
    pruned_utxos_global = utxos[:keep_count]

    log = f"""
    <b>Scan complete!</b><br><br>
    Found <b>{len(utxos):,}</b> UTXOs • Keeping the <b>{keep_count:,}</b> largest<br>
    Strategy: <b>{strategy.split(' (')[0]}</b><br>
    Format: <b>{detected_type}</b><br><br>
    Click below to consolidate
    """
    # ─────────────────────────────────────────────────────────────────────────────────────
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # THIS IS THE LINE YOU WERE LOOKING FOR — IT'S NOW USING "detected_type"
    # ─────────────────────────────────────────────────────────────────────────────────────
    return log, gr.update(visible=True), detected_type

def build_real_tx(addr, strategy, threshold, dest, sweep, invoice, xpub,
                  dao_percent, selfish_mode, dao_addr):
    global pruned_utxos_global, input_vb_global, output_vb_global
    if not pruned_utxos_global:
        return "Run analysis first!", gr.update(visible=False), ""

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)

    # Live fee rate from mempool.space
    try:
        fee_rate = requests.get("https://mempool.space/api/v3/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        try:
            # backup API (rarely needed, but bulletproof)
            fee_rate = requests.get("https://bitcoinfees.net/api/v1/fees/recommended", timeout=5).json()["fastestFee"]
        except:
            # final fallback — still better than the old static 12
            fee_rate = 20
    vsize = 10 + inputs + (input_vb_global * inputs) + (output_vb_global * (2 if not selfish_mode and dao_percent > 0 else 1))
    miner_fee = max(1000, int(vsize * fee_rate * 1.15))  # +15% buffer

    future_cost = int((input_vb_global * inputs + output_vb_global) * 120)  # assume 120 sat/vB future
    savings = future_cost - miner_fee

    # DAO logic — completely optional
    dao_cut = 0
    if not selfish_mode and dao_percent > 0 and savings > 2000:
        dao_cut = max(546, int(savings * dao_percent / 10000))  # percent → bps → fraction

    user_gets = total - miner_fee - dao_cut
    if user_gets < 546:
        return "Not enough for dust limit after fees", gr.update(visible=False), ""
            
    # On-chain path
    dest_addr = (dest or addr).strip() if dest else addr.strip()
    dest_script, dest_info = address_to_script_pubkey(dest_addr)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False), ""

    # Respect destination output type for accurate vsize
    if dest_info and 'output_vb' in dest_info:
        output_vb_global = dest_info['output_vb']

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))

    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut > 0:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    raw = tx.encode().hex()
    psbt = tx_to_psbt(tx)
    qr = make_qr(psbt)

    fee_text = "No thank-you" if dao_cut == 0 else f"Thank-you: <b>{format_btc(dao_cut)}</b> ({dao_percent/100:.2f}% of savings)"

    # Lightning path
    if invoice and invoice.strip().lower().startswith("lnbc"):
        return lightning_sweep_flow(pruned_utxos_global, invoice.strip(), miner_fee, savings, dao_cut, selfish_mode)

    return (
    f"""
    <div style="text-align:center; max-width:780px; margin:0 auto; padding:20px;">
        <h3 style="color:#f7931a; margin-bottom:32px;">Transaction Ready</h3>
        <p>Consolidated <b>{inputs}</b> inputs → <b>{format_btc(total)}</b> total<br>
        Live fee rate: <b>{fee_rate}</b> sat/vB → Miner fee <b>{format_btc(miner_fee)}</b><br>
        {fee_text}<br><br>
        <span style="font-size:32px; color:#00ff9d; font-weight:bold;">
            You receive: {format_btc(user_gets)}
        </span></p>
        <div style="display:flex; justify-content:center; margin:50px 0;">
            <img src="{qr}" style="width:440px; max-width:96vw; border-radius:20px; border:5px solid #f7931a;">
        </div>
        <small>Scan with any PSBT-compatible wallet (Sparrow, BlueWallet, Nunchuk…)</small>
        <details><summary>Raw hex / PSBT</summary>
        <pre style="text-align:left; background:#000; color:#0f0; padding:15px; border-radius:8px; overflow-x:auto;">
Raw hex:  {raw}
PSBT:     {psbt}</pre></details>
    </div>
    """,
    gr.update(visible=True, interactive=False),
    ""   # ← dummy third value so .then() chain works
)

def lightning_sweep_flow(utxos, invoice: str, miner_fee: int, savings: int, dao_cut: int, selfish_mode: bool):
    if not bolt11_decode:
        return "<b style='color:#ff3333'>bolt11 library missing — Lightning disabled</b>", ""

    try:
        decoded = bolt11_decode(invoice.strip())
        total = sum(u['value'] for u in utxos)
        user_gets = total - miner_fee - (0 if selfish_mode else dao_cut)

        if user_gets < 546:
            raise ValueError("Not enough after fees")

        expected_msats = user_gets * 1000
        if abs(expected_msats - (decoded.amount_msat or 0)) > 5_000_000:
            raise ValueError(f"Invoice must be for ~{format_btc(user_gets)} (±5k sats tolerance)")

        if not getattr(decoded, 'payment_address', None):
            raise ValueError("Invoice must support on-chain fallback (Phoenix, Breez, Blink, Muun)")

        dest_script, _ = address_to_script_pubkey(decoded.payment_address)
        tx = Tx()
        for u in utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))
        tx.tx_outs.append(TxOut(user_gets, dest_script))
        if dao_cut > 0 and not selfish_mode:
            dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
            tx.tx_outs.append(TxOut(dao_cut, dao_script))

        raw = tx.encode().hex()
        qr = make_qr(raw)

        thank_you_text = "No thank-you" if dao_cut == 0 else f"Thank-you: <b>{format_btc(dao_cut)}</b>"

        return (
            f"""
            <div style="text-align:center; font-size:24px; color:#00ff9d; margin:40px 0;">
                Lightning Sweep Ready
            </div>
            You receive <b>{format_btc(user_gets)}</b> instantly on Lightning<br>
            Miner fee: <b>{format_btc(miner_fee)}</b> • Thank-you: <b>{format_btc(dao_cut)}</b><br><br>
            <div style="display:flex; justify-content:center;">
                <img src="{qr}" style="max-width:100%; border-radius:16px; box-shadow:0 8px 40px rgba(0,255,157,0.6)">
            </div>
            <small>Scan with Phoenix, Breez, Blink, Muun, Zeus, etc.</small>
            """,
            gr.update(visible=True, interactive=False),
            ""
        )

    except Exception as e:
        error_msg = f"<b style='color:#ff3333'>Lightning failed:</b> {str(e)}"
        return (
            error_msg,
            gr.update(visible=True, interactive=False),
            ""
        )

# ==============================
# Gradio UI – clean & honest
# ==============================
with gr.Blocks(title="Omega Pruner v9.0 – Community Edition") as demo:
    gr.HTML(f"<style>{css}</style>")

    gr.Markdown("# Omega Pruner v9.0")
    gr.Markdown(disclaimer)

    with gr.Row():
        with gr.Column(scale=4, min_width=300):
            user_addr = gr.Textbox(label="Bitcoin address or xpub/zpub", placeholder="bc1q… or xpub…", lines=2)
        with gr.Column(scale=3, min_width=240):
            prune_choice = gr.Dropdown(
                ["Privacy First (30% pruned)", "Recommended (40% pruned)", "More Savings (50% pruned)"],
                value="Recommended (40% pruned)", label="Pruning Strategy"
            )

    dust_threshold = gr.Slider(0, 3000, 546, step=1, label="Dust threshold (sats)")
    dest_addr = gr.Textbox(label="Destination (optional – leave blank = same address)", placeholder="bc1q…")

    with gr.Row():
        selfish_mode = gr.Checkbox(label="Selfish mode – keep 100% (no thank-you)", value=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=120):
            dao_percent = gr.Slider(0, 500, value=50, step=10, label="Optional thank-you (bps)")
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("50 bps = 0.5% · 0 = keep 100%")

    dao_addr = gr.Textbox(
        label="Thank-you address (optional)",
        placeholder="Leave default to support the original Ω author",
        value=DEFAULT_DAO_ADDR,
        interactive=True
    )

    with gr.Row():
        submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary", scale=1)
        generate_btn = gr.Button("2. Generate Transaction", visible=False, interactive=False, variant="primary", scale=1)

    # ——————— MAIN OUTPUT AREA ———————
    xpub = user_addr   # allows xpub detection while keeping single input box
    output_log = gr.HTML()

    # LIGHTNING BOX — appears right after output_log, before RBF
    ln_invoice = gr.Textbox(
        label="Lightning Invoice → paste lnbc... here to sweep instantly",
        placeholder="Will appear automatically after transaction is generated",
        lines=3,
        visible=False
    )

    # RBF TOOL
    gr.Markdown("### Stuck tx? RBF bump")
    with gr.Row():
        rbf_in = gr.Textbox(label="Raw hex", lines=5, scale=4)
        rbf_btn = gr.Button("Bump +50 sat/vB", scale=1)
    rbf_out = gr.Textbox(label="New transaction", lines=8)

    status_msg = gr.Markdown("Click **1. Analyze UTXOs** to begin", visible=True)
    dummy_state = gr.State()   # optional, harmless if unused

    # ———————————————— EVENTS ————————————————
    submit_btn.click(
        fn=analysis_pass,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr, ln_invoice, xpub],
        outputs=[output_log, generate_btn]
    ).then(
        lambda: (
            gr.update(visible=True, interactive=True),
            gr.update(value="Ready → Click **2. Generate Transaction + Pay DAO**")
        ),
        outputs=[generate_btn, status_msg]
    )

    # MAIN FLOW: Generate on-chain → auto-show Lightning box
    generate_btn.click(
        build_real_tx,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr, ln_invoice, xpub],
        outputs=[output_log, generate_btn, ln_invoice]   # ← 3 outputs!
    ).then(
        lambda log: gr.update(
            visible=True,
            label="Lightning Invoice → paste lnbc... to sweep instantly",
            placeholder="Must be for the exact amount shown above"
        ),
        inputs=output_log,
        outputs=ln_invoice
    )

    # When user PASTES invoice → instantly generate Lightning sweep transaction
    ln_invoice.submit(
        build_real_tx,   # reuse same function — it detects invoice and does Lightning path
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr, ln_invoice, xpub],
        outputs=[output_log, generate_btn]
    ).then(
        lambda: gr.update(visible=False),  # hide the box again after use
        outputs=ln_invoice
    ).then(
        lambda: gr.update(value="Lightning sweep ready! Scan QR below"),
        outputs=status_msg
    )

    # RBF
    rbf_btn.click(lambda h: rbf_bump(h)[0], rbf_in, rbf_out)

    gr.Markdown("<br><hr><small>Made better by the community • Original Ω concept by anon • 2025</small>")
    
    # QR scanners + auto-show invoice box (same excellent code from v8.7)
    gr.HTML("""
    <!-- ORANGE CAMERA BUTTON - SCAN ON-CHAIN ADDRESS -->
    <label class="qr-button camera">
      <input type="file" accept="image/*" capture="environment" id="qr-camera-omega" style="display:none">
      <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:38px;pointer-events:none;">Camera</div>
    </label>

    <!-- GREEN LIGHTNING BUTTON - SCAN LIGHTNING INVOICE -->
    <label class="qr-button lightning">
      <input type="file" accept="image/*" capture="environment" id="qr-lightning-omega" style="display:none">
      <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:38px;pointer-events:none;">Lightning</div>
    </label>

    <script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
    <script>
    // Camera → Address field
    document.getElementById('qr-camera-omega').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
          const result = await ZXing.readBarcodeFromCanvas(canvas);
          if (result.text.startsWith('bitcoin:') || result.text.startsWith('bc1') || result.text.startsWith('1') || result.text.startsWith('3')) {
            document.querySelector("#user_addr input").value = result.text.replace(/^bitcoin:/i, '').split('?')[0];
            document.querySelector("#user_addr input").dispatchEvent(new Event('input'));
            alert("Address scanned successfully!");
          } else {
            alert("Not a Bitcoin address");
          }
        } catch { alert("No QR code found — try again"); }
      };
      img.src = URL.createObjectURL(file);
    });
    // Lightning → Invoice field
    document.getElementById('qr-lightning-omega').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
          const result = await ZXing.readBarcodeFromCanvas(canvas);
          const text = result.text.trim().toLowerCase();
          if (text.startsWith('lnbc') || text.startsWith('lnurl')) {
            const invoiceBox = document.querySelector("#ln_invoice textarea") || document.querySelector("#ln_invoice input");
            if (invoiceBox) {
              invoiceBox.value = result.text;
              invoiceBox.dispatchEvent(new Event('input'));
              invoiceBox.dispatchEvent(new Event('change'));
              alert("Lightning invoice scanned & pasted!");
            }
          } else {
            alert("Not a Lightning invoice");
          }
        } catch { alert("No QR code found"); }
      };
      img.src = URL.createObjectURL(file);
    });
    </script>
    """)

if __name__ == "__main__":
    import os
    demo.queue(max_size=30)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
        allowed_paths=["static"]   # optional, only if you have a /static folder
    )
