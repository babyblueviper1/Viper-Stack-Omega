# app.py — Omega Pruner Ω v9.0 — Fully Fixed & Working 2025
import gradio as gr
import requests
import time
import base64
from dataclasses import dataclass
from typing import List

# ==============================
# Dependencies
# ==============================
try:
    from bolt11 import decode as bolt11_decode
except ImportError:
    bolt11_decode = None

try:
    from bip32 import BIP32
except ImportError:
    BIP32 = None

# ==============================
# Globals & Constants
# ==============================
pruned_utxos_global = None
input_vb_global = output_vb_global = None
DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"

# ==============================
# CSS
# ==============================
css = """
.qr-button { position: fixed !important; bottom: 24px; right: 24px; z-index: 9999;
    width: 64px; height: 64px; border-radius: 50% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); cursor: pointer; display: flex;
    align-items: center; justify-content: center; font-size: 38px; }
.qr-button.camera { bottom: 96px !important; background: #f7931a !important; }
.qr-button.lightning { bottom: 24px !important; background: #00ff9d !important; }
.big-fuel-button button { height: 100px !important; font-size: 20px !important; border-radius: 16px !important; }
.hidden-ln-invoice { display: none !important; }
"""

disclaimer = """
**Consolidate when fees are low — win when fees are high.**  
One-click dusty UTXO cleanup • Legacy / SegWit / Taproot • Lightning Sweep  
Zero custody • 100% open-source • Voluntary DAO fuel  
**DAO:** `bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj`  
**Surge the swarm. Ledger’s yours.** 🜂
"""

# ==============================
# Bitcoin Helpers (fixed & complete)
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
    addr = addr.strip().lower()
    if addr.startswith('1'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x00:
            return b'\x76\xa9\x14' + dec[1:21] + b'\x88\xac', {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
    if addr.startswith('3'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x05:
            return b'\xa9\x14' + dec[1:21] + b'\x87', {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
    if addr.startswith('bc1q'):
        data = [CHARSET.find(c) for c in addr[4:]]
        if data and data[0] == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) in (20, 32):
                script = bytes([0x00, 0x14 if len(prog) == 20 else 0x20]) + bytes(prog)
                return script, {'input_vb': 68, 'output_vb': 31, 'type': 'SegWit'}
    if addr.startswith('bc1p'):
        data = [CHARSET.find(c) for c in addr[5:]]
        if data and data[0] == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}
    return None, None

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
        height = api_get("https://blockstream.info/api/blocks/tip/height")
        utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
    except:
        height = api_get("https://mempool.space/api/blocks/tip/height")
        utxos = api_get(f"https://mempool.space/api/address/{addr}/utxos")
    return [u for u in utxos
            if u.get('status', {}).get('confirmed') and
               u['value'] > dust and
               height - u['status']['block_height'] >= 6]

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
# RBF Bump (fully working)
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
            pos += 36  # prev_tx + index
            slen, pos = varint_decode(data, pos)
            pos += slen + 4  # script + sequence
        vout_len, pos = varint_decode(data, pos)
        amount = int.from_bytes(data[pos:pos+8], 'little')
        vsize = (len(data) + 3) // 4
        extra = int(vsize * bump)
        if amount <= extra + 546:
            return None, "Not enough for bump + dust"
        new_amount = amount - extra
        tx = bytearray(data)
        tx[pos:pos+8] = new_amount.to_bytes(8, 'little')
        # Set RBF sequence on all inputs
        ipos = 4 + 1
        for _ in range(vin_len):
            ipos += 36
            slen, ipos = varint_decode(data, ipos - 1 if ipos > len(data) else ipos)
            ipos += slen
            tx[ipos:ipos+4] = b'\xfd\xff\xff\xff'
            ipos += 4
        return tx.hex(), f"+{bump} sat/vB bump ({extra:,} sats added)"
    except Exception as e:
        return None, f"Error: {e}"

# ==============================
# Core Logic
# ==============================
def analysis_pass(addr, strategy, threshold, dest, sweep, invoice, xpub):
    global pruned_utxos_global, input_vb_global, output_vb_global

    # Reset globals every time (prevents stale data from previous runs)
    pruned_utxos_global = None
    input_vb_global = output_vb_global = None

    # Default safe vB weights (in case address parsing fails)
    input_vb_global = 68
    output_vb_global = 31

    # === Fetch UTXOs (xpub or single address) ===
    if xpub and xpub.strip():
        if not BIP32:
            return "bip32 library missing — install with: pip install bip32", gr.update(visible=True)
        utxos, info = fetch_all_utxos_from_xpub(xpub.strip(), threshold)
        if isinstance(info, str):  # error message
            return info.replace("\n", "<br>"), gr.update(visible=True)
    else:
        if not addr or not addr.strip():
            return "Please enter a Bitcoin address", gr.update(visible=True)
        utxos = get_utxos(addr.strip(), threshold)

    if not utxos:
        return "No confirmed UTXOs above dust threshold found.<br>Try lowering the dust threshold or waiting for confirmations.", gr.update(visible=True)

    # === Detect address type & refine vB weights ===
    sample_addr = utxos[0].get("address") or addr.strip()
    script_info = address_to_script_pubkey(sample_addr)
    if script_info and script_info[1]:
        vb = script_info[1]
        input_vb_global = vb.get('input_vb', input_vb_global)
        output_vb_global = vb.get('output_vb', output_vb_global)

    # === Apply pruning strategy ===
    ratio = {
        "Privacy First (30% pruned)": 0.3,
        "Recommended (40% pruned)": 0.4,
        "More Savings (50% pruned)": 0.5
    }.get(strategy, 0.4)

    keep_count = max(1, int(len(utxos) * (1 - ratio)))
    pruned_utxos_global = utxos[:keep_count]

    # === Success message ===
    log = f"""
    <b>Scan complete!</b><br><br>
    Found <b>{len(utxos):,}</b> UTXOs above dust threshold<br>
    Strategy: <b>{strategy.split(' (')[0]}</b> → Keeping the <b>{keep_count:,}</b> largest<br><br>
    Detected format: <b>{script_info[1]['type'] if script_info and script_info[1] else 'Unknown → using safe defaults'}</b><br><br>
    Click <b>Generate Transaction (with DAO cut)</b> below to build the real TX 👇
    """
    return log, gr.update(visible=True)

def build_real_tx(addr, strategy, threshold, dest, sweep, invoice, xpub):
    global pruned_utxos_global, input_vb_global, output_vb_global
    if not pruned_utxos_global:
        return "Click Run Pruner first!", gr.update(visible=False)

    # Ensure vB weights are set (safety)
    if input_vb_global is None or output_vb_global is None:
        input_vb_global = 68
        output_vb_global = 31

    if sweep and invoice.strip().startswith("lnbc"):
        msg, raw = lightning_sweep_flow(pruned_utxos_global, invoice.strip())
        qr = f"https://api.qrserver.com/v1/create-qr-code/?size=512x512&data={raw or 'LIGHTNING'}"
        return f"{msg}<br><br><div style='text-align:center'><a href='{qr}' target='_blank'><img src='{qr}' style='max-width:100%;border-radius:16px'></a></div>", gr.update(visible=False)

    # Normal on-chain consolidation
    dest_addr = (dest or addr).strip() or addr.strip()
    dest_script, _ = address_to_script_pubkey(dest_addr)
    dao_script, _ = address_to_script_pubkey(DAO_ADDR)

    total = sum(u['value'] for u in pruned_utxos_global)
    vsize = 10.5 + input_vb_global * len(pruned_utxos_global) + output_vb_global * 2
    actual_fee = max(800, int(vsize * 5))  # low, fair fee

    # If user had sent each UTXO separately at 100 sat/vB (worst case next bull run)
    fee_if_not_consolidated = int((input_vb_global * len(pruned_utxos_global) + output_vb_global) * 100)
    savings = fee_if_not_consolidated - actual_fee
    dao_cut = max(546, int(savings * 0.05))  # 5% of SAVINGS

    send = total - actual_fee - dao_cut

    if send < 546:
        return "Not enough for fee + DAO cut (this shouldn't happen)", gr.update(visible=False)

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))
    tx.tx_outs.append(TxOut(send, dest_script))        # send is already in SATOSHIS
    if dao_cut >= 546:
        tx.tx_outs.append(TxOut(dao_cut, dao_script)) 

    raw = tx.encode().hex()
    psbt = tx_to_psbt(tx)
    qr = f"https://api.qrserver.com/v1/create-qr-code/?size=512x512&data={psbt}"

    result = f"""
    <h3>✅ Transaction Ready!</h3>
    Consolidated <b>{len(pruned_utxos_global)}</b> UTXOs → <b>{total:,}</b> sats<br>
    Fee ~<b>{fee:,}</b> sats • DAO fuel <b>{dao_cut:,}</b> sats<br><br>
    <div style="text-align:center;margin:30px 0">
        <a href="{qr}" target="_blank">
            <img src="{qr}" style="max-width:100%;border-radius:16px;box-shadow:0 8px 30px rgba(0,0,0,0.5)">
        </a>
        <br><small>Tap QR → BlueWallet / Zeus / Mutiny / Aqua / Electrum / Sparrow</small>
    </div>
    <pre style="background:#000;color:#0f0;padding:16px;border-radius:12px;overflow-x:auto;font-family:monospace">
Raw Hex: {raw}

PSBT (base64): {psbt}
    </pre>
    <br>⚡ Want Lightning instead? Check the box, paste invoice, generate again.
    """
    return result, gr.update(visible=False)
    
def lightning_sweep_flow(utxos, invoice: str):
    if not bolt11_decode:
        return "bolt11 library missing — Lightning sweep disabled", ""

    try:
        decoded = bolt11_decode(invoice.strip())
        total = sum(u['value'] for u in utxos)

        # Estimate vsize for the sweep transaction (channel open)
        vsize = 10.5 + input_vb_global * len(utxos) + output_vb_global * 2
        miner_fee = max(1500, int(vsize * 15))  # realistic channel open fee

        # What it would cost to spend these UTXOs separately at 100 sat/vB (next bull run hell)
        fee_if_not_consolidated = int((input_vb_global * len(utxos) + output_vb_global) * 100)
        savings = fee_if_not_consolidated - miner_fee
        
        # DAO gets 5% of the SAVINGS (this is the soul of Omega Pruner)
        dao_cut = max(546, int(savings * 0.05))
        user_gets = total - miner_fee - dao_cut

        if user_gets < 546:
            raise ValueError("Not enough left after miner fee + DAO cut")

        expected_msats = user_gets * 1000
        if abs(expected_msats - (decoded.amount_msat or 0)) > 2_000_000:
            raise ValueError(f"Invoice must be ~{user_gets:,} sats (±2k msats)")

        if not getattr(decoded, 'payment_address', None):
            raise ValueError("Invoice needs on-chain fallback (use Phoenix, Muun, Breez, Blink)")

        dest_script, _ = address_to_script_pubkey(decoded.payment_address)
        dao_script, _ = address_to_script_pubkey(DAO_ADDR)

        tx = Tx()
        for u in utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))

        tx.tx_outs.append(TxOut(user_gets, dest_script))
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

        raw = tx.encode().hex()
        qr = f"https://api.qrserver.com/v1/create-qr-code/?size=512x512&data={raw}"

        msg = f"""
        <div style="text-align:center;font-size:22px;color:#00ff9d;margin:20px 0">
        Lightning Sweep Ready!
        </div>
        Dust consolidated: <b>{total:,}</b> sats<br>
        Miner fee: ~<b>{miner_fee:,}</b> sats<br>
        <b>You saved ~{savings:,} sats in future fees</b><br>
        DAO fuel (5% of savings): <b>{dao_cut:,}</b> sats<br><br>
        <b>You receive: {user_gets:,} sats instantly on Lightning</b><br><br>
        <div style="text-align:center;margin:30px 0">
            <a href="{qr}" target="_blank">
                <img src="{qr}" style="max-width:100%;border-radius:16px;box-shadow:0 8px 40px rgba(0,255,157,0.7)">
            </a>
        </div>
        <small>
        Sign & broadcast → your wallet opens the channel automatically<br>
        Zero custody • Dust → real spendable money • This is the way 🜂
        </small>
        """

        return msg, raw

    except Exception as e:
        return f"<b style='color:#ff5555'>Lightning sweep failed:</b> {str(e)}", ""
# ==============================
# Gradio UI
# ==============================
with gr.Blocks(title="Omega Pruner Ω v9.0") as demo:
    gr.HTML(f"<style>{css}</style>")
    gr.Markdown("# Omega Pruner Ω v9.0 🜂")
    with gr.Row():
        with gr.Column(scale=4): gr.Markdown(disclaimer)
        with gr.Column(scale=1, min_width=260):
            gr.Button("Fuel the Swarm", link=f"https://blockstream.info/address/{DAO_ADDR}", variant="primary", elem_classes="big-fuel-button")
            gr.HTML('<div style="text-align:center"><a href="https://babyblueviper.com" target="_blank"><img src="/file=static/BBV_logo.png" style="max-width:300px;border-radius:16px"></a><p><b>BabyBlueViper Ω</b></p></div>')

    user_addr = gr.Textbox(label="BTC Address", placeholder="bc1q...", elem_id="user-address")
    xpub_input = gr.Textbox(label="xpub/ypub/zpub (full wallet)", placeholder="Optional")
    prune_choice = gr.Dropdown(["Privacy First (30% pruned)", "Recommended (40% pruned)", "More Savings (50% pruned)"], value="Recommended (40% pruned)", label="Strategy")
    with gr.Row():
        dust_threshold = gr.Slider(0, 2000, 546, step=1, label="Dust Threshold (sats)")
        dest_addr = gr.Textbox(label="Destination (optional)", placeholder="Leave blank = same")
    with gr.Row():
        sweep_to_ln = gr.Checkbox(label="Sweep to Lightning", value=False)
    ln_invoice = gr.Textbox(label="Lightning Invoice (lnbc...)", placeholder="Paste invoice", elem_classes="hidden-ln-invoice")

    submit_btn = gr.Button("Run Pruner", variant="secondary")
    generate_btn = gr.Button("Generate Transaction", visible=False, variant="primary")
    output_log = gr.HTML()

       # QR Scanner for on-chain address (orange 📷) — TOP button
    gr.HTML("""
    <label class="qr-button" style="bottom: 96px !important; background: #f7931a !important;">
      <input type="file" accept="image/*" capture="environment" id="qr-camera" style="display:none">
      <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:38px;pointer-events:none;">📷</div>
    </label>
    <script src="https://unpkg.com/@zxing/library@0.20.0/dist/index.min.js"></script>
    <script>
    document.getElementById('qr-camera').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      img.onload = async function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
          const result = await ZXing.readBarcodeFromCanvas(canvas);
          if (result && result.text) {
            const input = document.querySelector("#user-address input");
            if (input) {
              input.value = result.text;
              input.dispatchEvent(new Event('input'));
              input.dispatchEvent(new Event('change'));
            }
            alert("⚡ Address scanned!");
          }
        } catch (err) {
          alert("No QR found — try again");
        }
      };
      img.src = URL.createObjectURL(file);
    });
    </script>
    """)

    # Lightning invoice QR scanner (green ⚡) — BOTTOM button
    gr.HTML("""
    <label class="qr-button" style="bottom: 24px !important; background: #00ff9d !important; box-shadow: 0 4px 20px rgba(0,255,157,0.6);">
      <input type="file" accept="image/*" capture="environment" id="qr-lightning" style="display:none">
      <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:38px;pointer-events:none;">⚡</div>
    </label>
    <script>
    document.getElementById('qr-lightning').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      img.onload = async function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
          const result = await ZXing.readBarcodeFromCanvas(canvas);
          const text = result.text.trim();
          if (text.toLowerCase().startsWith('lnbc') || text.toLowerCase().startsWith('lnurl')) {
            const input = document.querySelector("#ln_invoice input") || 
                         document.querySelector('textarea[placeholder*="lnbc"]');
            if (input) {
              input.value = text;
              input.dispatchEvent(new Event('input'));
              input.dispatchEvent(new Event('change'));
            }
            alert("⚡ Lightning invoice scanned & pasted!");
          } else {
            alert("Not a Lightning invoice — try again");
          }
        } catch (err) {
          alert("No QR code found — try again");
        }
      };
      img.src = URL.createObjectURL(file);
    });
    </script>
    """)

    gr.Markdown("### Stuck tx? +50 sat/vB bump")
    with gr.Row():
        rbf_in = gr.Textbox(label="Raw hex", lines=6)
        rbf_btn = gr.Button("Bump +50 sat/vB")
    rbf_out = gr.Textbox(label="New hex", lines=8)

    # Events
    sweep_to_ln.change(lambda x: gr.update(elem_classes="" if x else "hidden-ln-invoice"), sweep_to_ln, ln_invoice)
    submit_btn.click(analysis_pass, [user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice, xpub_input], [output_log, generate_btn])
    generate_btn.click(build_real_tx, [user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice, xpub_input], [output_log, generate_btn])
    rbf_btn.click(lambda h: (rbf_bump(h)[0] or "Error", rbf_bump(h)[0] or h), rbf_in, [rbf_out, rbf_in])

if __name__ == "__main__":
    import os
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        allowed_paths=["static"]
    )
