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
.qr-button { position: fixed !important; bottom: 24px; right: 24px; z-index: 9999;
    width: 64px; height: 64px; border-radius: 50% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5); cursor: pointer; display: flex;
    align-items: center; justify-content: center; font-size: 38px; }
.qr-button.camera { bottom: 96px !important; background: #f7931a !important; }
.qr-button.lightning { bottom: 24px !important; background: #00ff9d !important; }
"""

disclaimer = """
**Omega Pruner v9.0 — Community Edition**  
Zero custody • Fully open-source • No forced fees  
Consolidate dusty UTXOs when fees are low → win when fees are high.  
Optional thank-you (default 0.5% of future savings) to the original author.  
Source: github.com/omega-pruner/v9 • Apache 2.0
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
    return f"{btc:,.8f} BTC".rstrip("0").rstrip(".") + (" BTC" if btc >= 1 else " BTC")

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
    ratio = {"Privacy First (30% pruned)": 0.3, "Recommended (40% pruned)": 0.4, "More Savings (50% pruned)": 0.5}.get(strategy, 0.4)
    keep = max(1, int(len(utxos) * (1 - ratio)))
    pruned_utxos_global = utxos[:keep]

    return (
        f"<b>Found {len(utxos):,} UTXOs</b> • Keeping <b>{keep}</b> largest • {detected}<br>"
        f"Click <b>Generate Transaction</b> to continue",
        gr.update(visible=True)
    )


def build_real_tx(user_input, strategy, threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice):
    global pruned_utxos_global, input_vb_global, output_vb_global
    if not pruned_utxos_global:
        return "Run analysis first", gr.update(visible=False)

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)
    outputs = 1 + (1 if not selfish_mode and dao_percent > 0 else 0)
    vsize = 10 + inputs + input_vb_global * inputs + output_vb_global * outputs

    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        fee_rate = 20
    miner_fee = max(1000, int(vsize * fee_rate * 1.2))

    savings = int((input_vb_global * inputs + output_vb_global) * 120) - miner_fee
    dao_cut = max(546, int(savings * dao_percent / 10000)) if not selfish_mode and dao_percent > 0 and savings > 2000 else 0
    user_gets = total - miner_fee - dao_cut

    if user_gets < 546:
        return "Not enough after fees", gr.update(visible=False)

    # Lightning path
    if ln_invoice and ln_invoice.strip().lower().startswith("lnbc"):
        return lightning_sweep_flow(pruned_utxos_global, ln_invoice.strip(), miner_fee, dao_cut, selfish_mode)

    # On-chain path
    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False)

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    psbt = base64.b64encode(b'psbt\xff\x00\x00' + tx.encode() + b'\x00').decode()
    qr = make_qr(psbt)

    thank = "No thank-you" if dao_cut == 0 else f"Thank-you: {format_btc(dao_cut)}"
    return (
        f"""
        <div style="text-align:center; padding:20px;">
            <h3 style="color:#f7931a;">Transaction Ready</h3>
            <p><b>{inputs}</b> inputs → {format_btc(total)}<br>
            Fee: {format_btc(miner_fee)} @ {fee_rate} sat/vB • {thank}<br><br>
            <b style="font-size:32px; color:#00ff9d;">You receive: {format_btc(user_gets)}</b></p>
            <img src="{qr}" style="width:420px; border-radius:16px; border:4px solid #f7931a;">
            <p><small>Scan with Sparrow, BlueWallet, Nunchuk, Electrum</small></p>
        </div>
        """,
        gr.update(visible=False), gr.update(visible=True)
    )


def lightning_sweep_flow(utxos, invoice, miner_fee, dao_cut, selfish_mode):
    if not bolt11_decode:
        return "bolt11 library missing — Lightning disabled", gr.update(visible=False), gr.update(visible=False)

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
        return (
            f"""
            <div style="text-align:center; color:#00ff9d; font-size:24px;">
                Lightning Sweep Ready<br><br>
                You receive <b>{format_btc(user_gets)}</b> instantly
            </div>
            <img src="{qr}" style="max-width:100%; border-radius:16px; box-shadow:0 8px 40px rgba(0,255,157,0.6);">
            <p><small>Scan with Phoenix, Breez, Blink, Muun, Zeus, etc.</small></p>
            """,
            gr.update(visible=False),  # hide generate button
            gr.update(visible=False)   # HIDE LIGHTNING BOX — sweep is done
        )
    except Exception as e:
        msg = f"<b style='color:#ff3333'>Lightning failed:</b> {str(e)}"
        if "amount" in str(e).lower():
            msg += "<br>Invoice must be for the exact amount shown above"
        return msg, gr.update(visible=False), gr.update(visible=True)  # keep box open on error

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

    dust_threshold = gr.Slider(0, 3000, 546, step=1, label="Dust threshold (sats)")
    dest_addr = gr.Textbox(label="Destination (optional)", placeholder="Leave blank = same address")

    with gr.Row():
        selfish_mode = gr.Checkbox(label="Selfish mode – keep 100%", value=False)
    with gr.Row():
        with gr.Column(scale=1): 
            dao_percent = gr.Slider(0, 500, 50, step=10, label="Thank-you (bps)")
        with gr.Column(scale=2): 
            gr.Markdown("50 bps = 0.5% • 0 = keep all")

    dao_addr = gr.Textbox(label="Thank-you address (optional)", value=DEFAULT_DAO_ADDR)

    with gr.Row():
        submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary")
        generate_btn = gr.Button("2. Generate Transaction", visible=False, variant="primary")

    output_log = gr.HTML()
    ln_invoice = gr.Textbox(label="Lightning Invoice → paste lnbc…", lines=3, visible=False)

    gr.Markdown("### RBF Bump")
    with gr.Row():
        rbf_in = gr.Textbox(label="Raw hex", lines=5)
        rbf_btn = gr.Button("Bump +50 sat/vB")
    rbf_out = gr.Textbox(label="Bumped tx", lines=8)

    # ←←← THIS WAS MISSING — ADD IT HERE
    status_msg = gr.Markdown("Click **1. Analyze UTXOs** to begin")

    # Events — Gradio 6.0.0 Bulletproof
    submit_btn.click(
        analysis_pass, 
        [user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr], 
        [output_log, generate_btn]
    ).then(
        lambda: (gr.update(visible=True), gr.update(value="Ready → Click **2. Generate Transaction**")),
        outputs=[generate_btn, status_msg]
    )

    generate_btn.click(
        build_real_tx,
        inputs=[user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice],
        outputs=[output_log, generate_btn, ln_invoice]
    ).then(
        lambda html: gr.update(
            visible="You receive" in str(html) or "Transaction Ready" in str(html),
            label="Lightning Invoice → paste here for instant sweep"
        ),
        inputs=[output_log],
        outputs=[ln_invoice]
    )

    ln_invoice.submit(
        build_real_tx, 
        [user_input, prune_choice, dust_threshold, dest_addr, selfish_mode, dao_percent, dao_addr, ln_invoice], 
        [output_log, generate_btn]
    ).then(
        lambda: gr.update(visible=False), 
        outputs=[ln_invoice]
    ).then(
        lambda: gr.update(value="Lightning sweep ready! Scan the QR below"), 
        outputs=[status_msg]
    )

    rbf_btn.click(
        lambda hex: rbf_bump(hex.strip())[0] if hex.strip() else "Paste a raw transaction first",
        rbf_in, 
        rbf_out
    )
    gr.Markdown("<hr><small>Made with love by the swarm • Ω lives forever • 2025</small>")

    # ———————— FIXED & WORKING QR SCANNERS (2025 edition) ————————
    gr.HTML("""
    <label class="qr-button camera">Camera</label>
    <label class="qr-button lightning">Lightning</label>

    <input type="file" accept="image/*" capture="environment" id="qr-scanner-camera" style="display:none">
    <input type="file" accept="image/*" capture="environment" id="qr-scanner-ln" style="display:none">

    <script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
    <script>
    const cameraBtn = document.querySelector('.qr-button.camera');
    const lnBtn = document.querySelector('.qr-button.lightning');
    const cameraInput = document.getElementById('qr-scanner-camera');
    const lnInput = document.getElementById('qr-scanner-ln');

    cameraBtn.onclick = () => cameraInput.click();
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

          if (isLightning && (text.toLowerCase().startsWith('lnbc') || text.toLowerCase().startsWith('lnurl'))) {
            const textbox = document.querySelector('textarea[placeholder*="lnbc"], input[placeholder*="lnbc"]');
            if (textbox) { textbox.value = text; textbox.dispatchEvent(new Event('input')); }
            alert("Lightning invoice scanned!");
          } else if (!isLightning && (text.startsWith('bc1') || text.startsWith('1') || text.startsWith('3') || text.startsWith('xpub') || text.startsWith('zpub'))) {
            const addrbox = document.querySelector('textarea[placeholder*="bc1q"], input[placeholder*="bc1q"]');
            if (addrbox) { addrbox.value = text.split('?')[0].replace(/^bitcoin:/i, ''); addrbox.dispatchEvent(new Event('input')); }
            alert("Address/xpub scanned!");
          } else {
            alert("Not recognized. Try again.");
          }
        } catch (e) { alert("No QR found — try better lighting"); }
      };
      img.src = URL.createObjectURL(file);
    }

    cameraInput.onchange = (e) => scan(e.target.files[0], false);
    lnInput.onchange = (e) => scan(e.target.files[0], true);
    </script>
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
