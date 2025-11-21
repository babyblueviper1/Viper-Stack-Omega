import gradio as gr
import requests
import os                  
from dataclasses import dataclass
from typing import List, Union

# For Lightning invoice decoding
try:
    from bolt11 import decode as bolt11_decode
except ImportError:
    # If not installed, Lightning sweep will gracefully fail
    bolt11_decode = None

# Global state for two-step flow
pruned_utxos_global = None
input_vb_global = None
output_vb_global = None

# ==============================
# CSS + Disclaimer
# ==============================
css = """
.qr-button { 
    position: fixed !important;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    width: 64px;
    height: 64px;
    background: #f7931a !important;
    border-radius: 50% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
}
.big-fuel-button button {
    height: 100px !important;
    font-size: 20px !important;
    border-radius: 16px !important;
}

/* ← THIS IS THE MAGIC THAT KILLS THE EMPTY SPACE ← */
#output_text {
    min-height: 80px !important;
}
#output_text .textbox,
#output_text textarea {
    min-height: 80px !important;
    height: auto !important;
    max-height: 70vh !important;
    overflow-y: auto !important;
    resize: none !important;   /* stops manual dragging */
}

/* Optional: remove extra bottom margin in the whole form */
.container > .form {
    gap: 8px !important;
}

"""

disclaimer = """
**Consolidate when fees are low → win when fees are high.**  
Pay a few thousand sats today… or 10–20× more next cycle. This is fee insurance.

**One-click dusty wallet cleanup**  
• Paste any address (legacy · SegWit · Taproot)  
• Get a real, RBF-ready raw TX hex in <15 seconds  
• **New: Sweep to Lightning ⚡ — turn dead dust into spendable sats instantly**  
• Sign & broadcast with your own wallet — zero custody

**Stuck in the mempool?**  
Scroll down → paste raw hex → +50 sat/vB bump in one click. (Repeatable. Free)

100% open-source • non-custodial • voluntary “Fuel the Swarm” donations

**DAO address** bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj  
Every sat pays for maintenance + future features. Thank you 🜂

[**GitHub**](https://github.com/babyblueviper1/Viper-Stack-Omega) •[**babyblueviper.com**](https://babyblueviper.com) • Apache 2.0  
**Surge the swarm. Ledger’s yours.**

(Tap 📷 / ⚡ buttons to scan or upload QR)
"""

with gr.Blocks(css=css, title="Omega Pruner Ω v8.5 — Mobile + QR + Lightning 🜂") as demo:

    gr.Markdown("# Omega Pruner Ω v8.5 — Live 🜂")

    with gr.Row():
        with gr.Column(scale=4): gr.Markdown(disclaimer)
        with gr.Column(scale=1, min_width=260):
            gr.Button("⚡ Fuel the Swarm", variant="primary",
                      link="https://blockstream.info/address/bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj",
                      elem_classes="big-fuel-button")
            # ←←← YOUR LOGO UNDER THE BUTTON ←←←
            gr.HTML("""
            <div style="text-align: center; margin-top: 20px; padding-bottom: 20px;">
              <img src="/file=static/BBV_logo.png" 
                   alt="BabyBlueViper" 
                   style="width: 100%; max-width: 280px; height: auto; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,123,255,0.4);">
              <p style="margin: 15px 0 0; font-size: 14px; color: #f7931a; font-weight: bold; letter-spacing: 1px;">
                BabyBlueViper Ω
              </p>
              <p style="margin: 5px 0 0; font-size: 11px; color: #888;">
                Surge the swarm.
              </p>
            </div>
            """)

    with gr.Row():
        user_addr = gr.Textbox(label="Your BTC Address", placeholder="bc1q...", elem_id="user-address")
        prune_choice = gr.Dropdown(
            choices=["Conservative (70/30, Low Risk)", "Efficient (60/40, Default)", "Aggressive (50/50, Max Savings)"],
            value="Efficient (60/40, Default)", label="Prune Strategy"
        )
    with gr.Row():
        dust_threshold = gr.Slider(0, 2000, 546, step=1, label="Dust Threshold (sats)")
        dest_addr = gr.Textbox(label="Destination (optional)", placeholder="Leave blank = same address")

    submit_btn = gr.Button("Run Pruner", variant="secondary")
    output_text = gr.Textbox(label="Log", lines=7, max_lines=50)  # starts tiny, expands smoothly
    raw_tx_text = gr.Textbox(label="Unsigned Raw TX Hex", lines=12, visible=False)
    generate_btn = gr.Button("Generate Real TX Hex (with DAO cut)", visible=False)

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

    # Lightning
    with gr.Row():
        sweep_to_ln = gr.Checkbox(label="Sweep to Lightning ⚡ (turn dust into spendable balance)", value=False)
    with gr.Row():
        ln_invoice = gr.Textbox(label="Lightning Invoice (lnbc...)", placeholder="Paste invoice from Phoenix, Breez, Muun, etc.", visible=False)

    sweep_to_ln.change(fn=lambda x: gr.update(visible=x), inputs=sweep_to_ln, outputs=ln_invoice)

    # RBF
    gr.Markdown("### 🆙 Stuck tx? Paste hex → +50 sat/vB bump (repeatable, free)")
    with gr.Row():
        rbf_input = gr.Textbox(label="Stuck raw hex", lines=8, placeholder="01000000...")
        rbf_btn = gr.Button("Bump +50 sat/vB", variant="primary")
    rbf_output = gr.Textbox(label="New RBF hex", lines=10)

    # PWA
    gr.HTML("""
    <link rel="manifest" href="/manifest.json">
    <link rel="apple-touch-icon" href="/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#f7931a">
    """)

    # ==============================
    # Bitcoin Helpers
    # ==============================
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    dao_cut_addr = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"

    def bech32_polymod(values):
        GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
        chk = 1
        for v in values:
            b = (chk >> 25)
            chk = (chk & 0x1ffffff) << 5 ^ v
            for i in range(5):
                chk ^= GEN[i] if ((b >> i) & 1) else 0
        return chk

    def bech32_hrp_expand(s):
        return [ord(x) >> 5 for x in s] + [0] + [ord(x) & 31 for x in s]

    def bech32_verify_checksum(hrp, data):
        return bech32_polymod(bech32_hrp_expand(hrp) + data) == 1

    def bech32m_verify_checksum(hrp, data):
        return bech32_polymod(bech32_hrp_expand(hrp) + data) == 0x2bc830a3

    def bech32_decode(addr):
        if ' ' in addr or len(addr) < 8: return None, None
        pos = addr.rfind('1')
        if pos < 1 or pos + 7 > len(addr) or len(addr) > 90: return None, None
        hrp = addr[:pos]
        if not all(ord(c) < 128 for c in addr): return None, None
        data = [CHARSET.find(c) for c in addr[pos+1:]]
        if -1 in data: return None, None
        valid = (addr[pos+1] == 'q' and bech32_verify_checksum(hrp, data)) or (addr[pos+1] == 'p' and bech32m_verify_checksum(hrp, data))
        if not valid: return None, None
        return hrp, data[:-6]

    def convertbits(data, frombits, tobits, pad=True):
        acc = bits = 0
        ret = []
        maxv = (1 << tobits) - 1
        max_acc = (1 << (frombits + tobits - 1)) - 1
        for value in data:
            if value < 0 or (value >> frombits): return None
            acc = ((acc << frombits) | value) & max_acc
            bits += frombits
            while bits >= tobits:
                bits -= tobits
                ret.append((acc >> bits) & maxv)
        if pad and bits: ret.append((acc << (tobits - bits)) & maxv)
        elif bits >= frombits or ((acc << (tobits - bits)) & maxv): return None
        return ret

    def base58_decode(s):
        n = 0
        for c in s: n = n * 58 + BASE58_ALPHABET.index(c)
        leading_zeros = len(s) - len(s.lstrip('1'))
        bytes_out = n.to_bytes((n.bit_length() + 7) // 8, 'big')
        return b'\x00' * leading_zeros + bytes_out

    def address_to_script_pubkey(addr):
        if not addr or not addr.strip(): return None
        addr = addr.strip()

        if addr.startswith('1'):
            try:
                decoded = base58_decode(addr)
                if len(decoded) == 25 and decoded[0] == 0x00:
                    payload = decoded[1:21]
                    script = bytes([0x76, 0xa9, 0x14]) + payload + bytes([0x88, 0xac])
                    return script, {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
            except: 
                return None

        if addr.startswith('3'):
            try:
                decoded = base58_decode(addr)
                if len(decoded) == 25 and decoded[0] == 0x05:
                    payload = decoded[1:21]
                    script = bytes([0xa9, 0x14]) + payload + bytes([0x87])
                    # ← CORRECT vB weights for P2SH
                    return script, {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
            except:
                return None
            
        if addr.startswith('bc1q'):
            hrp, data = bech32_decode(addr)
            if hrp == 'bc' and data and data[0] == 0:
                prog = convertbits(data[1:], 5, 8, False)
                if prog and len(prog) in (20, 32):
                    length = 0x14 if len(prog) == 20 else 0x20
                    script = bytes([0x00, length]) + bytes(prog)
                    return script, {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WSH/P2WPKH'}
            return None

        if addr.startswith('bc1p'):
            hrp, data = bech32_decode(addr)
            if hrp == 'bc' and data and data[0] == 1:
                prog = convertbits(data[1:], 5, 8, False)
                if prog and len(prog) == 32:
                    script = bytes([0x51, 0x20]) + bytes(prog)
                    return script, {'input_vb': 57.25, 'output_vb': 43, 'type': 'P2TR'}
        return None

    def api_get(url, timeout=30, retries=3):
        for i in range(retries):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                return r
            except Exception as e:
                print(f"API retry {i+1}/{retries} {url}: {e}")
                if i < retries-1: time.sleep(2**i)
        raise Exception(f"Failed {url}")

    def get_utxos(addr, dust_threshold=546, current_height=None):
        try:
            if current_height is None:
                current_height = api_get('https://blockstream.info/api/blocks/tip/height').json()
            raw = api_get(f'https://blockstream.info/api/address/{addr}/utxo').json()
        except:
            try:
                if current_height is None:
                    current_height = api_get('https://mempool.space/api/blocks/tip/height').json()
                raw = api_get(f'https://mempool.space/api/address/{addr}/utxos').json()
            except:
                return [], None

        utxos = []
        for u in raw:
            if u['status']['confirmed']:
                confs = current_height - u['status']['block_height'] + 1
                if confs > 6 and u['value'] > dust_threshold:
                    utxos.append({
                        'txid': u['txid'],
                        'vout': u['vout'],
                        'amount': u['value'] / 1e8,
                        'confs': confs
                    })
        utxos.sort(key=lambda x: x['amount'], reverse=True)
        return utxos, current_height

    # TX primitives
    def encode_int(i, nbytes, encoding='little'):
        return i.to_bytes(nbytes, encoding)

    def encode_varint(i):
        if i < 0xfd: return bytes([i])
        if i < 0x10000: return b'\xfd' + encode_int(i, 2)
        if i < 0x100000000: return b'\xfe' + encode_int(i, 4)
        return b'\xff' + encode_int(i, 8)

    def varint_decode(data: bytes, pos: int):
        first = data[pos]
        pos += 1
        if first < 0xfd: return first, pos
        elif first == 0xfd: return int.from_bytes(data[pos:pos+2], 'little'), pos + 2
        elif first == 0xfe: return int.from_bytes(data[pos:pos+4], 'little'), pos + 4
        else: return int.from_bytes(data[pos:pos+8], 'little'), pos + 8

    @dataclass
    class Script:
        cmds: List[Union[int, bytes]] = None
        def __post_init__(self): self.cmds = self.cmds or []
        def encode(self):
            out = []
            for cmd in self.cmds:
                if isinstance(cmd, int):
                    out.append(encode_int(cmd, 1))
                else:
                    length = len(cmd)
                    if length < 75: out.append(encode_int(length, 1))
                    elif length < 256: out += [b'\x4c', encode_int(length, 1)]
                    elif length < 65536: out += [b'\x4d', encode_int(length, 2)]
                    else: raise ValueError("Script too long")
                    out.append(cmd)
            result = b''.join(out)
            return encode_varint(len(result)) + result

    @dataclass
    class TxIn:
        prev_tx: bytes
        prev_index: int
        script_sig: Script = None
        sequence: int = 0xfffffffd
        def __post_init__(self): self.script_sig = self.script_sig or Script([])
        def encode(self):
            return self.prev_tx + encode_int(self.prev_index, 4) + self.script_sig.encode() + encode_int(self.sequence, 4)

    @dataclass
    class TxOut:
        amount: int
        script_pubkey: bytes = b''
        def encode(self):
            return encode_int(self.amount, 8) + encode_varint(len(self.script_pubkey)) + self.script_pubkey

    @dataclass
    class Tx:
        version: int = 1
        tx_ins: List[TxIn] = None
        tx_outs: List[TxOut] = None
        locktime: int = 0
        def __post_init__(self):
            self.tx_ins = self.tx_ins or []
            self.tx_outs = self.tx_outs or []
        def encode(self):
            out = [encode_int(self.version, 4), encode_varint(len(self.tx_ins))]
            out += [i.encode() for i in self.tx_ins]
            out += [encode_varint(len(self.tx_outs))]
            out += [o.encode() for o in self.tx_outs]
            out += [encode_int(self.locktime, 4)]
            return b''.join(out)

        @staticmethod
        def decode(data: bytes):
            pos = 0
            version = int.from_bytes(data[pos:pos+4], 'little'); pos += 4
            vin_len, pos = varint_decode(data, pos)
            tx_ins = []
            for _ in range(vin_len):
                prev_tx = data[pos:pos+32][::-1]; pos += 32
                prev_index = int.from_bytes(data[pos:pos+4], 'little'); pos += 4
                script_len, pos = varint_decode(data, pos); pos += script_len
                sequence = int.from_bytes(data[pos:pos+4], 'little'); pos += 4
                tx_ins.append(TxIn(prev_tx, prev_index, sequence=sequence))
            vout_len, pos = varint_decode(data, pos)
            tx_outs = []
            for _ in range(vout_len):
                amount = int.from_bytes(data[pos:pos+8], 'little'); pos += 8
                script_len, pos = varint_decode(data, pos)
                script_pubkey = data[pos:pos+script_len]; pos += script_len
                tx_outs.append(TxOut(amount, script_pubkey))
            locktime = int.from_bytes(data[pos:pos+4], 'little')
            return Tx(version=version, tx_ins=tx_ins, tx_outs=tx_outs, locktime=locktime)

    def rbf_bump(raw_hex, bump=50):
        try:
            tx = Tx.decode(bytes.fromhex(raw_hex))
            vsize = len(tx.encode()) // 4
            extra = int(vsize * bump)
            if tx.tx_outs[0].amount <= extra + 546:
                return None, "Not enough for bump + dust"
            tx.tx_outs[0].amount -= extra
            for txin in tx.tx_ins: txin.sequence = 0xfffffffd
            return tx.encode().hex(), f"RBF +{bump} sat/vB (+{extra:,} sats)"
        except Exception as e:
            return None, f"Error: {e}"

    def lightning_sweep_flow(pruned_utxos, invoice: str, input_vb, output_vb):
        if not bolt11_decode:
            return "Error: bolt11 library missing — Lightning sweep disabled", ""

        try:
            decoded = bolt11_decode(invoice)
            original_msats = decoded.amount_msat or 0

            est_vb = 10.5 + input_vb * len(pruned_utxos) + output_vb * 2 + 50
            miner_fee_sats = int(est_vb * 15)

            total_in_sats = sum(int(u['amount'] * 1e8) for u in pruned_utxos)
            dao_cut_sats = max(546, int(total_in_sats * 0.05))   # 5% DAO cut
            user_receive_sats = total_in_sats - miner_fee_sats - dao_cut_sats

            if user_receive_sats < 546:
                raise ValueError("Not enough left after miner fee + DAO cut")

            expected_msats = user_receive_sats * 1000
            if abs(expected_msats - original_msats) > 2_000_000:
                raise ValueError(f"Invoice should be ~{user_receive_sats:,} sats (±2k msats)")

            if not getattr(decoded, 'payment_address', None):
                raise ValueError("Invoice needs on-chain fallback address")

            dest_script, _ = address_to_script_pubkey(decoded.payment_address)
            dao_script, _ = address_to_script_pubkey(dao_cut_addr)

            tx = Tx(tx_ins=[], tx_outs=[])
            for u in pruned_utxos:
                tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))

            tx.tx_outs.append(TxOut(user_receive_sats * 100_000_000, dest_script))
            tx.tx_outs.append(TxOut(dao_cut_sats * 100_000_000, dao_script))

            raw_hex = tx.encode().hex()

            # ← THIS MUST BE INDENTED UNDER try:
            result_text = (
                "⚡ Lightning Sweep Ready! ⚡\n\n"
                f"Total dust consolidated: {total_in_sats:,} sats\n"
                f"Miner channel-open fee: ~{miner_fee_sats:,} sats\n"
                f"DAO fuel cut (5%): {dao_cut_sats:,} sats → keeps Omega alive 🜂\n"
                f"You receive: {user_receive_sats:,} sats instantly spendable on Lightning!\n\n"
                "Sign & broadcast this TX → your wallet opens the channel automatically.\n"
                "Zero custody. Dust becomes real money.\n\n"
                "Surge the swarm. Ledger’s yours. 🜂"
            )

            return result_text, raw_hex

        except Exception as e:
            return f"Lightning sweep failed: {e}\nTip: Use Phoenix, Breez, or Muun for invoices with on-chain fallback.", ""

    def main_flow(user_addr, prune_choice, dest_addr, confirm_proceed, dust_threshold=546):
        output_parts = ["Omega Pruner Ω v8.4 — Live 🜂\n"]
        if not user_addr or not user_addr.strip():
            return "\n".join(output_parts + ["No address"]), ""

        try:
            _, vb = address_to_script_pubkey(user_addr.strip())
            input_vb = vb['input_vb']
            output_vb = vb['output_vb']
            output_parts.append(f"Address valid: {user_addr.strip()}\n")
        except:
            return "\n".join(output_parts + ["Invalid address"]), ""

        all_utxos, _ = get_utxos(user_addr.strip(), dust_threshold)
        if not all_utxos:
            return "\n".join(output_parts + ["No UTXOs above dust threshold"]), ""

        ratio = {"Conservative (70/30, Low Risk)": 0.3, "Efficient (60/40, Default)": 0.4,
                          "Aggressive (50/50, Max Savings)": 0.5}[prune_choice]
        keep = max(1, int(len(all_utxos) * (1 - ratio)))
        pruned_utxos = all_utxos[:keep]
        output_parts.append(f"Scan: {len(all_utxos)} → Will use: {len(pruned_utxos)} UTXOs ({prune_choice})")

        if not confirm_proceed:
            output_parts.append("\nClick 'Generate Real TX Hex' to build transaction")
            return "\n".join(output_parts), ""

       # Real TX is now built entirely in build_real_tx — nothing to do here
        return "\n".join(output_parts), ""

    # Two-step flow
    def analysis_pass(addr, strategy, threshold, dest, sweep, invoice):
        global pruned_utxos_global, input_vb_global, output_vb_global

        log, _ = main_flow(addr.strip(), strategy, dest, False, threshold)

        all_utxos, _ = get_utxos(addr.strip(), threshold)
        ratio = {
            "Conservative (70/30, Low Risk)": 0.3,
            "Efficient (60/40, Default)": 0.4,
            "Aggressive (50/50, Max Savings)": 0.5
        }[strategy]
        keep = max(1, int(len(all_utxos) * (1 - ratio)))
        pruned_utxos_global = all_utxos[:keep]

        _, vb = address_to_script_pubkey(addr.strip())
        input_vb_global = vb['input_vb']
        output_vb_global = vb['output_vb']

        if not pruned_utxos_global:
            log += "\nWarning: No UTXOs selected — nothing to consolidate."

        return log, gr.update(visible=True), gr.update(visible=False)


    def build_real_tx(addr, strategy, threshold, dest, sweep, invoice):
        global pruned_utxos_global, input_vb_global, output_vb_global

        if not pruned_utxos_global:
            return (
                "Hold on! ⚡\n\n"
                "You checked Lightning sweep, but we haven’t scanned your address yet.\n"
                "Please click **Run Pruner** first so we know which dusty UTXOs to consolidate.\n\n"
                "After that, paste your Lightning invoice and click Generate again — we’ll turn your dust into spendable sats instantly!",
                gr.update(visible=False),
                gr.update(visible=False)
            )

        try:
            if sweep and invoice.strip().startswith("lnbc"):
                log, hex_out = lightning_sweep_flow(
                    pruned_utxos_global, invoice.strip(),
                    input_vb_global, output_vb_global
                )
                return log, gr.update(value=hex_out, visible=True), gr.update(visible=False)

            # Normal on-chain consolidation
            dest_addr_to_use = dest.strip() if dest and dest.strip() else addr.strip()
            dest_script, _ = address_to_script_pubkey(dest_addr_to_use)
            dao_script, _ = address_to_script_pubkey(dao_cut_addr)

            tx = Tx(tx_ins=[], tx_outs=[])
            total_in = 0
            for u in pruned_utxos_global:
                tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
                total_in += int(u['amount'] * 1e8)

            est_vb = 10.5 + input_vb_global * len(pruned_utxos_global) + output_vb_global * 2
            fee = int(est_vb * 5)
            dao_cut = int(fee * 0.05)
            send_amount = total_in - fee - dao_cut

            if send_amount < 546:
                raise ValueError("Not enough sats left after fee + DAO cut")

            tx.tx_outs.append(TxOut(send_amount, dest_script))
            if dao_cut >= 546:
                tx.tx_outs.append(TxOut(dao_cut, dao_script))
            else:
                log = log  # keep previous log

            raw_hex = tx.encode().hex()
            success_msg = (
                f"Success! Consolidated {len(pruned_utxos_global)} UTXOs ({total_in_sats:,} sats total)\n"
                f"Estimated fee: ~{fee:,} sats | DAO cut: {dao_cut:,} sats\n"
                "Copy hex → Load in Electrum / Sparrow → Sign → Broadcast\n\n"
                "⚡ Want instant Lightning balance instead?\n\n"
                f"→ Create a Lightning invoice for exactly **{total_in_sats - fee - dao_cut:,} sats**\n"
                "   (this is your dust minus the small miner fee + DAO cut)\n\n"
                "Then check “Sweep to Lightning ⚡” below, paste the invoice, and hit Generate.\n"
                "Your dust becomes real spendable Lightning in seconds — zero custody.\n\n"
                "Surge the swarm. Ledger’s yours. 🜂"
            )

            return success_msg, gr.update(value=raw_hex, visible=True), gr.update(visible=False)

        except Exception as e:
            error = f"Transaction failed: {e}"
            return error, gr.update(visible=False), gr.update(visible=False)
        
    # Button wiring
    submit_btn.click(fn=analysis_pass,
                     inputs=[user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice],
                     outputs=[output_text, generate_btn, raw_tx_text])

    generate_btn.click(fn=build_real_tx,
                       inputs=[user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice],
                       outputs=[output_text, raw_tx_text, generate_btn])

    # RBF
    rbf_btn.click_count = 0
    def do_rbf(hex_in):
        if not hex_in or not hex_in.strip(): return "Paste hex first", None
        rbf_btn.click_count += 1
        new_hex, msg = rbf_bump(hex_in.strip(), 50)
        if new_hex:
            return f"Bump #{rbf_btn.click_count} (+{50 * rbf_btn.click_count} sat/vB)\n\n{new_hex}", new_hex
        return msg or "Error", None
    rbf_btn.click(do_rbf, rbf_input, [rbf_output, rbf_input])

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), allowed_paths=["static"])
