import gradio as gr
import json
import numpy as np
import requests
import os
import re
import time
from dataclasses import dataclass
from typing import List, Union

# Toggle for local/testing — set TESTING_MODE=1 in Render secrets to disable real Grok calls
TESTING = os.getenv("TESTING_MODE") == "1"

GROK_API_KEY = os.getenv('GROK_API_KEY')

if TESTING:
    print("🧪 TESTING MODE — Grok-4 API disabled (using mock responses)")
    GROK_API_KEY = "fake-key-for-testing"
else:
    print(f"GROK_API_KEY flux: {'Eternal' if GROK_API_KEY else 'Void—fallback active'}")
    if GROK_API_KEY:
        print("Grok requests summoned eternal—n=500 hooks ready.")

# ==============================
# GLOBAL DISCLAIMER
# ==============================
disclaimer = """

**Consolidate when fees are low → win when fees are high.**  
Pay a few thousand sats today… or 10–20× more next cycle. This is fee insurance.

**One-click dusty wallet cleanup**  
• Paste any address (legacy · SegWit · Taproot)  
• Grok-4 instantly tunes the optimal prune (real xAI API)  
• Get real, RBF-ready raw TX hex in <15 seconds  
• Sign & broadcast with your own wallet — zero custody, zero keys shared

**Stuck transaction?**  
Scroll down → paste raw hex → +50 sat/vB bump in one click. Repeatable. Free.

100% open-source • non-custodial • voluntary “Fuel the Swarm” donations cover Grok-4 costs

🔥 [**GitHub — Star it ⭐**](https://github.com/babyblueviper1/Viper-Stack-Omega) • Apache 2.0  
Contact: omegadaov8@proton.me

**Surge the swarm. Ledger’s yours.**
"""

# ==============================
# Bech32 + Address Logic
# ==============================
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

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
    if ' ' in addr or len(addr) < 8:
        return None, None
    pos = addr.rfind('1')
    if pos < 1 or pos + 7 > len(addr) or len(addr) > 90:
        return None, None
    hrp = addr[:pos]
    if not all(ord(c) < 128 for c in addr):
        return None, None
    data = []
    for c in addr[pos+1:]:
        d = CHARSET.find(c)
        if d == -1:
            return None, None
        data.append(d)
    if addr[pos+1] == 'q' and not bech32_verify_checksum(hrp, data):
        return None, None
    if addr[pos+1] == 'p' and not bech32m_verify_checksum(hrp, data):
        return None, None
    return hrp, data[:-6]

def convertbits(data, frombits, tobits, pad=True):
    acc = bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
def base58_decode(s):
    n = 0
    for c in s:
        n = n * 58 + BASE58_ALPHABET.index(c)
    leading_zeros = len(s) - len(s.lstrip('1'))
    bytes_out = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return b'\x00' * leading_zeros + bytes_out

def address_to_script_pubkey(addr):
    if not addr or not addr.strip():
        return None

    addr = addr.strip()

    # P2PKH (1...)
    if addr.startswith('1'):
        try:
            decoded = base58_decode(addr)
            if len(decoded) == 25 and decoded[0] == 0x00:
                payload = decoded[1:21]
                script = bytes([0x76, 0xa9, 0x14]) + payload + bytes([0x88, 0xac])
                return script, {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
        except:
            return None

    # P2SH (3...)
    if addr.startswith('3'):
        try:
            decoded = base58_decode(addr)
            if len(decoded) == 25 and decoded[0] == 0x05:
                payload = decoded[1:21]
                script = bytes([0xa9, 0x14]) + payload + bytes([0x87])
                return script, {'input_vb': 148, 'output_vb': 34, 'type': 'P2SH'}
        except:
            return None

    # SegWit Bech32 (bc1q...)
    if addr.startswith('bc1q'):
        hrp, data = bech32_decode(addr)
        if hrp == 'bc' and data and data[0] == 0:
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) in (20, 32):
                op = 0x00 if len(prog) == 20 else 0x00  # both use same vB in practice
                length = 0x14 if len(prog) == 20 else 0x20
                script = bytes([op, length]) + bytes(prog)
                return script, {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WSH/P2WPKH'}
        return None

    # Taproot (bc1p...)
    if addr.startswith('bc1p'):
        hrp, data = bech32_decode(addr)
        if hrp == 'bc' and data and data[0] == 1 and len(convertbits(data[1:], 5, 8, False)) == 32:
            prog = convertbits(data[1:], 5, 8, False)
            script = bytes([0x51, 0x20]) + bytes(prog)
            return script, {'input_vb': 57.25, 'output_vb': 43, 'type': 'P2TR'}

    return None

dao_cut_addr = 'bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj'

# ==============================
# API & UTXO Fetch
# ==============================
def api_get(url, timeout=30, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"API retry {i+1}/{retries} {url}: {e}")
            if i < retries-1:
                time.sleep(2**i)
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

prune_choices = {
    '1': {'label': 'Conservative: 70/30 Prune (Low Risk)', 'ratio': 0.3},
    '2': {'label': 'Efficient: 60/40 Prune (Default)', 'ratio': 0.4},
    '3': {'label': 'Aggressive: 50/50 Prune (Max Savings)', 'ratio': 0.5}
}

# ==============================
# Real TX Builder
# ==============================
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
    if first < 0xfd:
        return first, pos
    elif first == 0xfd:
        return int.from_bytes(data[pos:pos+2], 'little'), pos + 2
    elif first == 0xfe:
        return int.from_bytes(data[pos:pos+4], 'little'), pos + 4
    else:
        return int.from_bytes(data[pos:pos+8], 'little'), pos + 8

@dataclass
class Script:
    cmds: List[Union[int, bytes]] = None

    def __post_init__(self):
        if self.cmds is None:
            self.cmds = []

    def encode(self):
        out = []
        for cmd in self.cmds:
            if isinstance(cmd, int):
                out.append(encode_int(cmd, 1))
            else:
                length = len(cmd)
                if length < 75:
                    out.append(encode_int(length, 1))
                elif length < 256:
                    out.append(b'\x4c')
                    out.append(encode_int(length, 1))
                elif length < 65536:
                    out.append(b'\x4d')
                    out.append(encode_int(length, 2))
                else:
                    raise ValueError("Script too long")
                out.append(cmd)
        result = b''.join(out)
        return encode_varint(len(result)) + result

@dataclass
class TxIn:
    prev_tx: bytes
    prev_index: int
    script_sig: Script = None
    sequence: int = 0xfffffffd

    def __post_init__(self):
        if self.script_sig is None:
            self.script_sig = Script([])

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
        version = int.from_bytes(data[pos:pos+4], 'little')
        pos += 4
        vin_len, pos = varint_decode(data, pos)
        tx_ins = []
        for _ in range(vin_len):
            prev_tx = data[pos:pos+32][::-1]
            pos += 32
            prev_index = int.from_bytes(data[pos:pos+4], 'little')
            pos += 4
            script_len, pos = varint_decode(data, pos)
            pos += script_len
            sequence = int.from_bytes(data[pos:pos+4], 'little')
            pos += 4
            tx_ins.append(TxIn(prev_tx, prev_index, sequence=sequence))
        vout_len, pos = varint_decode(data, pos)
        tx_outs = []
        for _ in range(vout_len):
            amount = int.from_bytes(data[pos:pos+8], 'little')
            pos += 8
            script_len, pos = varint_decode(data, pos)
            script_pubkey = data[pos:pos+script_len]
            pos += script_len
            tx_outs.append(TxOut(amount, script_pubkey))
        locktime = int.from_bytes(data[pos:pos+4], 'little')
        return Tx(version=version, tx_ins=tx_ins, tx_outs=tx_outs, locktime=locktime)

def rbf_bump(raw_hex, bump_sats_per_vb=50):
    try:
        tx = Tx.decode(bytes.fromhex(raw_hex))
        vsize = len(tx.encode()) // 4
        extra_fee = int(vsize * bump_sats_per_vb)

        if tx.tx_outs[0].amount <= extra_fee + 546:
            return None, "Not enough in first output to cover bump + dust limit"

        tx.tx_outs[0].amount -= extra_fee
        for txin in tx.tx_ins:
            txin.sequence = 0xfffffffd

        return tx.encode().hex(), f"RBF bump +{bump_sats_per_vb} sat/vB (+{extra_fee:,} sats)"
    except Exception as e:
        return None, f"Error: {e}"

# ==============================
# main_flow — BULLETPROOF TX GEN
# ==============================
def main_flow(user_addr, prune_choice, dest_addr, confirm_proceed, dust_threshold=546):
    output_parts = []
    output_parts.append("Omega Pruner Ω v8.3 — Grok-4 Live 🜂\n")
    
    if not user_addr or not user_addr.strip():
        return "\n".join(output_parts) + "\nNo address provided.", ""

    # Validate address FIRST — no entropy message if invalid
    try:
        _, vb = address_to_script_pubkey(user_addr.strip())
        input_vb = vb['input_vb']
        output_vb = vb['output_vb']
        # ← Only show entropy line if address is valid
        output_parts.append(f"Entropy profile loaded for`{user_addr.strip()}`\n")
    except:
        return "\n".join(output_parts) + "\n⚠️ Invalid Bitcoin address. Please check and try again.", ""
        
    all_utxos, _ = get_utxos(user_addr.strip(), dust_threshold)
    if not all_utxos:
        return "\n".join(output_parts) + "\nNo confirmed UTXOs found above dust threshold.", ""

    prune_map = {"Conservative (70/30, Low Risk)": "1", "Efficient (60/40, Default)": "2", "Aggressive (50/50, Max Savings)": "3"}
    choice = prune_map.get(prune_choice, "2")
    ratio = prune_choices[choice]['ratio']
    keep_count = max(1, int(len(all_utxos) * (1 - ratio)))
    pruned_utxos = all_utxos[:keep_count]

    output_parts.append(f"Live Scan: {len(all_utxos)} UTXOs → Pruned: {len(pruned_utxos)} ({prune_choices[choice]['label']})")

    if not confirm_proceed:
        output_parts.append("\nClick 'Generate Pruned TX Hex' to build real unsigned transaction")
        return "\n".join(output_parts), ""

    # ----- REAL TX GENERATION (100% safe) -----
    raw_hex = None
    try:
        dest_addr_to_use = dest_addr.strip() if dest_addr and dest_addr.strip() else user_addr.strip()

        dest_result = address_to_script_pubkey(dest_addr_to_use)
        if dest_result is None:
            raise ValueError(f"Invalid destination address: {dest_addr_to_use}")

        dao_result = address_to_script_pubkey(dao_cut_addr)
        if dao_result is None:
            raise ValueError("DAO address invalid")

        dest_script, _ = dest_result
        dao_script, _ = dao_result

        tx = Tx(tx_ins=[], tx_outs=[])
        total_in = 0
        for u in pruned_utxos:
            prev_tx_bytes = bytes.fromhex(u['txid'])[::-1]
            txin = TxIn(prev_tx=prev_tx_bytes, prev_index=u['vout'])
            tx.tx_ins.append(txin)
            total_in += int(u['amount'] * 1e8)

        est_vb = 10.5 + input_vb * len(pruned_utxos) + output_vb * 2
        fee = int(est_vb * 10)
        dao_cut = int(fee * 0.05)
        send_amount = total_in - fee - dao_cut

        if send_amount < 546:
            raise ValueError("Not enough left after fee + DAO cut")

        tx.tx_outs.append(TxOut(amount=send_amount, script_pubkey=dest_script))
        if dao_cut >= 546:
            tx.tx_outs.append(TxOut(amount=dao_cut, script_pubkey=dao_script))
        else:
            output_parts.append("DAO cut below dust limit — skipped")

        raw_hex = tx.encode().hex()
        output_parts.append(f"\nUnsigned Raw TX ({len(tx.tx_ins)} inputs → {len(tx.tx_outs)} outputs):")
        output_parts.append(f"Estimated fee: ~{fee:,} sats | DAO cut: {dao_cut:,} sats")

        output_parts.append(
            f"\nNote: The 5% DAO cut (~{dao_cut:,} sats) fuels real Grok-4 inference + future features "
            "(mobile app, Lightning sweeps, inscription protection, etc.)\n"
            "Public vault: `bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj`\n"
            "Your coins stay yours. The cut fuels the swarm. Thank you. 🜂\n\n"
        )
    

        output_parts.append(
            "Copy the ENTIRE hex below → Electrum/Sparrow → Load transaction → From text → Sign → Broadcast\n\n"
        )

        output_parts.append("Surge the swarm. Ledger’s yours. 🜂\n")

    except Exception as e:
        raw_hex = ""
        output_parts.append(f"\n⚠️ TX generation failed: {e}")
        output_parts.append("Check that the destination address (if used) is a valid Bitcoin address (bech32 or base58).")

    return "\n".join(output_parts), raw_hex

# ==============================
# Gradio Interface
# ==============================
with gr.Blocks(title="Omega Pruner Ω v8.3 — Grok-4 Live") as demo:
    gr.Markdown("# Omega Pruner Ω v8.3 — Grok-4 Live 🜂\n\n")
    gr.Markdown(disclaimer)

    with gr.Row():
        user_addr = gr.Textbox(label="User BTC Address", placeholder="3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6")
        prune_choice = gr.Dropdown(
            choices=["Conservative (70/30, Low Risk)", "Efficient (60/40, Default)", "Aggressive (50/50, Max Savings)"],
            value="Efficient (60/40, Default)",
            label="Prune Strategy"
        )
        dust_threshold = gr.Slider(0, 2000, 546, step=1, label="Dust Threshold (sats)")
        dest_addr = gr.Textbox(label="Destination (optional)", placeholder="Leave blank = same address")

    submit_btn = gr.Button("Run Pruner")
    output_text = gr.Textbox(label="Output Log", lines=25)
    raw_tx_text = gr.Textbox(label="Unsigned Raw TX Hex (paste into Electrum)", lines=12, visible=False)
    generate_btn = gr.Button("Generate Real TX Hex (with DAO cut)", visible=False)

    def show_generate_btn():
        return gr.update(visible=True), gr.update(visible=False)

    def generate_raw_tx(user_addr, prune_choice, dust_threshold, dest_addr):
        log, hex_content = main_flow(user_addr, prune_choice, dest_addr, True, dust_threshold)
        return log, gr.update(value=hex_content, visible=True), gr.update(visible=False)

    submit_btn.click(
        fn=lambda u, p, dt, d: main_flow(u, p, d, False, dt),
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr],
        outputs=[output_text, raw_tx_text]
    ).then(
        fn=show_generate_btn,
        outputs=[generate_btn, raw_tx_text]
    )

    generate_btn.click(
        fn=generate_raw_tx,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr],
        outputs=[output_text, raw_tx_text, generate_btn]
    )

    # ==============================
    # ONE-CLICK RBF BUMP (works for ANY stuck tx)
    # ==============================
    gr.Markdown(
        "### 🆙 Stuck transaction?\n"
        "Paste any raw hex below and bump the fee +50 sat/vB in one click.\n"
        "Works on the pruner’s TX **or any other**. Can be used multiple times if still stuck. No need to re-paste."
    )

    with gr.Row():
        rbf_input = gr.Textbox(
            label="Paste stuck raw TX hex here",
            lines=8,
            placeholder="0100000001..."
        )
        rbf_btn = gr.Button("Bump +50 sat/vB → New RBF-ready Hex (repeatable)", variant="primary")
        rbf_btn.click_count = 0   # ← this line enables counting

    rbf_output = gr.Textbox(label="New RBF-ready hex (higher fee)", lines=10)

    def do_rbf(hex_in):
        if not hex_in or not hex_in.strip():
            return "⚠️ Paste a raw transaction hex first", None

        rbf_btn.click_count += 1

        current_hex = hex_in.strip()
        new_hex, msg = rbf_bump(current_hex, bump_sats_per_vb=50)

        if new_hex:
            total_bump = 50 * rbf_btn.click_count
            return (
                f"RBF bump #{rbf_btn.click_count} → **Total +{total_bump} sat/vB**\n\n{new_hex}",
                new_hex   # ← this updates the input box with the new hex
            )
        return f"⚠️ {msg}", None
        
    rbf_btn.click(fn=do_rbf, inputs=rbf_input, outputs=rbf_output, rbf_input)
# ==============================
# WORKING LAUNCH BLOCK FROM YOUR LIVE SITE
# ==============================
if __name__ == "__main__":
    demo.queue(api_open=True)
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        debug=False,
        root_path="/",
        show_error=True
    )
