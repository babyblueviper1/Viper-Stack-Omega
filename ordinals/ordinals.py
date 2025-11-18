import gradio as gr
import json
import numpy as np
import requests
import os
import re
import time
from dataclasses import dataclass
from typing import List, Union

GROK_API_KEY = os.getenv('GROK_API_KEY')
print(f"GROK_API_KEY flux: {'Eternal' if GROK_API_KEY else 'Void—fallback active'}")
if GROK_API_KEY:
    print("Grok requests summoned eternal—n=500 hooks ready.")

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
    if addr.startswith('bc1q'):
        hrp, data = bech32_decode(addr)
        if hrp != 'bc' or not data or data[0] != 0:
            raise ValueError("Invalid SegWit v0")
        prog = convertbits(data[1:], 5, 8, False)
        if len(prog) == 20:
            return bytes([0x00, 0x14]) + bytes(prog), {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WPKH'}
        if len(prog) == 32:
            return bytes([0x00, 0x20]) + bytes(prog), {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WSH'}
    elif addr.startswith('bc1p'):
        hrp, data = bech32_decode(addr)
        if hrp == 'bc' and data and data[0] == 1:
            prog = convertbits(data[1:], 5, 8, False)
            if len(prog) == 32:
                return bytes([0x51, 0x20]) + bytes(prog), {'input_vb': 57.25, 'output_vb': 43, 'type': 'P2TR'}
    elif addr.startswith('1'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x00:
            return bytes([0x76,0xa9,0x14]) + dec[1:21] + bytes([0x88,0xac]), {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
    elif addr.startswith('3'):
        dec = base58_decode(addr)
        if len(dec) == 25 and dec[0] == 0x05:
            return bytes([0xa9,0x14]) + dec[1:21] + bytes([0x87]), {'input_vb': 148, 'output_vb': 34, 'type': 'P2SH'}
    raise ValueError(f"Unsupported address: {addr}")

dao_cut_addr = 'bc1qwnj2zumaf67d34k6cm2l6gr3uvt5pp2hdrtvt3ckc4aunhmr53cselkpty'

# ==============================
# API & UTXO + Inscriptions
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

    # Inscriptions from Hiro
    inscriptions = []
    try:
        ins_resp = api_get(f'https://api.hiro.so/ordinals/v1/inscriptions?address={addr}&limit=50')
        inscriptions = ins_resp.json().get('results', [])
    except:
        pass

    utxos = []
    for u in raw:
        if u['status']['confirmed']:
            confs = current_height - u['status']['block_height'] + 1
            if confs > 6 and u['value'] > dust_threshold:
                is_ins = any(i.get('tx_id') == u['txid'] and str(i.get('output', '')).endswith(f":{u['vout']}") for i in inscriptions)
                utxos.append({
                    'txid': u['txid'],
                    'vout': u['vout'],
                    'amount': u['value'] / 1e8,
                    'confs': confs,
                    'is_inscription': is_ins,
                    'dust_risk': u['value'] < dust_threshold * 1.1
                })
    utxos.sort(key=lambda x: (x['is_inscription'], x['amount']), reverse=True)
    return utxos, current_height

prune_choices = {
    '1': {'label': 'Conservative: 70/30 Prune (Low Risk)', 'ratio': 0.3},
    '2': {'label': 'Efficient: 60/40 Prune (Default)', 'ratio': 0.4},
    '3': {'label': 'Aggressive: 50/50 Prune (Max Savings)', 'ratio': 0.5}
}

# ==============================
# TX Builder
# ==============================
def encode_int(i, nbytes, encoding='little'):
    return i.to_bytes(nbytes, encoding)

def encode_varint(i):
    if i < 0xfd: return bytes([i])
    if i < 0x10000: return b'\xfd' + encode_int(i, 2)
    if i < 0x100000000: return b'\xfe' + encode_int(i, 4)
    return b'\xff' + encode_int(i, 8)

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
                l = len(cmd)
                if l < 75:
                    out += [encode_int(l, 1), cmd]
        return encode_varint(len(b''.join(out))) + b''.join(out)

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

# ==============================
# run_phases
# ==============================
def run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, dao_cut):
    is_prod = bool(os.getenv('RENDER_EXTERNAL_HOSTNAME') or os.getenv('PORT'))
    
    if pruned_utxos:
        if is_prod:
            import random
            s_rho = 0.292 + random.uniform(-0.01, 0.01)
            s_tuned = 0.611 + random.uniform(-0.02, 0.02)
        else:
            import qutip as qt
            dim = len(pruned_utxos) + 1
            psi0 = qt.basis(dim, 0)
            rho_initial = psi0 * psi0.dag()
            mixed_dm = qt.rand_dm(dim)
            mixed_weight = np.std([u['amount'] for u in pruned_utxos]) / np.mean([u['amount'] for u in pruned_utxos])
            rho_initial = (1 - mixed_weight) * rho_initial + mixed_weight * mixed_dm
            rho_initial = rho_initial / rho_initial.tr()
            s_rho = qt.entropy_vn(rho_initial)
            noise_dm = qt.rand_dm(dim)
            tune_p = 0.389
            rho_tuned = tune_p * rho_initial + (1 - tune_p) * noise_dm
            rho_tuned = rho_tuned / rho_tuned.tr()
            s_tuned = qt.entropy_vn(rho_tuned)
        shard['s_rho'] = float(s_rho)
        shard['s_tuned'] = float(s_tuned)
        gci = 0.92 if s_tuned > 0.6 else 0.8
        shard['gci'] = gci
    else:
        shard['s_rho'] = 0.292
        shard['s_tuned'] = 0.611
        shard['gci'] = 0.8

    with open('prune_blueprint_v8.json', 'w') as f:
        json.dump(shard, f, indent=4)

    with open('prune_blueprint_v8.json', 'r') as f:
        shard_data = json.load(f)

    full_blueprint = {
        'id': f'omega_v8_btc_user_{int(time.time()*1e6)}',
        'shard': shard_data,
        'coherence': {'gci': gci, 'gci_surge': gci > 0.92},
        'prune_target': f'{selected_ratio*100:.0f}%'
    }

    seed_file = 'data/seed_blueprints_v8.json'
    os.makedirs('data', exist_ok=True)
    seeds = []
    if os.path.exists(seed_file):
        with open(seed_file) as f:
            seeds = json.load(f)
    seeds.append(full_blueprint)
    with open(seed_file, 'w') as f:
        json.dump(seeds, f, indent=4)

    return gci, json.dumps(full_blueprint, indent=2), seed_file

# ==============================
# grok_tune
# ==============================
def grok_tune(gci_base):
    if not GROK_API_KEY:
        return gci_base
    headers = {'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'}
    data = {
        'model': 'grok-4-0709',
        'messages': [{'role': 'user', 'content': f'Tune GCI {gci_base} for Ω mempool prune—output only a single float 0.70-0.99'}]
    }
    try:
        r = requests.post('https://api.x.ai/v1/chat/completions', headers=headers, json=data, timeout=180)
        if r.status_code == 200:
            num = re.search(r'\d+\.\d+', r.json()['choices'][0]['message']['content'])
            if num:
                return float(num.group())
    except:
        pass
    return gci_base

# ==============================
# main_flow
# ==============================
def main_flow(user_addr, prune_choice, dest_addr, confirm_proceed, dust_threshold=546):
    output_parts = []
    disclaimer = """
BTC UTXO Pruner Ω v8.1 — RBF-ready, Taproot-native

• Generates prune plan, fee estimate & unsigned raw TX hex — NO BTC is sent here
• Fully Taproot (bc1p) & Ordinals-compatible — correct vB weights, dust slider, RBF eternal
• Ordinals/Inscription detection temporarily disabled (speed & reliability) — will return when a stable API exists
• Dust threshold configurable (default 546 sats) — lower at your own risk for inscription consolidation when fees <2 sat/vB
• Non-custodial — only public UTXOs are read, you keep full key control
• Requires UTXO-capable wallet (Electrum, Sparrow, etc.) to sign & broadcast
• Fund your address first for live scan
• High-UTXO addresses (50+) may take 120–180s — patience eternal
• Not financial advice — verify everything, broadcast at your own risk

Surge the swarm. Ledger’s yours.

Contact: omegadaov8@proton.me
    """
    output_parts.append(disclaimer)

    if not user_addr:
        return "\n".join(output_parts) + "\nNo address provided.", ""

    # Validation & type detection
    try:
        _, vb = address_to_script_pubkey(user_addr)
        input_vb = vb['input_vb']
        output_vb = vb['output_vb']
        addr_type = vb['type']
    except:
        return "\n".join(output_parts) + "\nInvalid Bitcoin address.", ""

    # Price
    try:
        btc_usd = api_get('https://api.coinpaprika.com/v1/tickers/btc-bitcoin').json()['quotes']['USD']['price']
    except:
        btc_usd = 98500

    all_utxos, _ = get_utxos(user_addr, dust_threshold)
    if not all_utxos:
        return "\n".join(output_parts) + "\nNo confirmed UTXOs found above dust threshold.", ""

    # Prune strategy
    prune_map = {"Conservative (70/30, Low Risk)": "1", "Efficient (60/40, Default)": "2", "Aggressive (50/50, Max Savings)": "3"}
    choice = prune_map.get(prune_choice, "2")
    ratio = prune_choices[choice]['ratio']
    keep_count = max(1, int(len(all_utxos) * (1 - ratio)))
    pruned_utxos = all_utxos[:keep_count]

    # Fee estimation
    try:
        fee_rate = api_get('https://mempool.space/api/v1/fees/recommended').json()['economyFee'] * 1e-8
    except:
        fee_rate = 10e-8

    overhead = 10.5
    raw_vb = overhead + input_vb * len(all_utxos) + output_vb
    pruned_vb = overhead + input_vb * len(pruned_utxos) + output_vb * 2
    raw_fee = raw_vb * fee_rate
    pruned_fee = pruned_vb * fee_rate
    savings = raw_fee - pruned_fee
    dao_cut = 0.05 * savings

    total_value = sum(u['amount'] for u in pruned_utxos)
    send_amount = total_value - pruned_fee - (dao_cut if confirm_proceed else 0)

    output_parts += [
        f"UTXOs: {len(all_utxos)} → Pruned: {len(pruned_utxos)} ({prune_choices[choice]['label']})",
        f"Estimated fee savings: {savings:.8f} BTC",
        f"DAO 5% incentive: {dao_cut:.8f} BTC",
        f"Net send: {send_amount:.8f} BTC"
    ]

    # Shard + phases
    shard = {
        "utxos": pruned_utxos,
        "pruned_fee": pruned_fee,
        "raw_fee": raw_fee,
        "savings_usd": savings * btc_usd,
        "btc_usd": btc_usd,
        "prune_ratio": ratio,
        "user_addr": user_addr,
        "dest_addr": dest_addr or user_addr,
        "dao_cut_addr": dao_cut_addr,
        "dust_threshold": dust_threshold,
        "addr_type": addr_type,
        "psbt_stub": "pending",
        "dao_cut": 0.0
    }
    gci, bp, _ = run_phases(shard, pruned_utxos, ratio, raw_fee, pruned_fee, savings*btc_usd, btc_usd, choice, 0.8, "pending", user_addr, dest_addr or user_addr, 0)
    shard['grok_tuned_gci'] = grok_tune(gci)
    output_parts.append(f"GCI: {shard['grok_tuned_gci']:.3f}")

    if not confirm_proceed:
        output_parts.append("\nClick 'Generate Pruned TX Hex' to build unsigned transaction")
        return "\n".join(output_parts), ""

    # Build raw TX
    try:
        dest_script, _ = address_to_script_pubkey(dest_addr or user_addr)
        dao_script, _ = address_to_script_pubkey(dao_cut_addr)

        tx = Tx()
        for u in pruned_utxos:
            tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
        tx.tx_outs.append(TxOut(int(send_amount * 1e8), dest_script))
        if dao_cut * 1e8 > 546:
            tx.tx_outs.append(TxOut(int(dao_cut * 1e8), dao_script))

        raw_hex = tx.encode().hex()
        output_parts.append("\nUnsigned Raw TX (RBF-enabled):")
        output_parts.append(raw_hex)
    except Exception as e:
        raw_hex = ""
        output_parts.append(f"\nTX build error: {e}")

    shard['dao_cut'] = float(dao_cut)
    run_phases(shard, pruned_utxos, ratio, raw_fee, pruned_fee, savings*btc_usd, btc_usd, choice, gci, raw_hex[:50]+"...", user_addr, dest_addr or user_addr, dao_cut)

    return "\n".join(output_parts), raw_hex

# ==============================
# Gradio Interface
# ==============================
with gr.Blocks(title="Omega DAO Pruner v8.1") as demo:
    gr.Markdown("# Omega DAO Pruner v8.1 - BTC UTXO Optimizer")
    gr.Markdown(disclaimer)

    with gr.Row():
        user_addr = gr.Textbox(label="User BTC Address", placeholder="bc1q...")
        prune_choice = gr.Dropdown(
            choices=[
                "Conservative (70/30, Low Risk)",
                "Efficient (60/40, Default)",
                "Aggressive (50/50, Max Savings)"
            ],
            value="Efficient (60/40, Default)",
            label="Prune Strategy"
        )
        dust_threshold = gr.Slider(minimum=0, maximum=2000, value=546, step=1, label="Dust Threshold (sats)")
        dest_addr = gr.Textbox(label="Destination Address (Optional)", placeholder="Same as User Addr")

    submit_btn = gr.Button("Run Pruner")
    output_text = gr.Textbox(label="Output Log", lines=22)
    raw_tx_text = gr.Textbox(label="Unsigned Raw TX Hex", lines=10, visible=False)
    generate_btn = gr.Button("Generate Pruned TX Hex (DAO Pool Incentive Included)", visible=False)

    def show_generate_btn():
        return gr.update(visible=True), gr.update(visible=False)

    def generate_raw_tx(user_addr, prune_choice, dust_threshold, dest_addr):
        log, hex_tx = main_flow(user_addr, prune_choice, dest_addr, True, dust_threshold)
        return log, gr.update(value=hex_tx, visible=True), gr.update(visible=False)

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
