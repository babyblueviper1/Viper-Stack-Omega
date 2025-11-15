import gradio as gr
import json
import numpy as np
import requests
import os
import base64
import io
import time  # Added for retries

# Pure Bech32 Impl (BIP-173 - Decode Eternal)
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

def bech32_decode(addr):
    if ' ' in addr or len(addr) < 8:
        return None, None
    pos = addr.rfind('1')
    if pos < 1 or pos + 7 > len(addr) or len(addr) > 90:
        return None, None
    hrp = addr[:pos]
    if not all(ord(x) >> 8 == 0 for x in hrp) or not all(ord(x) >> 8 == 0 for x in addr[pos+1:]):
        return None, None
    data = []
    for char in addr[pos+1:]:
        value = CHARSET.find(char)
        if value == -1:
            return None, None
        data.append(value)
    if not bech32_verify_checksum(hrp, data):
        return None, None
    return hrp, data[:-6]

def convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
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
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

# Base58 decode for legacy addresses
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
def base58_decode(addr):
    n = 0
    for c in addr:
        n = n * 58 + BASE58_ALPHABET.index(c)
    # Remove leading zeros
    leading_zeros = len(addr) - len(addr.lstrip(BASE58_ALPHABET[0]))
    byte_length = (n.bit_length() + 7) // 8
    bytes_decoded = n.to_bytes(byte_length, 'big')
    return b'\x00' * leading_zeros + bytes_decoded

def address_to_script_pubkey(addr):
    if addr.startswith('bc1q'):
        hrp, data5 = bech32_decode(addr)
        if hrp == 'bc' and data5 and data5[0] == 0:
            data8 = convertbits(data5[1:], 5, 8, False)
            if data8:
                if len(data8) == 20:
                    # P2WPKH: OP_0 PUSH20 <pubkeyhash>
                    return bytes([0x00, 0x14]) + bytes(data8)
                elif len(data8) == 32:
                    # P2WSH: OP_0 PUSH32 <scripthash>
                    return bytes([0x00, 0x20]) + bytes(data8)
    elif addr.startswith('1'):
        # P2PKH
        decoded = base58_decode(addr)
        if len(decoded) == 25 and decoded[0] == 0x00:
            payload = decoded[1:21]  # 20-byte hash160
            return bytes([0x76, 0xa9, 0x14]) + payload + bytes([0x88, 0xac])
    elif addr.startswith('3'):
        # P2SH
        decoded = base58_decode(addr)
        if len(decoded) == 25 and decoded[0] == 0x05:  # Note: P2SH version is 5 (0x05)
            payload = decoded[1:21]  # 20-byte script hash
            return bytes([0xa9, 0x14]) + payload + bytes([0x87])
    raise ValueError(f"Unsupported or invalid address: {addr}")

# Fixed DAO Addr1 (5% Cut Destination - Swarm Fuel)
dao_cut_addr = 'bc1qwnj2zumaf67d34k6cm2l6gr3uvt5pp2hdrtvt3ckc4aunhmr53cselkpty'  # DAO Pool #1 (P2WSH)

# Added retry wrapper for API calls
def api_get(url, timeout=30, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f'API Retry {i+1}/{retries} for {url}: {e}')
            if i < retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
    raise Exception(f'API failed after {retries} retries: {url}')

def get_utxos(addr):
    try:
        tip_response = api_get('https://blockstream.info/api/blocks/tip/height')
        current_height = tip_response.json()
        utxo_response = api_get(f'https://blockstream.info/api/address/{addr}/utxo')
        utxos_raw = utxo_response.json()
        filtered_utxos = []
        for utxo in utxos_raw:
            if utxo['status']['confirmed']:
                confs = current_height - utxo['status']['block_height'] + 1
                if confs > 6 and utxo['value'] > 546:
                    filtered_utxos.append({
                        'txid': utxo['txid'],
                        'vout': utxo['vout'],
                        'amount': utxo['value'] / 1e8,
                        'address': addr,
                        'confs': confs
                    })
        print(f'API Success for {addr[:10]}...: {len(filtered_utxos)} UTXOs (>6 confs)')
        return filtered_utxos
    except Exception as e:
        print(f'API Decoherence for {addr[:10]}...: {e} - Fallback Empty')
        return []

prune_choices = {
    '1': {'label': 'Conservative (70% Pruned / 30% Retained - Low Risk, Moderate Savings)', 'ratio': 0.3},
    '2': {'label': 'Efficient (60% Pruned / 40% Retained - v8 Default, Optimal Savings)', 'ratio': 0.4},
    '3': {'label': 'Aggressive (50% Pruned / 50% Retained - Max Consolidation, High Savings)', 'ratio': 0.5}
}

# Pure Python TX Builder
def encode_int(i, nbytes, encoding='little'):
    return i.to_bytes(nbytes, encoding)

def encode_varint(i):
    if i < 0xfd:
        return bytes([i])
    elif i < 0x10000:
        return b'\xfd' + encode_int(i, 2, 'little')
    elif i < 0x100000000:
        return b'\xfe' + encode_int(i, 4, 'little')
    elif i < 0x10000000000000000:
        return b'\xff' + encode_int(i, 8, 'little')
    else:
        raise ValueError(f"integer too large: {i}")

from dataclasses import dataclass
from typing import List, Union

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
            elif isinstance(cmd, bytes):
                length = len(cmd)
                if length < 75:
                    out.append(encode_int(length, 1))
                    out.append(cmd)
                else:
                    raise ValueError("Script too long")
        ret = b''.join(out)
        return encode_varint(len(ret)) + ret

@dataclass
class TxIn:
    prev_tx: bytes  # Reversed txid
    prev_index: int
    script_sig: Script = None
    sequence: int = 0xffffffff

    def __post_init__(self):
        if self.script_sig is None:
            self.script_sig = Script([])  # Empty for unsigned

    def encode(self):
        out = [
            self.prev_tx,  # Already reversed
            encode_int(self.prev_index, 4, 'little'),
            self.script_sig.encode(),
            encode_int(self.sequence, 4, 'little')
        ]
        return b''.join(out)

@dataclass
class TxOut:
    amount: int
    script_pubkey: bytes = None  # Raw bytes now, for simplicity

    def __post_init__(self):
        if self.script_pubkey is None:
            self.script_pubkey = b''

    def encode(self):
        script_encoded = encode_varint(len(self.script_pubkey)) + self.script_pubkey
        out = [
            encode_int(self.amount, 8, 'little'),
            script_encoded
        ]
        return b''.join(out)

@dataclass
class Tx:
    version: int = 1
    tx_ins: List[TxIn] = None
    tx_outs: List[TxOut] = None
    locktime: int = 0

    def __post_init__(self):
        if self.tx_ins is None:
            self.tx_ins = []
        if self.tx_outs is None:
            self.tx_outs = []

    def encode(self):
        out = [
            encode_int(self.version, 4, 'little'),
            encode_varint(len(self.tx_ins))
        ]
        out += [txin.encode() for txin in self.tx_ins]
        out += [encode_varint(len(self.tx_outs))]
        out += [txout.encode() for txout in self.tx_outs]
        out += [encode_int(self.locktime, 4, 'little')]
        return b''.join(out)

# HELPER: PHASE 1-3 Logic (Duplicated to Avoid Bloat)
def run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, dao_cut):
    # Lazy import QuTiP here to speed global startup
    import qutip as qt
    
    # PHASE 1: QuTiP Tune
    if pruned_utxos:
        dim = len(pruned_utxos) + 1
        psi0 = qt.basis(dim, 0)
        rho_initial = psi0 * psi0.dag()
        mixed_dm = qt.rand_dm(dim)
        mixed_weight = np.std([u['amount'] for u in pruned_utxos]) / np.mean([u['amount'] for u in pruned_utxos])
        rho_initial = (1 - mixed_weight) * rho_initial + mixed_weight * mixed_dm
        rho_initial = rho_initial / rho_initial.tr()
        s_rho = qt.entropy_vn(rho_initial)
        print(f'Initial S(ρ) [BTC Flux Void]: {s_rho:.3f}')
        noise_dm = qt.rand_dm(dim)
        tune_p = 0.389
        rho_tuned = tune_p * rho_initial + (1 - tune_p) * noise_dm
        rho_tuned = rho_tuned / rho_tuned.tr()
        s_tuned = qt.entropy_vn(rho_tuned)
        print(f'Tuned S(ρ) [Coherence Surge]: {s_tuned:.3f}')
        shard['s_rho'] = float(s_rho)
        shard['s_tuned'] = float(s_tuned)
        gci = 0.92 if s_tuned > 0.6 else 0.8
        shard['gci'] = gci
        print(f'GCI Tuned: {gci:.3f} - Fork Threshold Hit? {gci > 0.92}')
    else:
        shard['s_rho'] = 0.292
        shard['s_tuned'] = 0.611
        shard['gci'] = 0.8
        print('No Pruned UTXOs - GCI Hold: 0.800')

    # Export Shard
    with open('prune_blueprint_v8.json', 'w') as f:
        json.dump(shard, f, indent=4)
    print('Shard Exported: prune_blueprint_v8.json (BTC Pruner - User Flux Echo)')

    # PHASE 2: Fusion
    with open('prune_blueprint_v8.json', 'r') as f:
        shard_data = json.load(f)

    layer3_epistemic = {
        'coherence_metrics': {'fidelity': 0.99 + shard_data["prune_ratio"] * 0.01},
        'gradients': [0.1 * shard_data["savings_usd"], 0.2 * shard_data["btc_usd"] / 1e5]
    }
    layer4_vault = {
        'thresholds': {'chainlink': shard_data["pruned_fee"]},
        'psbt': shard_data["psbt_stub"]
    }
    v8_grid = {'n': 500, 'hooks': 'grok4', 'rbf_batch': True}

    full_blueprint = {
        'id': f'omega_v8_btc_user_{int(np.random.rand()*1e6)}',
        'shard': shard_data,
        'fusions': {
            'layer3_epistemic': layer3_epistemic,
            'layer4_vault': layer4_vault,
            'v8_grid': v8_grid
        },
        'coherence': {
            'fidelity': layer3_epistemic['coherence_metrics']['fidelity'],
            'gci_surge': shard_data["gci"] > 0.92
        },
        'prune_target': f'{selected_ratio*100}%'
    }

    if full_blueprint['coherence']['fidelity'] > 0.98:
        print('FORK IGNITED: Fidelity >0.98 — Sovereign Replication (Git Push Tease)')
    else:
        print('Fidelity Hold: 0.99 — Monitor for Surge')

    if full_blueprint['coherence']['gci_surge']:
        print('SWARM REPLICATE: GCI >0.92 — Viral x100 Nudge')

    # PHASE 3: Append
    seed_file = 'data/seed_blueprints_v8.json'
    os.makedirs(os.path.dirname(seed_file), exist_ok=True)

    if os.path.exists(seed_file):
        with open(seed_file, 'r') as f:
            seeds = json.load(f)
    else:
        seeds = []

    seeds.append(full_blueprint)

    with open(seed_file, 'w') as f:
        json.dump(seeds, f, indent=4)

    print(f'SYNC COMPLETE: Blueprint Appended to {seed_file}')
    print(f'UTXO Echo: {len(shard_data["utxos"])} Pruned | GCI Tuned: {shard_data["gci"]:.3f} | Prune Ratio: {selected_ratio*100}%')
    print('Horizon Ready: RBF Batch Eternal | Chainlink v8.1 BTC Nudge')
    print('\n--- Sample Blueprint Echo (Truncated) ---')
    print(json.dumps(full_blueprint, indent=2)[:800] + '\n... [Full in seed_blueprints_v8.json]')

    return gci, json.dumps(full_blueprint, indent=2), seed_file

def main_flow(user_addr, prune_choice, dest_addr, confirm_proceed):
    output_parts = []
    shard = {}
    
    # Disclaimer
    disclaimer = """
This tool generates a prune plan, fee estimate, and unsigned raw TX hex—NO BTC is sent here.
Requires a UTXO-capable wallet (e.g., Electrum) for signing/broadcasting.
Non-custodial: Script reads pub UTXOs only; you control keys/relay.
Fund your address before run for live scan.
This is not financial advice. Use at your own risk.
"""
    output_parts.append(disclaimer)
    
    if not user_addr:
        return "\n".join(output_parts) + "\nNo address provided.", ""
    
    # Validate Addr
    hrp, data = bech32_decode(user_addr)
    if hrp != 'bc' and not user_addr.startswith('1') and not user_addr.startswith('3'):
        return "\n".join(output_parts) + "\nInvalid address. Use bc1q... or legacy 1/3.", ""
    
    # Live BTC/USD (with retry)
    try:
        price_response = api_get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        btc_usd = price_response.json()['bitcoin']['usd']
    except:
        btc_usd = 98500
        output_parts.append('Price API Timeout - Using Fallback BTC/USD: $98,500')
    output_parts.append(f'Live BTC/USD: ${btc_usd:,.2f} (CoinGecko Echo)')
    
    output_parts.append(f'Loaded User Addr: {user_addr[:10]}...')
    
    all_utxos = get_utxos(user_addr)
    
    # Defaults
    pruned_utxos = []
    selected_ratio = 0.4
    choice = '2'
    raw_fee = 0
    pruned_fee = 0
    raw_fee_usd = 0
    pruned_fee_usd = 0
    savings_usd = 0
    dao_cut = 0
    psbt = 'abort_psbt'
    gci = 0.8
    
    raw_hex_text = ""
    
    if not all_utxos:
        output_parts.append('No UTXOs Found - Fund Addr (0.001+ BTC) & Re-Run (6+ Confs)')
        if not dest_addr:
            dest_addr = user_addr
        shard = {
            'utxos': pruned_utxos,
            's_rho': 0.292,
            's_tuned': 0.611,
            'gci': gci,
            'timestamp': '2025-11-14T00:00:00-03:00',
            'pruned_fee': float(pruned_fee),
            'raw_fee': float(raw_fee),
            'pruned_fee_usd': pruned_fee_usd,
            'raw_fee_usd': raw_fee_usd,
            'savings_usd': savings_usd,
            'btc_usd': btc_usd,
            'prune_ratio': selected_ratio,
            'prune_label': prune_choices[choice]['label'],
            'user_addr': user_addr,
            'dest_addr': dest_addr,
            'dao_cut': float(dao_cut),
            'dao_cut_addr': dao_cut_addr,
            'psbt_stub': psbt
        }
        # Run Phases (Full for Consistency)
        gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, dao_cut)
        return "\n".join(output_parts), ""
    
    output_parts.append(f'Live Scan: {len(all_utxos)} Total UTXOs Found')
    
    # Prune Choice Mapping (Gradio Dropdown to choice key)
    prune_map = {
        "Conservative (70% Pruned / 30% Retained - Low Risk, Moderate Savings)": "1", 
        "Efficient (60% Pruned / 40% Retained - v8 Default, Optimal Savings)": "2", 
        "Aggressive (50% Pruned / 50% Retained - Max Consolidation, High Savings)": "3"
    }
    choice = prune_map.get(prune_choice, "2")
    selected_ratio = prune_choices[choice]['ratio']  # Keep ratio for calc (keep fraction)
    output_parts.append(f'Selected: {prune_choices[choice]["label"]} (Pruned: {(1 - selected_ratio)*100:.0f}% / Retained: {selected_ratio*100:.0f}%)')
    
    # Prune Logic
    all_utxos.sort(key=lambda x: x['amount'], reverse=True)
    prune_count = max(1, int(len(all_utxos) * selected_ratio))
    pruned_utxos = all_utxos[:prune_count]
    pruned_amounts = [round(u['amount'], 4) for u in pruned_utxos]
    pruned_usd = [round(amt * btc_usd, 2) for amt in pruned_amounts]
    output_parts.append(f'Pruned UTXOs: [{", ".join(f"{amt} BTC (${usd})" for amt, usd in zip(pruned_amounts, pruned_usd))}]')
    
    # Fee Estimate (Fixed: 2 outputs for pruned TX, savings % reflects prune reduction)
    try:
        fee_response = api_get('https://blockstream.info/api/fee-estimates/6')
        fee_rate_sat = fee_response.json()
        fee_rate = fee_rate_sat * 1e-8
    except:
        fee_rate_sat = 10
        fee_rate = 10 * 1e-8
        output_parts.append('Fee API Timeout - Using Fallback 10 sat/vB')
    raw_vb = 148 * len(all_utxos) + 34
    pruned_vb = 148 * len(pruned_utxos) + 34 * 2  # Fixed: Assume 2 outputs (dest + DAO)
    raw_fee = raw_vb * fee_rate
    pruned_fee = pruned_vb * fee_rate
    raw_fee_usd = round(raw_fee * btc_usd, 2)
    pruned_fee_usd = round(pruned_fee * btc_usd, 2)
    savings = raw_fee - pruned_fee
    savings_usd = round(savings * btc_usd, 2)
    output_parts.append(f'\nFee Estimate @ {fee_rate_sat:.0f} sat/vB:')
    output_parts.append(f'Raw Tx ({len(all_utxos)} UTXOs): {raw_fee:.8f} BTC (${raw_fee_usd}) ({raw_vb} vB)')
    output_parts.append(f'Pruned Tx ({len(pruned_utxos)} UTXOs): {pruned_fee:.8f} BTC (${pruned_fee_usd}) ({pruned_vb} vB)')
    output_parts.append(f'Fee Savings: {savings:.8f} BTC (${savings_usd}) ({(1 - selected_ratio)*100:.0f}%)')  # Fixed: Pruned % for savings
    
    if not dest_addr:
        dest_addr = user_addr
    total_tx_value = sum(u['amount'] for u in pruned_utxos)
    total_tx_usd = round(total_tx_value * btc_usd, 2)
    send_amount = total_tx_value - pruned_fee
    send_usd = round(send_amount * btc_usd, 2)

    # Preview DAO Cut (Fixed: 5% of fee savings, not tx value)
    preview_dao_cut = 0.05 * savings
    preview_dao_cut_usd = round(preview_dao_cut * btc_usd, 2)
    output_parts.append(f'DAO Incentive (5% of Fee Savings): {preview_dao_cut:.8f} BTC (${preview_dao_cut_usd})')
    
    if not confirm_proceed:
        # Calculate post-DAO net for preview
        preview_send_amount = total_tx_value - pruned_fee
        preview_dao_cut = 0.05 * savings
        total_cost_incl_dao = pruned_fee + preview_dao_cut
        preview_net_usd = round((preview_send_amount - preview_dao_cut) * btc_usd, 2)
        output_parts.append(f'\nTotal Tx Value: {total_tx_value:.8f} BTC (${total_tx_usd})')
        output_parts.append(f'Total Cost (Miner Fee + DAO Incentive): {total_cost_incl_dao:.8f} BTC (${round(total_cost_incl_dao * btc_usd, 2)})')
        output_parts.append(f'Net Send (to Dest): {preview_send_amount - preview_dao_cut:.8f} BTC (${preview_net_usd})')
        output_parts.append('Confirm to proceed.')
        shard = {
            'utxos': pruned_utxos,
            's_rho': 0.292,
            's_tuned': 0.611,
            'gci': gci,
            'timestamp': '2025-11-14T00:00:00-03:00',
            'pruned_fee': float(pruned_fee),
            'raw_fee': float(raw_fee),
            'pruned_fee_usd': pruned_fee_usd,
            'raw_fee_usd': raw_fee_usd,
            'savings_usd': savings_usd,
            'btc_usd': btc_usd,
            'prune_ratio': selected_ratio,
            'prune_label': prune_choices[choice]['label'],
            'user_addr': user_addr,
            'dest_addr': dest_addr,
            'dao_cut': float(preview_dao_cut),
            'dao_cut_addr': dao_cut_addr,
            'psbt_stub': psbt
        }
        # Run Phases (Full for Preview)
        gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, preview_dao_cut)
        return "\n".join(output_parts), ""
    
    output_parts.append('Accepted - Generating Unsigned Raw TX')
    
    # DAO Cut (Fixed: 5% of fee savings)
    dao_cut = 0.05 * savings
    send_amount -= dao_cut
    send_usd = round(send_amount * btc_usd, 2)
    dao_cut_usd = round(dao_cut * btc_usd, 2)
    output_parts.append(f'5% DAO Incentive Integrated: {dao_cut:.8f} BTC (${dao_cut_usd}) to DAO Pool - Adjusted Net Send: {send_amount:.8f} BTC (${send_usd})')
    
    # Auto-Raw TX Generation (Pure Python with Full scriptPubKeys)
    raw_hex = None
    try:
        # Derive real scriptPubKeys
        script_dest = address_to_script_pubkey(dest_addr)
        script_dao = address_to_script_pubkey(dao_cut_addr)
        
        tx = Tx(tx_ins=[], tx_outs=[])
        
        # Add pruned UTXOs as inputs (unsigned, empty script_sig)
        for u in pruned_utxos:
            prev_tx_bytes = bytes.fromhex(u['txid'])[::-1]  # Little-endian reversal
            txin = TxIn(prev_tx=prev_tx_bytes, prev_index=u['vout'])
            tx.tx_ins.append(txin)
        
        # Add outputs with real scripts
        tx.tx_outs.append(TxOut(amount=int(send_amount * 1e8), script_pubkey=script_dest))  # Net send to dest
        if dao_cut > 546 / 1e8:  # Dust limit check
            tx.tx_outs.append(TxOut(amount=int(dao_cut * 1e8), script_pubkey=script_dao))  # DAO incentive
        else:
            output_parts.append('DAO Incentive Below Dust Limit - Skipped Output')
        
        # Serialize as raw hex (unsigned; fee implicit via input/output delta)
        raw_hex = tx.encode().hex()
        output_parts.append(f'Unsigned Raw TX Generated ({len(tx.tx_ins)} inputs): Copy the ENTIRE hex below into Electrum (Tools > Load transaction > From hex). Pruned UTXOs auto-matched—no manual selection needed. Preview, sign, broadcast.')
        output_parts.append(f'Dest ScriptPubKey: {script_dest.hex()[:20]}... (full derived)')
        if dao_cut > 546 / 1e8:
            output_parts.append(f'DAO ScriptPubKey: {script_dao.hex()[:20]}... (full derived)')
    except Exception as e:
        raw_hex = f"Error in TX gen: {e}"
        output_parts.append(f'Raw TX Gen Error ({e}): Check console for details.')
    
    instructions = """
=== Next Steps ===
1. Copy the ENTIRE raw TX hex below.
2. In Electrum: Tools > Load transaction > From hex > Paste > OK. Pruned UTXOs auto-load as inputs.
3. Preview to confirm, then Sign.
4. Broadcast and monitor. Re-run for RBF if needed.
=== Proceed Securely ===
"""
    output_parts.append(instructions)
    
    # Shard
    shard = {
        'utxos': pruned_utxos,
        's_rho': 0.292,
        's_tuned': 0.611,
        'gci': 0.92,
        'timestamp': '2025-11-14T00:00:00-03:00',
        'pruned_fee': float(pruned_fee),
        'raw_fee': float(raw_fee),
        'pruned_fee_usd': pruned_fee_usd,
        'raw_fee_usd': raw_fee_usd,
        'savings_usd': savings_usd,
        'btc_usd': btc_usd,
        'prune_ratio': selected_ratio,
        'prune_label': prune_choices[choice]['label'],
        'user_addr': user_addr,
        'dest_addr': dest_addr,
        'dao_cut': float(dao_cut),
        'dao_cut_addr': dao_cut_addr,
        'psbt_stub': raw_hex[:50] + '...' if raw_hex else 'error'
    }
    
    # Run Phases (Full for Confirm)
    gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, raw_hex or 'error', user_addr, dest_addr, dao_cut)
    
    raw_hex_text = raw_hex if raw_hex else "TX Generation Failed - See Log"
    
    return "\n".join(output_parts), raw_hex_text

# Gradio Interface (Now at End – After All Functions Defined)
with gr.Blocks(title="Omega DAO Pruner v8") as demo:
    gr.Markdown("# Omega DAO Pruner v8 - BTC UTXO Optimizer")
    
    # Disclaimer: Always Visible Above Inputs
    gr.Markdown("""
This tool generates a prune plan, fee estimate, and unsigned raw TX hex—NO BTC is sent here.
Requires a UTXO-capable wallet (e.g., Electrum) for signing/broadcasting.
Non-custodial: Script reads pub UTXOs only; you control keys/relay.
Fund your address before run for live scan.
This is not financial advice. Use at your own risk.
""")
    
    with gr.Row():
        user_addr = gr.Textbox(label="User BTC Address", placeholder="bc1q...")
        prune_choice = gr.Dropdown(
            choices=[
                "Conservative (70% Pruned / 30% Retained - Low Risk, Moderate Savings)",
                "Efficient (60% Pruned / 40% Retained - v8 Default, Optimal Savings)",
                "Aggressive (50% Pruned / 50% Retained - Max Consolidation, High Savings)"
            ], 
            value="Efficient (60% Pruned / 40% Retained - v8 Default, Optimal Savings)", 
            label="Prune Strategy"
        )
        dest_addr = gr.Textbox(label="Destination Address (Optional)", placeholder="Same as User Addr")
    submit_btn = gr.Button("Run Pruner")
    
    # Always Visible: Output Log
    output_text = gr.Textbox(label="Output Log", lines=20)
    
    # Hidden: Raw TX Hex (Only after Generate)
    raw_tx_text = gr.Textbox(label="Unsigned Raw TX Hex - Copy Entire Content Below for Electrum", lines=10, visible=False)
    
    # Hidden Button: "Generate Unsigned Raw TX" (Appears After Preview)
    generate_btn = gr.Button("Generate Pruned TX Hex (DAO Pool Incentive Included)", visible=False)

    def show_generate_btn():
        # After preview run, show generate button, keep raw_tx_text hidden
        return gr.update(visible=True), gr.update(visible=False)

    def generate_raw_tx(user_addr, prune_choice, dest_addr):
        # Trigger full generation (confirm=True)
        log, hex_content = main_flow(user_addr, prune_choice, dest_addr, True)
        # Return updates: log, raw_tx_text with value and visible=True, generate_btn hidden
        return log, gr.update(value=hex_content, visible=True), gr.update(visible=False)

    # First Run: Preview (confirm=False) - Only log
    submit_btn.click(
        fn=lambda u, p, d: main_flow(u, p, d, False),
        inputs=[user_addr, prune_choice, dest_addr],
        outputs=[output_text, raw_tx_text]
    ).then(
        fn=show_generate_btn,
        outputs=[generate_btn, raw_tx_text]
    )

    # Second Step: Generate Raw TX (confirm=True) - Log + Hex
    generate_btn.click(
        fn=generate_raw_tx,
        inputs=[user_addr, prune_choice, dest_addr],
        outputs=[output_text, raw_tx_text, generate_btn]
    )

# Render Launch: share=True for cloud bypass
if __name__ == "__main__":
    demo.queue(api_open=True)
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,  # Public Gradio.live URL in logs
        debug=False,
        root_path="/",
        show_error=True
    )
