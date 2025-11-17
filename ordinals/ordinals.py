import gradio as gr
import json
import numpy as np
import requests
import os
import base64
import io
import re
import time  # Added for retries
GROK_API_KEY = os.getenv('GROK_API_KEY')
print(f"GROK_API_KEY flux: {'Eternal' if GROK_API_KEY else 'Void—fallback active'}")  # Echo here
if GROK_API_KEY:
    print("Grok requests summoned eternal—n=500 hooks ready.")  # Your line, forced

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

# FIXED: Define bech32m_verify_checksum BEFORE bech32_decode to avoid NameError
def bech32m_verify_checksum(hrp, data):
    return bech32_polymod(bech32_hrp_expand(hrp) + data) == 0x2bc830a3  # Bech32m constant

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
    # Route to Bech32 or Bech32m based on first char after '1' ('q' = v0, 'p' = v1 Taproot)
    if addr[pos+1] == 'q':
        if not bech32_verify_checksum(hrp, data):
            return None, None
    elif addr[pos+1] == 'p':
        if not bech32m_verify_checksum(hrp, data):
            return None, None
    else:
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

# ENHANCED: address_to_script_pubkey returns (script, vb_weights dict)
def address_to_script_pubkey(addr):
    if addr.startswith('bc1q'):
        hrp, data5 = bech32_decode(addr)
        if hrp == 'bc' and data5 and data5[0] == 0:  # v0
            data8 = convertbits(data5[1:], 5, 8, False)
            if data8:
                if len(data8) == 20:
                    script = bytes([0x00, 0x14]) + bytes(data8)
                    weights = {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WPKH'}
                elif len(data8) == 32:
                    script = bytes([0x00, 0x20]) + bytes(data8)
                    weights = {'input_vb': 67.25, 'output_vb': 31, 'type': 'P2WSH'}
                else:
                    raise ValueError("Invalid witness program length")
                return script, weights
    elif addr.startswith('bc1p'):
        hrp, data5 = bech32_decode(addr)
        if hrp == 'bc' and data5 and data5[0] == 1:  # v1 Taproot
            data8 = convertbits(data5[1:], 5, 8, False)
            if data8 and len(data8) == 32:
                script = bytes([0x51, 0x20]) + bytes(data8)  # OP_1 PUSH32
                weights = {'input_vb': 57.25, 'output_vb': 43, 'type': 'P2TR'}
                return script, weights
    elif addr.startswith('1'):
        # P2PKH
        decoded = base58_decode(addr)
        if len(decoded) == 25 and decoded[0] == 0x00:
            payload = decoded[1:21]
            script = bytes([0x76, 0xa9, 0x14]) + payload + bytes([0x88, 0xac])
            weights = {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
            return script, weights
    elif addr.startswith('3'):
        # P2SH
        decoded = base58_decode(addr)
        if len(decoded) == 25 and decoded[0] == 0x05:
            payload = decoded[1:21]
            script = bytes([0xa9, 0x14]) + payload + bytes([0x87])
            weights = {'input_vb': 148, 'output_vb': 34, 'type': 'P2SH'}
            return script, weights
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

# ROBUST get_utxos: Blockstream primary, Mempool fallback, Hiro fixed limit=50
def get_utxos(addr, dust_threshold=546, current_height=None):
    utxos_raw = []
    try:
        if current_height is None:
            # Primary: Blockstream tip (reliable)
            tip_response = api_get('https://blockstream.info/api/blocks/tip/height')
            current_height = tip_response.json()
            print(f'Block Height (Blockstream): {current_height}')
        # Primary: Blockstream UTXOs (singular /utxo)
        utxo_response = api_get(f'https://blockstream.info/api/address/{addr}/utxo')
        utxos_raw = utxo_response.json()
        print(f'Raw UTXOs Fetched (Blockstream): {len(utxos_raw)}')
    except Exception as e:
        print(f'Blockstream Fail for {addr[:10]}...: {e} - Falling to Mempool')
        try:
            if current_height is None:
                tip_response = api_get('https://mempool.space/api/blocks/tip/height')
                current_height = tip_response.json()
            utxo_response = api_get(f'https://mempool.space/api/address/{addr}/utxos')  # Plural for Mempool
            utxos_raw = utxo_response.json()
            print(f'Raw UTXOs Fetched (Mempool Fallback): {len(utxos_raw)}')
        except Exception as e2:
            print(f'Mempool Fallback Fail: {e2}')
            return [], None

    # Ordinals: FIXED limit=50 (max <60)
    inscriptions = []
    try:
        ordinals_response = api_get(f'https://api.hiro.so/ordinals/v1/inscriptions?address={addr}&limit=50', timeout=60)  # Bump eternal
        inscriptions = ordinals_response.json().get('results', [])
        print(f'Inscriptions Fetched (Hiro): {len(inscriptions)}')
    except Exception as e:
        print(f'Hiro Ordinals Fail for {addr[:10]}...: {e} - No Flags')

    # Filter + Flag
    filtered_utxos = []
    for utxo in utxos_raw:
        if utxo['status']['confirmed']:
            confs = current_height - utxo['status']['block_height'] + 1
            if confs > 6 and utxo['value'] > dust_threshold:
                inscription_flag = any(ins['tx_id'] == utxo['txid'] and ins['output'] == utxo['vout'] for ins in inscriptions)
                filtered_utxos.append({
                    'txid': utxo['txid'],
                    'vout': utxo['vout'],
                    'amount': utxo['value'] / 1e8,
                    'address': addr,
                    'confs': confs,
                    'is_inscription': inscription_flag,
                    'dust_risk': utxo['value'] < (dust_threshold * 1.1)
                })
    # Sort: Prioritize inscriptions
    filtered_utxos.sort(key=lambda u: (u['is_inscription'], u['amount']), reverse=True)
    
    inscription_count = sum(1 for u in filtered_utxos if u['is_inscription'])
    print(f'API Success for {addr[:10]}...: {len(filtered_utxos)} UTXOs ({inscription_count} inscriptions) >6 confs (from {len(utxos_raw)} raw)')
    return filtered_utxos, current_height

prune_choices = {
    '1': {'label': 'Conservative: 70/30 Prune (Low Risk)', 'ratio': 0.3},
    '2': {'label': 'Efficient: 60/40 Prune (Default)', 'ratio': 0.4},
    '3': {'label': 'Aggressive: 50/50 Prune (Max Savings)', 'ratio': 0.5}
}

# Pure Python TX Builder (unchanged)
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
    sequence: int = 0xfffffffd  # RBF-enabled

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

# HELPER: PHASE 1-3 Logic (unchanged)
def run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, dao_cut):
    # Detect prod env (e.g., Render) to skip qutip for fast builds
    is_prod = bool(os.getenv('RENDER_EXTERNAL_HOSTNAME') or os.getenv('PORT'))
    
    # PHASE 1: QuTiP Tune (Conditional)
    if pruned_utxos:
        if is_prod:
            # Mock for prod: Use fallback values with slight randomization for variety
            import random
            s_rho = 0.292 + random.uniform(-0.01, 0.01)
            s_tuned = 0.611 + random.uniform(-0.02, 0.02)
            print(f'Prod Mode: qutip Mocked - Initial S(ρ) [BTC Flux Void]: {s_rho:.3f}')
            print(f'Prod Mode: qutip Mocked - Tuned S(ρ) [Coherence Surge]: {s_tuned:.3f}')
        else:
            # Lazy import QuTiP here to speed global startup
            import qutip as qt
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

# Hybrid Hook: Tune GCI with Grok n=10 (test scale)
def grok_tune(gci_base):
    if not GROK_API_KEY:
        print("API key void—fallback GCI tune")
        return gci_base
    headers = {
        'Authorization': f'Bearer {GROK_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'grok-4-0709',  # Sharper eternal
        'messages': [{'role': 'user', 'content': f'Tune GCI {gci_base} for Ω mempool prune—output QuTiP params (p=0.389, S(ρ)=0.611) vs. Lightning baselines.'}]
    }
    for attempt in range(3):  # Retry eternal, 3 tries
        try:
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60  # Bumped eternal—no choke
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Robust parse: Extract first float from content (e.g., "0.92" in text)
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    tuned_gci = float(numbers[0])  # First number eternal
                    print(f"Grok tuned: {tuned_gci:.3f}—edges vs. Lightning eternal")
                    return tuned_gci
                else:
                    print(f"Grok content raw: {content[:200]}...—no number, fallback")
                    return gci_base
            else:
                print(f"Grok flux error: {response.status_code}—body: {response.text[:200]}...")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Backoff eternal
                else:
                    return gci_base
        except requests.exceptions.ReadTimeout:
            print(f"ReadTimeout on attempt {attempt+1}/3—retrying eternal...")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Backoff eternal
            else:
                print("Retries exhausted—fallback GCI tune")
                return gci_base
    return gci_base  # Final fallback eternal

# ENHANCED main_flow: Proper Taproot detect, dynamic vB, fixed prune_map, dust via get_utxos

# Test in main_flow (post-QuTiP)
def main_flow(user_addr, prune_choice, dest_addr, confirm_proceed, dust_threshold=546):
    output_parts = []
    shard = {}
    
    # Disclaimer
    disclaimer = """
This tool generates a prune plan, fee estimate, and unsigned raw TX hex—NO BTC is sent here.

Taproot (bc1p) and Ordinals-compatible for modern stacks. Dust threshold: Configurable (default 546 sats) to exclude/batch tiny UTXOs—lower for risk-tolerant inscription consolidation when fees are low (<2 sat/vB).

Requires a UTXO-capable wallet (e.g., Electrum or Sparrow) for signing/broadcasting.

Non-custodial: Script reads pub UTXOs only; you control keys/relay.

Fund your address before run for live scan.

This is not financial advice. Use at your own risk.

⚠️ Processing Note: For addresses with a lot of UTXOs (e.g., 50+), fetching and analysis may take 1-3+ minutes (up to 200s on busy networks). Be patient. If stuck >5 min, refresh and try again. 

Contact: omegadaov8@proton.me
    """
    output_parts.append(disclaimer)
    
    if not user_addr:
        return "\n".join(output_parts) + "\nNo address provided.", ""
    
    # Validate Addr
    hrp, data = bech32_decode(user_addr)
    print(f"Debug: Addr='{user_addr}' | HRP={hrp} | Start1={user_addr.startswith('1')} | Start3={user_addr.startswith('3')} | Startbc1p={user_addr.startswith('bc1p')}")  # Echo eternal

    if hrp != 'bc' and not user_addr.startswith('1') and not user_addr.startswith('3') and not user_addr.startswith('bc1p'):
        print("Debug: Condition True—Invalid return")
        return "\n".join(output_parts) + "\nInvalid address. Use bc1q/bc1p... or legacy 1/3.", ""
    print("Debug: Condition False—Validation passed eternal")
    
    # FIXED: Proper Taproot/SegWit detection
    is_taproot = user_addr.startswith('bc1p')
    is_segwit = user_addr.startswith('bc1q')
    if is_taproot:
        output_parts.append('Taproot (bc1p) Detected: Ordinals/Inscription-Ready (57.25 vB/input savings!)')
        addr_type = "Taproot P2TR"
    elif is_segwit:
        addr_type = "SegWit P2WPKH/P2WSH"
    else:
        addr_type = "Legacy P2PKH/P2SH"
    
    # Derive weights for user_addr
    try:
        _, user_vb_weights = address_to_script_pubkey(user_addr)
        input_vb = user_vb_weights['input_vb']
        output_vb = user_vb_weights['output_vb']
    except:
        # Fallback
        if is_taproot:
            input_vb, output_vb = 57.25, 43
        elif is_segwit:
            input_vb, output_vb = 67.25, 31
        else:
            input_vb, output_vb = 148, 34
    
    # Live BTC/USD (with retry)
    try:
        price_response = api_get('https://api.coinpaprika.com/v1/tickers/btc-bitcoin')
        btc_usd = price_response.json()['quotes']['USD']['price']
    except:
        btc_usd = 98500  # Fallback eternal
        output_parts.append('Price API Timeout - Using Fallback BTC/USD: $98,500')
    output_parts.append(f'Live BTC/USD: ${btc_usd:,.2f} (CoinGecko Echo)')
    
    output_parts.append(f'Loaded User Addr: {user_addr[:10]}... ({addr_type})')
    
    # REUSED: Fetch via get_utxos for consistency (robust now)
    all_utxos, current_height = get_utxos(user_addr, dust_threshold)
    
    # ENHANCED Dust Stats: Refetch raw (dust_threshold=0) to count excluded
    raw_utxos, _ = get_utxos(user_addr, 0, current_height)
    dust_utxos = []
    inscriptions = []  # From Hiro in get_utxos, but for dust: assume from raw
    # Simple: Filter raw_utxos for <=dust_threshold, >6 confs
    for u in raw_utxos:
        confs = current_height - u['status']['block_height'] + 1 if 'status' in u else 0
        if confs > 6 and u['value'] <= dust_threshold:
            dust_utxos.append(u)
    dust_count = len(dust_utxos)
    dust_value = sum(u['value'] for u in dust_utxos)
    # Dust inscriptions: Placeholder (full impl would refetch Hiro)
    dust_inscriptions = 0  # For now; extend if needed
    
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
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
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
            'psbt_stub': psbt,
            'dust_threshold': dust_threshold,
            'addr_type': addr_type
        }
        # Run Phases (Full for Consistency)
        gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, dao_cut)
        return "\n".join(output_parts), ""
    
    output_parts.append(f'Live Scan: {len(all_utxos)} Total UTXOs Found')
    
    # FIXED prune_map for short labels
    prune_map = {
        "Conservative (70/30, Low Risk)": "1", 
        "Efficient (60/40, Default)": "2", 
        "Aggressive (50/50, Max Savings)": "3"
    }
    choice = prune_map.get(prune_choice, "2")
    selected_ratio = prune_choices[choice]['ratio']
    output_parts.append(f'Selected: {prune_choices[choice]["label"]} (Pruned: {(1 - selected_ratio)*100:.0f}% / Retained: {selected_ratio*100:.0f}%)')
    
    # Prune Logic - Enhanced Sort (Inscriptions prioritized)
    all_utxos.sort(key=lambda x: (x.get('is_inscription', False), x['amount']), reverse=True)
    prune_count = max(1, int(len(all_utxos) * selected_ratio))
    pruned_utxos = all_utxos[:prune_count]
    pruned_amounts = [round(u['amount'], 4) for u in pruned_utxos]
    pruned_usd = [round(amt * btc_usd, 2) for amt in pruned_amounts]
    output_parts.append(f'Pruned UTXOs: [{", ".join(f"{amt} BTC (${usd})" for amt, usd in zip(pruned_amounts, pruned_usd))}]')
    # Ordinals/Inscription Tease
    inscription_count = sum(1 for u in pruned_utxos if u.get('is_inscription', False))
    if inscription_count > 0 or dust_count > 0:
        output_parts.append(f'Inscribed UTXOs Pruned: {inscription_count} (Ordinals/BRC-20 dust prioritized at {dust_threshold} sat threshold)')
        # Dust Threshold Disclaimer
        output_parts.append(f'''
⚠️ Dust Threshold: Set to {dust_threshold} sats. Lower values batch more tiny inscriptions for optimization (e.g., when fees <2 sat/vB), but risk net losses if fees spike. Excluded Dust UTXOs: {dust_count} (handle manually in Sparrow/Electrum). Not financial advice—sim fees first!
        ''')
        # Quick vB Calc for Dust Batch Cost (dynamic vb)
        try:
            fee_response = api_get('https://mempool.space/api/v1/fees/recommended')  # Live eternal, no 404
            fee_rate_sat = fee_response.json()['economyFee']  # Parse sat/vB
        except:
            fee_rate_sat = 10
        overhead_vb = 10
        dust_batch_vb = overhead_vb + input_vb * dust_count + output_vb * 1  # Batch to 1 output
        dust_batch_fee = dust_batch_vb * (fee_rate_sat * 1e-8)
        dust_batch_fee_sats = int(dust_batch_fee * 1e8)
        dust_value_sats = int(dust_value)
        output_parts.append(f'💡 Est. Dust Batch Cost: {dust_batch_vb:.1f} vB ({dust_batch_fee_sats} sats @ {fee_rate_sat:.0f} sat/vB) for {dust_count} UTXOs ({dust_inscriptions} inscribed). Total dust value: {dust_value_sats} sats. Net worth it if future fee savings > batch cost!')
    elif dust_count > 0:
        output_parts.append(f'Dust UTXOs Detected (but not pruned): {dust_count} below {dust_threshold} sats threshold.')
    
    # Fee Estimate (Dynamic vB)
    try:
        fee_response = api_get('https://mempool.space/api/v1/fees/recommended')  # Live eternal, no 404
        fee_rate_sat = fee_response.json()['economyFee']  # Parse sat/vB
        fee_rate = fee_rate_sat * 1e-8
    except:
        fee_rate_sat = 10
        fee_rate = 10 * 1e-8
        output_parts.append('Fee API Timeout - Using Fallback 10 sat/vB')
    
    addr_note = f"({addr_type})"
    overhead_vb = 10
    raw_vb = overhead_vb + input_vb * len(all_utxos) + output_vb * 1  # Raw: all inputs, 1 output
    pruned_vb = overhead_vb + input_vb * len(pruned_utxos) + output_vb * 2  # Pruned: few inputs, 2 outputs (dest + DAO)
    
    raw_fee = raw_vb * fee_rate
    pruned_fee = pruned_vb * fee_rate
    raw_fee_usd = round(raw_fee * btc_usd, 2)
    pruned_fee_usd = round(pruned_fee * btc_usd, 2)
    savings = raw_fee - pruned_fee
    savings_usd = round(savings * btc_usd, 2)
    
    if not dest_addr:
        dest_addr = user_addr
    total_tx_value = sum(u['amount'] for u in pruned_utxos)
    total_tx_usd = round(total_tx_value * btc_usd, 2)
    send_amount = total_tx_value - pruned_fee
    send_usd = round(send_amount * btc_usd, 2)

    # Preview DAO Cut (Fixed: 5% of fee savings, not tx value)
    preview_dao_cut = 0.05 * savings
    preview_dao_cut_usd = round(preview_dao_cut * btc_usd, 2)
    output_parts.append(f'\nFee Estimate @ {fee_rate_sat:.0f} sat/vB for transaction ({total_tx_value:.8f} BTC (${total_tx_usd:,.2f})) {addr_note}:')
    output_parts.append(f'Raw Tx ({len(all_utxos)} UTXOs): {raw_fee:.8f} BTC (${raw_fee_usd}) ({raw_vb:.1f} vB)')
    output_parts.append(f'Pruned Tx ({len(pruned_utxos)} UTXOs): {pruned_fee:.8f} BTC (${pruned_fee_usd}) ({pruned_vb:.1f} vB)')
    output_parts.append(f'Fee Savings: {savings:.8f} BTC (${savings_usd}) ({(1 - selected_ratio)*100:.0f}%)')
    
    output_parts.append(f'DAO Incentive (5% of Fee Savings): {preview_dao_cut:.8f} BTC (${preview_dao_cut_usd})')
    
    # Run Phases first to set shard['gci']
    shard = {
        'utxos': pruned_utxos,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
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
        'dust_threshold': dust_threshold,
        'addr_type': addr_type,
        'psbt_stub': psbt,
        'dao_cut': 0,
        'dao_cut_addr': dao_cut_addr
    }
    gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, psbt, user_addr, dest_addr, 0)
    
    # Test in main_flow (post-QuTiP)
    print("Calling grok_tune—flux incoming")  # Force call echo
    shard['grok_tuned_gci'] = grok_tune(shard['gci'])  # n=1 burn, scale to 10
    print(f"Shard updated: Grok tuned GCI {shard['grok_tuned_gci']:.3f}")
    
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
        shard['dao_cut'] = float(preview_dao_cut)
        shard['dao_cut_addr'] = dao_cut_addr
        shard['psbt_stub'] = psbt
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
        # Derive real scriptPubKeys (unpack weights if needed, but not for TX)
        script_dest, _ = address_to_script_pubkey(dest_addr)
        script_dao, _ = address_to_script_pubkey(dao_cut_addr)
        
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
        output_parts.append(f'Unsigned Raw TX Generated ({len(tx.tx_ins)} inputs, {len(tx.tx_outs)} outputs: net send to dest + DAO incentive): Copy the ENTIRE hex below into Electrum (Tools > Load transaction > From hex). Pruned UTXOs auto-matched—no manual selection needed. Preview, sign, broadcast.')
        output_parts.append(f'Dest ScriptPubKey: {script_dest.hex()[:20]}... (full derived)')
        if dao_cut > 546 / 1e8:
            output_parts.append(f'DAO ScriptPubKey: {script_dao.hex()[:20]}... (full derived)')
        output_parts.append('This transaction is RBF-enabled (sequence: 0xfffffffd), allowing you to bump the fee in your wallet if it gets stuck in the mempool.')
    except Exception as e:
        raw_hex = f"Error in TX gen: {e}"
        output_parts.append(f'Raw TX Gen Error ({e}): Check console for details.')
    
    instructions = """
=== Next Steps ===
1. Copy the ENTIRE raw TX hex below.
2. In Electrum: Tools > Load transaction > From hex > Paste > OK. Pruned UTXOs auto-load as inputs, net send + DAO incentive auto-load as outputs.
3. Preview to confirm (TX is RBF-enabled: sequence signals allow fee bumps without re-signing).
5. Sign.
6. Broadcast and monitor.  If unconfirmed (low fee/mempool full), right-click TX in Electrum > "Increase fee" for RBF bump (auto-higher rate; DAO unchanged). Re-run pruner for major tweaks (e.g., new UTXOs/strategy—will recalc fees/DAO).
7. Enjoy.
=== Proceed Securely ===
"""
    output_parts.append(instructions)
    
    # Shard
    shard['dao_cut'] = float(dao_cut)
    shard['dao_cut_addr'] = dao_cut_addr
    shard['psbt_stub'] = raw_hex[:50] + '...' if raw_hex else 'error'
    
    # Run Phases (Full for Confirm)
    gci, full_bp, seed_file = run_phases(shard, pruned_utxos, selected_ratio, raw_fee, pruned_fee, savings_usd, btc_usd, choice, gci, raw_hex or 'error', user_addr, dest_addr, dao_cut)
    
    raw_hex_text = raw_hex if raw_hex else "TX Generation Failed - See Log"
    
    return "\n".join(output_parts), raw_hex_text

# Gradio Interface (unchanged)
with gr.Blocks(title="Omega DAO Pruner v8.1") as demo:
    gr.Markdown("# Omega DAO Pruner v8.1 - BTC UTXO Optimizer")
    
    # Disclaimer: Always Visible Above Inputs
    gr.Markdown("""
This tool generates a prune plan, fee estimate, and unsigned raw TX hex—NO BTC is sent here.

Taproot (bc1p) and Ordinals-compatible for modern stacks. Dust threshold: Configurable (default 546 sats) to exclude/batch tiny UTXOs—lower for risk-tolerant inscription consolidation when fees are low (<2 sat/vB).

Requires a UTXO-capable wallet (e.g., Electrum or Sparrow) for signing/broadcasting.

Non-custodial: Script reads pub UTXOs only; you control keys/relay.

Fund your address before run for live scan.

This is not financial advice. Use at your own risk.

⚠️ Processing Note: For addresses with a lot of UTXOs (e.g., 50+), fetching and analysis may take 1-3+ minutes (up to 200s on busy networks). Be patient. If stuck >5 min, refresh and try again. 

Contact: omegadaov8@proton.me
""")
    
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
        # Dust Threshold Slider
        dust_threshold = gr.Slider(minimum=0, maximum=2000, value=546, step=1, label="Dust Threshold (sats) - Lower = Include More Dust (Riskier Batching)")
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

    def generate_raw_tx(user_addr, prune_choice, dust_threshold, dest_addr):
        log, hex_content = main_flow(user_addr, prune_choice, dest_addr, True, dust_threshold)
        return log, gr.update(value=hex_content, visible=True), gr.update(visible=False)

    # First Run: Preview (confirm=False) - Only log
    submit_btn.click(
        fn=lambda u, p, dt, d: main_flow(u, p, d, False, dt),
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr],
        outputs=[output_text, raw_tx_text]
    ).then(
        fn=show_generate_btn,
        outputs=[generate_btn, raw_tx_text]
    )

    # Second Step: Generate Raw TX (confirm=True) - Log + Hex
    generate_btn.click(
        fn=generate_raw_tx,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr],
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
