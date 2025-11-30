# Omega_v10_pt.py — Versión en español
import gradio as gr
import requests, time, base64, io, qrcode
from dataclasses import dataclass
from typing import List, Tuple, Optional
import urllib.parse
import warnings
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning)

print(f"Gradio version: {gr.__version__}")

# ==============================
# Optional deps
# ==============================

try:
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC as HDWALLET_BTC
except ImportError:
    HDWallet = None
    HDWALLET_BTC = None

# ==============================
# Constants
# ==============================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"
pruned_utxos_global = None
input_vb_global = output_vb_global = None

# ==============================
# CSS
# ==============================
css = """
/* —————————————————————— ΩMEGA PRUNER v10 CSS —————————————————————— */

/* 1. SANE, BEAUTIFUL GAPS — GRADIO 6+ FIX */
.gr-row { gap: 14px !important; }
.gr-row:has(.full-width),
.gr-row:has(.bump-with-gap),
.gr-row:has(.gr-button.size-lg) { gap: 16px !important; }
#generate-and-startover-row { gap: 22px !important; }

/* Kill rogue margins/padding */
.full-width, .full-width > div, .full-width button,
.bump-with-gap, .bump-with-gap > div, .bump-with-gap button {
    margin: 0 !important; padding: 0 !important;
}

/* 2. BEEFY PREMIUM BUTTONS */
.gr-button button, .gr-button > div, .gr-button > button,
.gr-button [class*="svelte"], button[class*="svelte"] {
    font-size: 1.25rem !important; font-weight: 600 !important;
    padding: 16px 28px !important; min-height: 62px !important;
    border-radius: 14px !important; box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    transition: all 0.22s ease !important; line-height: 1.4 !important;
    width: 100% !important; text-align: center !important;
}
.gr-button[variant="primary"], .gr-button.size-lg,
.full-width, .bump-with-gap, .tall-button {
    font-size: 1.38rem !important; font-weight: 750 !important;
    padding: 22px 32px !important; min-height: 72px !important;
    box-shadow: 0 6px 20px rgba(247,147,26,0.38) !important;
}
.gr-button[variant="secondary"] button,
.gr-button[variant="secondary"] > button {
    font-size: 1.28rem !important; font-weight: 600 !important;
    padding: 18px 28px !important; min-height: 64px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
.gr-button:hover button, .gr-button:hover > button, .gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22) !important;
}
.gr-button[variant="primary"]:hover, .gr-button.size-lg:hover {
    box-shadow: 0 14px 32px rgba(247,147,26,0.5) !important;
    transform: translateY(-4px) !important;
}

/* 3. MISC FIXES */
details summary { list-style: none; cursor: pointer; }
details summary::-webkit-details-marker { display: none; }

/* ——— FAB BUTTONS ——— */
.qr-fab {
  position: fixed !important; right: 20px !important;
  width: 70px !important; height: 70px !important;
  border-radius: 50% !important;
  box-shadow: 0 10px 40px rgba(0,0,0,0.7) !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  font-size: 38px !important; font-weight: bold !important; cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
  border: 5px solid white !important; user-select: none !important;
  text-shadow: 0 2px 8px rgba(0,0,0,0.5) !important; z-index: 9999 !important;
  animation: pulse 4s infinite ease-in-out !important;   /* ← SUBTLE PULSE */
}
.qr-fab:hover {
  transform: scale(1.18) !important;
  box-shadow: 0 16px 50px rgba(0,0,0,0.8) !important;
  animation: none !important;   /* stop pulse on hover → feels snappier */
}
.qr-fab.btc  { bottom: 100px !important; background: linear-gradient(135deg, #f7931a, #f9a43f) !important; color: white !important; }

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50%      { transform: scale(1.08); }
}

/* ——— DIM GIANT Ω WHEN TYPING ——— */
input:focus ~ #omega-bg-container-fixed,
textarea:focus ~ #omega-bg-container-fixed,
input:focus-within ~ #omega-bg-container-fixed,
textarea:focus-within ~ #omega-bg-container-fixed {
    opacity: 0.22 !important;
    transition: opacity 0.5s ease !important;
}
/* ——— KEEP GRADIO'S NATIVE BOTTOM BUTTONS SKINNY & NORMAL ——— */
.gradio-container .bottom-buttons .gr-button,
.gradio-container footer .gr-button,
.gradio-container button[data-testid="block-settings"],
.gradio-container button[title="Show API"],
.gradio-container button[title="View API"],
.gradio-container button[title="Clear"],
.gradio-container button[title="Stop"] {
    all: revert !important;
    font-size: 0.9rem !important;
    padding: 8px 14px !important;
    min-height: auto !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ——— QR CENTERING & STYLING ——— */
.qr-center {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  margin: 40px 0 !important;
}
.qr-center img {
  width: 460px !important;
  max-width: 96vw !important;
  border-radius: 20px !important;
  border: 6px solid #f7931a !important;
  box-shadow: 0 12px 50px rgba(247,147,26,0.6) !important;
}
#omega-footer {
    margin-bottom: -10px !important;
    padding-bottom: 4px !important;
}
.gradio-container .gradio-footer,
.gradio-container footer {
    display: none !important;   /* nukes Gradio's own footer completely */
}
@media (max-width: 768px) {
    .qr-fab { bottom: 80px !important; right: 16px !important; width: 64px !important; height: 64px !important; font-size: 34px !important; }
    .qr-center img { width: 380px !important; }
}

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
                return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}

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
        xpub_clean = xpub.strip()

        # === Try to import hdwallet gracefully ===
        try:
            from hdwallet import HDWallet
            from hdwallet.symbols import BTC as HDWALLET_BTC
        except ImportError:
            return [], "Missing dependency: pip install hdwallet"

        # === Auto-detect xpub type and set correct derivation path ===
        if xpub_clean.startswith("zpub") or xpub_clean.startswith("vpub"):
            path_prefix = "m/84'/0'/0'"   # Native SegWit (bc1q)
        elif xpub_clean.startswith("ypub") or xpub_clean.startswith("upub"):
            path_prefix = "m/49'/0'/0'"   # Nested SegWit (P2SH-P2WPKH, starts with 3)
        elif xpub_clean.startswith("xpub"):
            path_prefix = "m/44'/0'/0'"   # Legacy (starts with 1) — fallback
        else:
            return [], "Unsupported xpub prefix (use xpub/ypub/zpub)"

        hdw = HDWallet(symbol=HDWALLET_BTC)
        hdw.from_xpublic_key(xpub_clean)

        addresses = []
        receive_chain = 0
        change_chain = 1
        max_per_chain = 100
        gap_limit = 20

        def scan_chain(chain: int):
            empty_count = 0
            for i in range(max_per_chain):
                path = f"{path_prefix}/{chain}/{i}"
                try:
                    if path_prefix == "m/84'/0'/0'":
                        addr = hdw.from_path(path).p2wpkh_address()
                    elif path_prefix == "m/49'/0'/0'":
                        addr = hdw.from_path(path).p2sh_p2wpkh_address()
                    else:  # m/44'
                        addr = hdw.from_path(path).p2pkh_address()
                except:
                    addr = None

                if not addr:
                    break
                addresses.append(addr)

                # Early exit if gap limit reached
                if i >= gap_limit - 1:
                    recent = addresses[-(gap_limit):]
                    if all(len(get_utxos(a, dust)) == 0 for a in recent):
                        empty_count = gap_limit
                        break

        # Scan receive (0) and change (1)
        scan_chain(receive_chain)
        scan_chain(change_chain)

        # Dedupe just in case
        addresses = list(dict.fromkeys(addresses))[:200]

        all_utxos = []
        for addr in addresses:
            try:
                utxos = api_get(f"https://blockstream.info/api/address/{addr}/utxo")
            except:
                try:
                    utxos = api_get(f"https://mempool.space/api/address/{addr}/utxo")
                except:
                    continue
            confirmed = [u for u in utxos if u.get('status', {}).get('confirmed', True)]
            all_utxos.extend([u for u in confirmed if u['value'] > dust])
            time.sleep(0.08)  # Be nice to public APIs

        all_utxos.sort(key=lambda x: x['value'], reverse=True)
        scanned = len(addresses)
        found = len(all_utxos)

        addr_type = "Native SegWit" if "84'" in path_prefix else "Nested SegWit" if "49'" in path_prefix else "Legacy"
        return all_utxos, f"Scanned {scanned} addresses ({addr_type}) → Found {found} UTXOs"

    except Exception as e:
        return [], f"xpub error: {str(e)}"

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
    
def _correct_tx_encode(self, segwit=True):
    base = [
        self.version.to_bytes(4, 'little'),
        b'\x00\x01' if segwit else b'',  # marker + flag
        encode_varint(len(self.tx_ins)),
        b''.join(inp.encode() for inp in self.tx_ins),
        encode_varint(len(self.tx_outs)),
        b''.join(out.encode() for out in self.tx_outs),
        self.locktime.to_bytes(4, 'little')
    ]
    raw = b''.join(base)
    
    if segwit:
        raw += b'\x00' * len(self.tx_ins)  # ← ONE \x00 PER INPUT = empty witness stack
    
    return raw

Tx.encode = _correct_tx_encode
del _correct_tx_encode

def make_psbt(tx: Tx) -> str:
    raw = tx.encode(segwit=True)
    
    global_tx = b'\x00' + encode_varint(len(raw)) + raw + b'\x00'
    psbt = b'psbt\xff' + global_tx + b'\xff'
    return base64.b64encode(psbt).decode()
# =================================================================

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
# Core Functions
# ==============================
def analysis_pass(user_input, strategy, threshold, dest_addr, dao_percent, future_multiplier):
    global pruned_utxos_global, input_vb_global, output_vb_global

    addr = user_input.strip()
    is_xpub = addr.startswith(('xpub', 'zpub', 'ypub', 'tpub'))

    if is_xpub:
        utxos, msg = fetch_all_utxos_from_xpub(addr, threshold)
        if not utxos:
            return msg or "Erro ao escanear xpub", gr.update(visible=False)
    else:
        if not addr:
            return "Cole uma carteira ou xpub", gr.update(visible=False)
        utxos = get_utxos(addr, threshold)
        if not utxos:
            return "Nenhum UTXO acima do limite de pó", gr.update(visible=False)

    utxos.sort(key=lambda x: x['value'], reverse=True)

    # Detect script type
    sample = [u.get('address') or addr for u in utxos[:10]]
    types = [address_to_script_pubkey(a)[1]['type'] for a in sample]
    from collections import Counter
    detected = Counter(types).most_common(1)[0][0] if types else "SegWit"

    vb_map = {
        'P2PKH': (148, 34),
        'P2SH': (91, 32),
        'SegWit': (68, 31),
        'Taproot': (57, 43)
    }
    input_vb_global, output_vb_global = vb_map.get(detected.split()[0], (68, 31))

    # === MAPEAMENTO DAS ESTRATÉGIAS (tem que bater 100% com o dropdown) ===
    NUCLEAR_OPTION = "NUCLEAR PRUNE (90% sacrificado — só pra quem tem culhão)"

    ratio_map = {
        "Privacidade Primeiro (30% podado)": 0.3,
        "Recomendado (40% podado)": 0.4,
        "Mais Economia (50% podado)": 0.5,
        NUCLEAR_OPTION: 0.9,
    }
    ratio = ratio_map.get(strategy, 0.4)

    name_map = {
        "Privacidade Primeiro (30% podado)": "Privacidade Primeiro",
        "Recomendado (40% podado)": "Recomendado",
        "Mais Economia (50% podado)": "Mais Economia",
        NUCLEAR_OPTION: '<span style="color:#ff1361; font-weight:900; text-shadow: 0 0 10px #ff0066;">NUCLEAR PRUNE</span>',
    }
    strategy_name = name_map.get(strategy, strategy.split(" (")[0])

    # === NUCLEAR VERDADEIRO: máximo 3 inputs mesmo com 5000 UTXOs ===
    if strategy == NUCLEAR_OPTION:
        keep = min(3, len(utxos)) if len(utxos) > 0 else 0
        keep = max(1, keep)  # nunca zero
    else:
        keep = max(1, int(len(utxos) * (1 - ratio)))

    pruned_utxos_global = utxos[:keep]

    # === Aviso Nuclear (porque sim) ===
    nuclear_warning = ""
    if strategy == NUCLEAR_OPTION:
        nuclear_warning = '<br><span style="color:#ff0066; font-weight:bold; font-size:18px;">MODO NUCLEAR ATIVADO<br>Só os fortes sobrevivem.</span>'

    return (
        f"""
        <div style="text-align:center; padding:20px;">
            <b style="font-size:22px; color:#f7931a;">Análise concluída</b><br><br>
            Encontrados <b>{len(utxos):,}</b> UTXOs • Mantendo <b>{keep}</b> maiores<br>
            <b style="color:#f7931a;">Estratégia:</b> <b>{strategy_name}</b> • Formato: <b>{detected}</b><br>
            {nuclear_warning}
            <br><br>
            Clique em <b>Gerar Transação</b> para continuar
        </div>
        """,
        gr.update(visible=True),
        gr.update(visible=True)
    )
# ==============================
# UPDATED build_real_tx — PSBT ONLY
# ==============================
def build_real_tx(user_input, strategy, threshold, dest_addr, dao_percent, future_multiplier):
    global pruned_utxos_global, input_vb_global, output_vb_global

    if not pruned_utxos_global:
        return "Primeiro faça a análise", gr.update(visible=False), gr.update(visible=False)

    sample_addr = pruned_utxos_global[0].get('address') or user_input.strip()
    _, info = address_to_script_pubkey(sample_addr)
    detected = info['type']

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)

    # === PEGA TAXA ATUAL DO MEMPOOL ===
    try:
        fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10).json()["fastestFee"]
    except:
        try:
            fee_rate = requests.get("https://mempool.space/api/v1/fees/recommended").json()["fastestFee"]
        except:
            fee_rate = 10

    # === TAXA FUTURA (apocalipse mode) ===
    future_rate = int(fee_rate * future_multiplier)
    future_rate = max(future_rate, fee_rate + 5)

    # === Estimativa inicial de vsize (1 saída) ===
    outputs = 1
    base_weight = 160
    input_weight = inputs * input_vb_global * 4
    output_weight = outputs * output_vb_global * 4
    witness_overhead = 2 if detected in ("SegWit", "Taproot") else 0
    total_weight = base_weight + input_weight + output_weight + witness_overhead
    vsize = (total_weight + 3) // 4

    # === TAXA DO MINERADOR (primeira passada) ===
    miner_fee = int(vsize * fee_rate * 1.06) + 1
    miner_fee = max(miner_fee, vsize * 12)
    miner_fee = min(miner_fee, total // 5)

    # === ESTIMA ECONOMIA FUTURA ===
    future_cost = int((input_vb_global * inputs + output_vb_global * 2 + 10) * future_rate)
    savings = future_cost - miner_fee

    # === AGRADECIMENTO (0–500 bps, capado em 25% da economia) ===
    dao_cut = 0
    if dao_percent > 0 and savings > 3000:
        raw_cut = int(savings * dao_percent / 10_000)
        dao_cut = max(546, raw_cut)
        dao_cut = min(dao_cut, savings // 4)

    # === Quantidade final de saídas (agora sabemos se tem agradecimento) ===
    outputs = 1 + (1 if dao_cut > 0 else 0)

    # === RECALCULA vsize com número correto de saídas ===
    output_weight = outputs * output_vb_global * 4
    total_weight = base_weight + input_weight + output_weight + witness_overhead
    vsize = (total_weight + 3) // 4

    # === TAXA FINAL DO MINERADOR ===
    miner_fee = int(vsize * fee_rate * 1.06) + 1
    miner_fee = max(miner_fee, vsize * 12)
    miner_fee = min(miner_fee, total // 5)

    # === VALOR FINAL DO USUÁRIO ===
    user_gets = total - miner_fee - dao_cut
    if user_gets < 546:
        return "Não sobra saldo suficiente após taxas", gr.update(visible=False), gr.update(visible=False)

    # === DESTINO ===
    dest = (dest_addr or user_input).strip()
    dest_script, _ = address_to_script_pubkey(dest)
    if len(dest_script) < 20:
        return "Endereço de destino inválido", gr.update(visible=False), gr.update(visible=False)

    # === MONTA A TRANSAÇÃO ===
    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid'])[::-1], u['vout']))
    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut > 0:
        dao_script, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    # === GERA PSBT & QR ===
    psbt_b64 = make_psbt(tx)
    qr = make_qr(psbt_b64)
    thank = "Nenhum agradecimento" if dao_cut == 0 else f"Agradecimento enviado: {format_btc(dao_cut)}"

    # === BOTÃO COPIAR PSBT (com toast verde foda) ===
    copy_button = f"""
    <button onclick="navigator.clipboard.writeText(`{psbt_b64}`).then(()=>{{
        const t=document.createElement('div');
        t.textContent='PSBT copiado!';
        t.style.cssText=`position:fixed;bottom:100px;left:50%;transform:translateX(-50%);
                         z-index:10000;background:#00ff9d;color:#000;padding:16px 36px;
                         border-radius:50px;font-weight:bold;font-size:18px;
                         box-shadow:0 12px 40px rgba(0,0,0,0.6);animation:pop 2s forwards;`;
        document.body.appendChild(t);
        setTimeout(()=>t.remove(),2000);
    }})" 
    style="margin:30px auto;display:block;padding:18px 42px;font-size:1.19rem;
           font-weight:800;border-radius:16px;border:none;background:#f7931a;
           color:white;cursor:pointer;box-shadow:0 8px 30px rgba(247,147,26,0.5);
           transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 14px 40px rgba(247,147,26,0.7)'"
    onmouseout="this.style.transform='';this.style.boxShadow='0 8px 30px rgba(247,147,26,0.5)'">
        Copiar PSBT para área de transferência
    </button>
    """

    html = f"""
    <div style="text-align:center; padding:20px;">
    <h3 style="color:#f7931a;">Transação pronta — PSBT gerado</h3>
    <p><b>{inputs}</b> entradas → {format_btc(total)} • Taxa minerador: <b>{format_btc(miner_fee)}</b> @ <b>{fee_rate}</b> sat/vB<br>
       <span style="font-size:18px; color:#f7931a; font-weight:700;">{thank}</span></p>
    
    <b style="font-size:32px; color:black; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d;">
        Você recebe: {format_btc(user_gets)}
    </b>
    
    <div style="margin: 30px 0; padding: 18px; background: rgba(247,147,26,0.12); border-radius: 14px; border: 1px solid #f7931a;">
         Economia futura ≈ <b style="font-size:24px; color:#00ff9d; text-shadow: 0 0 12px black, 0 0 24px black;">
            {format_btc(savings)}
        </b> (@ <b>{future_rate}</b> sat/vB)
    </div>

    <div style="margin:40px 0;" class="qr-center">
        <img src="{qr}">
    </div>

    {copy_button}

    <p><small>Escanear com sua wallet • ou copiar e colar</small></p>

    <details style="margin-top: 32px;">
        <summary style="cursor: pointer; color: #f7931a; font-weight: bold; font-size: 18px; text-align:center;">
            Ver PSBT bruto (clique para expandir)
        </summary>
        <pre style="background:#000; color:#0f0; padding:18px; border-radius:12px; overflow-x:auto; margin-top:12px; font-size:12px;">
{psbt_b64}
        </pre>
    </details>
    </div>
    """

    return html, gr.update(visible=False), gr.update(visible=False)

# ==============================
# Gradio UI — Final & Perfect
# ==============================
with gr.Blocks(
    title="Ωmega Pruner v10.0 — MODO NUCLEAR: Podar UTXOs para sempre",
) as demo:
    # ——— BULLETPROOF OG TAGS — FORCES THUMBNAIL + DESCRIPTION EVERYWHERE ———
    gr.HTML("""
    <head>
        <meta property="og:title" content="Ωmega Pruner v10.0 — MODO NUCLEAR: Podar UTXOs para sempre">
        <meta property="og:description" content="O último consolidator . . . MODO NUCLEAR. Para cada Bitcoiner. Do primeiro ao último.">
        <meta property="og:image" content="https://raw.githubusercontent.com/babyblueviper1/Viper-Stack-Omega/main/Omega_v10/omega_thumbnail.png">
        <meta property="og:image:width" content="1200">
        <meta property="og:image:height" content="630">
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://omega-v10-pt.onrender.com">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Ωmega Pruner v10.0 — MODO NUCLEAR">
        <meta name="twitter:description" content="MODO NUCLEAR. Para cada Bitcoiner. Do primeiro ao último.">
        <title>Ωmega Pruner v10.0 — MODO NUCLEAR</title>
    </head>
    """, visible=False)
    
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px 0 10px;">
            <h1 style="font-size: 3.2rem; margin: 0; background: linear-gradient(135deg, #f7931a, #ff9900); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(247,147,26,0.4);">
                Ωmega Pruner v10
            </h1>
        </div>
        """,
        elem_id="omega-title"
    )
    
    gr.HTML("""
    <div style="text-align:center; margin:0 0 40px; font-size:1rem; color:#ccc; text-shadow: 0 0 8px rgba(247,147,26,0.4);">
        <strong>Idioma:</strong> 
        <a href="https://omega-v10.onrender.com/" style="color:#f7931a; margin:0 15px; text-decoration:none; font-weight:600;">English</a> • 
        <a href="https://omega-v10-es.onrender.com/" style="color:#f7931a; margin:0 15px; text-decoration:none; font-weight:600;">Español</a> • 
        <span style="color:#f7931a; margin:0 15px; font-weight:700;">Português</span> • 
        <span style="color:#666; margin:0 10px;">Français (bientôt)</span> • 
        <span style="color:#666; margin:0 10px;">Deutsch (bald)</span>
    </div>
    """)
    
    gr.HTML(
        """
        <div id="omega-bg" style="
            position: fixed !important;
            inset: 0 !important;
            top: 0 !important; left: 0 !important;
            width: 100vw !important; height: 100vh !important;
            pointer-events: none !important;
            z-index: -1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            background: transparent;
        ">
            <span class="omega-symbol" style="
                font-size: 100vh !important;
                font-weight: 900 !important;
                background: linear-gradient(135deg, rgba(247,147,26,0.28), rgba(247,147,26,0.15)) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                color: transparent !important;
                text-shadow: 0 0 220px rgba(247,147,26,0.72) !important;
                animation: omega-breath 28s infinite ease-in-out !important;
                user-select: none !important;
                line-height: 1 !important;
                opacity: 0.96 !important;
            ">Ω</span>
        </div>

        <style>
        @keyframes omega-breath {
            0%, 100% { opacity: 0.76; transform: scale(0.95) rotate(0deg);   }
            50%      { opacity: 1.0;  transform: scale(1.05) rotate(180deg); }
        }
        .gradio-container { 
            position: relative !important; 
            z-index: 0 !important; 
            background: transparent !important;
            overflow-y: auto !important;
        }
        body { overflow-y: auto !important; }
        #omega-bg { 
            isolation: isolate !important; 
            will-change: transform, opacity !important; 
        }
        .omega-symbol { 
            animation-play-state: running !important; 
        }
        </style>

        <script>
        // A força sagrada do reflow — faz aparecer 100% das vezes
        window.addEventListener('load', () => {
            const omega = document.getElementById('omega-bg');
            if (omega) {
                omega.style.display = 'none';
                setTimeout(() => { omega.style.display = 'flex'; }, 120);
            }
        });
        </script>
        """,
        elem_id="omega-bg-container-fixed"
    )
      
    # ====================== LAYOUT STARTS HERE ======================
    with gr.Row():
    with gr.Column(scale=4):
        user_input = gr.Textbox(
            label="Carteira ou xpub",
            placeholder="bc1q… ou xpub…",
            lines=2,
            autofocus=True
        )
    with gr.Column(scale=3):
        prune_choice = gr.Dropdown(
            choices=[
                "Privacidade Primeiro (30% podado)",
                "Recomendado (40% podado)",
                "Mais Economia (50% podado)",
                "NUCLEAR PRUNE (90% sacrificado — só pra quem tem culhão)",
            ],
            value="Recomendado (40% podado)",
            label="Estratégia",
            info="Quantos UTXOs pequenos vamos queimar pra economizar taxas pra sempre"
        )

with gr.Row(equal_height=False):
    with gr.Column(scale=1, min_width=300):
        dust_threshold = gr.Slider(0, 3000, value=546, step=1,
            label="Limite de pó (sats)",
            info="Ignorar UTXOs menores que esse valor")
    with gr.Column(scale=1, min_width=300):
        dao_percent = gr.Slider(0, 500, value=50, step=10,
            label="Agradecimento (bps)",
            info="0–500 bps do seu futuro ganho (máximo 25% por segurança)")
        live_thankyou = gr.Markdown(
            "<div style='text-align:right;margin-top:8px;font-size:20px;color:#f7931a;font-weight:bold;'>"
            "→ 0,50% do seu futuro ganho"
            "</div>"
        )
    with gr.Column(scale=1, min_width=300):
        future_multiplier = gr.Slider(3, 20, value=6, step=1,
            label="Teste de estresse futuro",
            info="6× = pico histórico 2017–2024 • 15× = próximo bull run • 20× = apocalipse total"
        )

# Atualiza % do agradecimento em tempo real
def update_thankyou_label(bps):
    pct = bps / 100
    return f"<div style='text-align:right;margin-top:8px;font-size:20px;color:#f7931a;font-weight:bold;'>→ {pct:.2f}% do seu futuro ganho</div>"
dao_percent.change(update_thankyou_label, dao_percent, live_thankyou)

with gr.Row():
    dest_addr = gr.Textbox(
        label="Destino (opcional)",
        placeholder="Deixe vazio → mesma carteira",
        lines=1
    )

with gr.Row():
    submit_btn = gr.Button("1. Analisar UTXOs", variant="secondary", size="lg")

output_log = gr.HTML()

with gr.Row(visible=False) as generate_row:
    generate_btn = gr.Button(
        "2. Gerar Transação",
        visible=False,
        variant="primary",
        size="lg",
        elem_classes="full-width"
    )

with gr.Row():
    start_over_btn = gr.Button(
        "Reiniciar — Apagar tudo",
        variant="secondary",
        size="lg",
        elem_classes="full-width"
    )
    # ==================================================================
    # Events
    # ==================================================================
    submit_btn.click(
    analysis_pass,
    [user_input, prune_choice, dust_threshold, dest_addr, dao_percent, future_multiplier],
    [output_log, generate_btn, generate_row]
)

generate_btn.click(
    fn=build_real_tx,
    inputs=[user_input, prune_choice, dust_threshold, dest_addr, dao_percent, future_multiplier],
    outputs=[output_log, generate_btn, generate_row]
)

start_over_btn.click(
    lambda: (
        "", 
        "Recomendado (40% podado)",  # ← português agora
        546, 
        "", 
        50,
        "",  
        gr.update(visible=False), 
        gr.update(visible=False)
    ),
    outputs=[
        user_input, 
        prune_choice, 
        dust_threshold, 
        dest_addr, 
        dao_percent, 
        output_log, 
        generate_btn, 
        generate_row
    ]
)

# ——— ESCÂNER QR + TOAST EM PORTUGUÊS ———
gr.HTML("""
<!-- Botão Flutuante BTC Scanner -->
<label class="qr-fab btc" title="Escanear carteira / xpub">₿</label>
<input type="file" accept="image/*" capture="environment" id="qr-scanner-btc" style="display:none">

<script src="https://unpkg.com/@zxing/library@0.21.0/dist/index.min.js"></script>
<script>
function showToast(msg, err = false) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `position:fixed !important; bottom:100px !important; left:50% !important;
        transform:translateX(-50%) !important; z-index:10000 !important;
        background:${err?'#300':'rgba(0,0,0,0.92)'} !important;
        color:${err?'#ff3366':'#00ff9d'} !important;
        padding:16px 36px !important; border-radius:50px !important;
        font-weight:bold !important; font-size:17px !important;
        border:3px solid ${err?'#ff3366':'#00ff9d'} !important;
        box-shadow:0 12px 40px rgba(0,0,0,0.7) !important;
        backdrop-filter:blur(12px) !important;
        animation:pop 2.4s forwards !important;`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2400);
}
if (!document.getElementById('toast-style')) {
    const s = document.createElement('style');
    s.id = 'toast-style';
    s.textContent = `@keyframes pop{
        0%{transform:translateX(-50%) translateY(30px);opacity:0}
        12%,88%{transform:translateX(-50%) translateY(0);opacity:1}
        100%{transform:translateX(-50%) translateY(-30px);opacity:0}
    }`;
    document.head.appendChild(s);
}

document.querySelector('.qr-fab.btc')?.addEventListener('click', () => 
    document.getElementById('qr-scanner-btc').click()
);

document.getElementById('qr-scanner-btc').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
            const res = await ZXing.readBarcodeFromCanvas(canvas);
            const txt = res.text.trim().split('?')[0].replace(/^bitcoin:/i, '');
            if (/^(bc1|[13]|xpub|ypub|zpub|tpub)/i.test(txt)) {
                const box = document.querySelector('textarea[placeholder*="bc1q"], textarea[placeholder*="xpub"]') || 
                           document.querySelector('textarea');
                if (box) {
                    box.value = txt;
                    box.dispatchEvent(new Event('input', {bubbles:true}));
                    box.dispatchEvent(new Event('change', {bubbles:true}));
                }
                showToast("Escaneado com sucesso!");
            } else showToast("Não é carteira/xpub BTC", true);
        } catch { showToast("QR não detectado", true); }
    };
    img.src = URL.createObjectURL(file);
};
</script>
""")

# ——— FOOTER EM PORTUGUÊS ———
gr.HTML(
    """
    <div style="
            margin: 30px auto 6px auto !important; 
            padding: 12px 0 8px 0 !important; 
            text-align: center; 
            font-size: 0.92rem; 
            color: #888; 
            opacity: 0.94;
            max-width: 640px;
        ">
            <strong style="color:#f7931a; font-size:1.02rem;">Ωmega Pruner v10.0 — MODO NUCLEAR</strong><br>
            <a href="https://github.com/babyblueviper1/Viper-Stack-Omega/tree/main/Omega_v10" 
               target="_blank" rel="noopener" 
               style="color: #f7931a; text-decoration: none; font-weight:600;">
                GitHub • Código Aberto • Apache 2.0
            </a>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <a href="#verified-prunes" style="color:#f7931a; text-decoration:none; font-weight:600;">
                Prunes NUCLEARES verificados
            </a><br><br>
            
            <span style="font-size:0.92rem; color:#ff9900; font-weight:600;">
            Licença+ vitalícia — 0.042 BTC (só os primeiros 21) → 
            <a href="https://www.babyblueviper.com/p/mega-pruner-lifetime-license-0042" 
               style="color:#ff9900; text-decoration:underline;">detalhes</a>
            </span><br><br>

            <span style="color:#868686; font-size:0.85rem; text-shadow: 0 0 8px rgba(247,147,26,0.4);">Podar hoje. Vencer pra sempre. • Para cada Bitcoiner — do primeiro ao último. • Ω</span>
    </div>
    """,
    elem_id="omega-footer"
)

# ——— SEÇÃO VERIFIED PRUNES EM PORTUGUÊS ———
gr.HTML(
    """
    <div id="verified-prunes" style="margin:80px auto 40px; max-width:900px; padding:0 20px;">
        <h1 style="text-align:center; color:#f7931a; font-size:2.5rem; margin-bottom:20px;">
            Prunes NUCLEARES verificados
        </h1>
        <p style="text-align:center; color:#868686; font-size:1.1rem; margin-bottom:60px; text-shadow: 0 0 8px rgba(247,147,26,0.4);">
            O muro começa vazio.<br>
            Cada prune verificado fica gravado na blockchain pra sempre via TXID.<br>
            Os primeiros serão lembrados como lendas.
        </p>

        <div style="text-align:center; padding:50px 20px; background:#111; border:2px dashed #f7931a; border-radius:16px;">
            <p style="color:#f7931a; font-size:1.5rem; margin:0;">Ainda não há prunes verificados.</p>
            <p style="color:#aaa; margin:20px 0 0; font-size:1rem;">
                Seja o primeiro. Execute NUCLEAR. Envie seu TXID.
            </p>
        </div>

        <p style="text-align:center; color:#f7931a; margin-top:60px; font-size:1rem;">
            Responda no X com seu TXID → seu prune entra aqui pra sempre.<br><br>
            <a href="https://twitter.com/intent/tweet?text=Acabei%20de%20executar%20um%20prune%20NUCLEAR%20com%20%40babyblueviper1%20%E2%98%A2%EF%B8%8F%20TXID%3A" 
               target="_blank" 
               style="color:#ff9900; text-decoration:underline; font-weight:bold;">
                → Tuitar seu prune
            </a>
        </p>
    </div>
    """
)
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

if __name__ == "__main__":
    import os
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    demo.queue(default_concurrency_limit=None, max_size=40)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
        debug=False,
        max_threads=40,
        show_error=True,
        quiet=True,  
        allowed_paths=["./"],
        ssl_verify=False,
        css=css
    )
