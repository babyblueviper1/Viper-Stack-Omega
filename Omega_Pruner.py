# Omega_Pruner.py
import gradio as gr
import requests, time, base64, io, qrcode, json, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import warnings, logging
from functools import partial
import threading
import hashlib
import tempfile
from datetime import datetime

# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# =========================
# Config / Constants
# =========================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"
MEMPOOL_API = "https://mempool.space/api"
BLOCKSTREAM_API = "https://blockstream.info/api/address/{addr}/utxo"
BITCOINER_API = "https://bitcoiner.live/api/address/{addr}/utxo"
MIN_KEEP_UTXOS = 1  # minimum UTXOs to keep regardless of strategy
HEALTH_PRIORITY = {
    "DUST": 0,
    "HEAVY": 1,
    "CAREFUL": 2,
    "MEDIUM": 3,
    "MANUAL": 3,     # Neutral position — doesn't push to prune or keep aggressively
    "OPTIMAL": 4,
}
CHECKBOX_COL = 0
SOURCE_COL   = 1
TXID_COL     = 2
VOUT_COL     = 3
VALUE_COL    = 4
ADDRESS_COL  = 5
WEIGHT_COL   = 6
TYPE_COL     = 7
HEALTH_COL   = 8

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

# =========================
# Global Requests Session (production-grade)
# =========================
session = requests.Session()
session.headers.update({
    "User-Agent": "OmegaPruner-v11 (+https://github.com/babyblueviper1/Viper-Stack-Omega)",
    "Accept": "application/json",
    "Connection": "keep-alive",
})


# ===========================
# Selection resolver
# ===========================
def _resolve_selected(df_rows: List[list], enriched_state: List[dict]) -> List[dict]:
    """Robust, defensive resolution of selected UTXOs from UI state."""
    # Build map only from valid UTXOs
    utxo_map = {
        (u["txid"], u["vout"]): u
        for u in enriched_state
        if isinstance(u, dict) and "txid" in u and "vout" in u
    }

    selected = []
    for row in df_rows or []:
        if not row:
            continue
        try:
            if not row[CHECKBOX_COL]:
                continue
            # Be forgiving: convert to str/int if possible
            txid = str(row[TXID_COL])
            vout = int(row[VOUT_COL])
            utxo = utxo_map.get((txid, vout))
            if utxo:
                selected.append(utxo)
        except (IndexError, ValueError, TypeError, KeyError):
            # Silently skip malformed rows — never crash
            continue

    return selected

def _selection_snapshot(selected_utxos: List[dict]) -> dict:
    """
    Deterministic, audit-friendly snapshot of the current UTXO selection.
    This is the canonical representation of user intent.
    """
    return {
        "fingerprint": _selection_fingerprint(selected_utxos),
        "count": len(selected_utxos),
        "total_value": sum(u["value"] for u in selected_utxos),
        "utxos": [
            {
                "txid": u["txid"],
                "vout": u["vout"],
                "value": u["value"],
                "address": u.get("address"),
                "script_type": u.get("script_type"),
                "health": u.get("health"),
                "source": u.get("source"),
            }
            for u in sorted(
                selected_utxos,
                key=lambda u: (u["txid"], u["vout"])
            )
        ]
    }

def _selection_fingerprint(selected_utxos: List[dict]) -> str:
    """Deterministic short hash of selected inputs (sorted txid:vout)."""
    if not selected_utxos:
        return "none"
    keys = sorted((u["txid"], u["vout"]) for u in selected_utxos)
    data = ":".join(f"{txid}:{vout}" for txid, vout in keys).encode()
    return hashlib.sha256(data).hexdigest()[:16]


# =========================
# Utility Functions
# =========================

def safe_get(url: str) -> Optional[requests.Response]:
    """Robust GET with retries and exponential backoff."""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:  # Rate-limited
                sleep_time = 1.5 ** attempt
                log.warning(f"Rate limited on {url}, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            log.warning(f"Request failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(0.2 * (2 ** attempt))
    return None

def get_live_fees() -> Optional[Dict[str, int]]:
    """Fetch recommended fees from mempool.space with 30-second caching."""
    if hasattr(get_live_fees, "cache") and time.time() - get_live_fees.cache_time < 30:
        return get_live_fees.cache

    try:
        r = session.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        if r.status_code == 200:
            data = r.json()
            fees = {
                "fastest": data["fastestFee"],
                "half_hour": data["halfHourFee"],
                "hour": data["hourFee"],
                "economy": data["economyFee"],
                "minimum": data["minimumFee"]
            }
            get_live_fees.cache = fees
            get_live_fees.cache_time = time.time()
            return fees
    except requests.exceptions.RequestException as e:
        log.warning(f"Failed to fetch live fees: {e}")
    return None




def sats_to_btc_str(sats: int) -> str:
    btc = sats / 100_000_000
    if btc >= 1:
        return f"{btc:,.8f}".rstrip("0").rstrip(".") + " BTC"
    return f"{int(sats):,} sats"

def calculate_privacy_score(selected_utxos: List[dict], total_utxos: int) -> int:
    if not selected_utxos or len(selected_utxos) <= 1:
        return 95  # Single input or none = minimal linkage
    
    input_count = len(selected_utxos)
    distinct_addrs = len(set(u["address"] for u in selected_utxos))
    total_value_btc = sum(u["value"] for u in selected_utxos) / 100_000_000
    
    score = 100
    
    # CIOH linkage penalty (core privacy cost)
    if input_count >= 50: score -= 60
    elif input_count >= 20: score -= 45
    elif input_count >= 10: score -= 30
    elif input_count >= 5: score -= 15
    elif input_count > 1: score -= 8
    
    # Address merging penalty
    if distinct_addrs >= 15: score -= 40
    elif distinct_addrs >= 8: score -= 25
    elif distinct_addrs >= 4: score -= 12
    elif distinct_addrs > 1: score -= 6
    
    # Wealth reveal
    if total_value_btc >= 10: score -= 20
    elif total_value_btc >= 1: score -= 8
    elif total_value_btc >= 0.1: score -= 3
    
    # Tiny bonus for script diversity (confuses clustering)
    script_types = set(u.get("script_type", "Unknown") for u in selected_utxos)
    if len(script_types) >= 3: score += 6
    elif len(script_types) == 2: score += 3
    
    return max(5, min(100, score))  # Never 0 — some privacy always remains


def get_cioh_warning(input_count: int, distinct_addrs: int, privacy_score: int) -> str:
    if input_count <= 1:
        return ""

    if privacy_score <= 30:
        return (
            "<div style='margin-top:14px;padding:14px;background:#440000;border:3px solid #ff3366;border-radius:12px;"
            "box-shadow:0 0 40px rgba(255,51,102,0.9);font-size:1.1rem;'>"
            "<strong style='color:#ff3366;font-size:1.3rem;'>EXTREME CIOH LINKAGE</strong><br>"
            "<strong style='color:#ff6688;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            "This consolidation strongly proves common ownership of many inputs/addresses.<br>"
            "Privacy significantly reduced. Consider CoinJoin, PayJoin, or silent payments afterward."
            "</div>"
        )
    elif privacy_score <= 50:
        return (
            "<div style='margin-top:12px;padding:12px;background:#331100;border:2px solid #ff8800;border-radius:10px;'>"
            "<strong style='color:#ff9900;'>High CIOH Risk</strong><br>"
            "<strong style='color:#ffaa44;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            f"Merging {input_count} inputs from {distinct_addrs} address(es) → analysts will cluster them as yours.<br>"
            "Good fee savings, but real privacy trade-off."
            "</div>"
        )
    elif privacy_score <= 70:
        return (
            "<div style='margin-top:10px;padding:10px;background:#113300;border:1px solid #00ff9d;border-radius:8px;color:#aaffaa;'>"
            "<strong style='color:#00ff9d;'>Moderate CIOH</strong><br>"
            "<strong style='color:#66ffaa;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            "Some linkage created, but not extreme. Acceptable during low-fee periods."
            "</div>"
        )
    else:
        return (
            "<div style='margin-top:8px;color:#aaffaa;font-size:0.9rem;'>"
            "Low CIOH impact <strong style='color:#00ffdd;'>(Common Input Ownership Heuristic)</strong> — minimal new linkage."
            "</div>"
        )

# =========================
# Helper Functions for Bitcoin Addresses
# =========================

def base58_decode(s: str) -> bytes:
    """Decode base58 encoded string."""
    n = 0
    for c in s:
        n = n * 58 + BASE58_ALPHABET.index(c)
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + n.to_bytes((n.bit_length() + 7) // 8, 'big')

def bech32_polymod(values) -> int:
    """Checksum calculation for Bech32 addresses."""
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_verify_checksum(hrp: str, data: List[int]) -> bool:
    return bech32_polymod([ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp] + data) == 1

def bech32m_verify_checksum(hrp: str, data: List[int]) -> bool:
    return bech32_polymod([ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp] + data) == 0x2bc830a3

def convertbits(data: List[int], frombits: int, tobits: int, pad: bool = True) -> List[int]:
    """Convert bit sizes for address formats."""
    acc = bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append(acc >> bits & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or (acc << (tobits - bits)) & maxv:
        raise ValueError("Invalid padding")
    return ret

def address_to_script_pubkey(addr: str) -> Tuple[bytes, Dict[str, Any]]:
    """Convert Bitcoin address to script pubkey and associated metadata."""
    addr = (addr or "").strip().lower()
    if not addr:
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'unknown'}

    # P2PKH (Legacy)
    if addr.startswith('1'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x00:
                return b'\x76\xa9\x14' + dec[1:21] + b'\x88\xac', {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
        except:
            pass

    # P2SH
    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:
                return b'\xa9\x14' + dec[1:21] + b'\x87', {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
        except:
            pass

    # P2WPKH / P2WSH (Bech32)
    if addr.startswith('bc1q'):
        data = [CHARSET.find(c) for c in addr[4:] if c in CHARSET]
        if data and data[0] == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 20:
                return b'\x00\x14' + bytes(prog), {'input_vb': 68, 'output_vb': 31, 'type': 'P2WPKH'}
            if prog and len(prog) == 32:
                return b'\x00\x20' + bytes(prog), {'input_vb': 69, 'output_vb': 43, 'type': 'P2WSH'}

    # Taproot (Bech32m)
    if addr.startswith('bc1p'):
        data = [CHARSET.find(c) for c in addr[4:] if c in CHARSET]
        if data and len(data) > 1 and data[0] == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}

    # Fallback
    return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'fallback'}

# =========================
# Transaction Economics (Single Source of Truth)
# =========================

@dataclass(frozen=True)
class TxEconomics:
    total_in: int          # Sum of selected UTXO values
    vsize: int             # Final virtual size in vbytes
    fee: int               # Final fee (including absorbed dust/DAO)
    remaining: int         # Amount left after fee (before DAO/change split)
    dao_amt: int           # Final DAO donation (0 if dust)
    change_amt: int        # Final change output (0 if dust)

def estimate_tx_economics(
    selected_utxos: List[dict],
    fee_rate: int,
    dao_percent: float,
) -> TxEconomics:
    """Estimate transaction economics based on selected UTXOs and fee rate."""
    if not selected_utxos:
        raise ValueError("No UTXOs selected")

    total_in = sum(u["value"] for u in selected_utxos)
    input_weight = sum(u["input_weight"] for u in selected_utxos)
    input_count = len(selected_utxos)

    # Base SegWit vsize calculation
    base_vsize = (input_weight + 43 * 4 + 31 * 4 + 10 * 4 + input_count) // 4 + 10
    conservative_vsize = (input_weight + 150 + input_count * 60) // 4

    vsize = max(base_vsize, conservative_vsize)
    fee = max(600, int(vsize * fee_rate))  # Minimum effective 1 sat/vB
    remaining = total_in - fee

    # DAO contribution (dust-proof)
    dao_raw = int(remaining * (dao_percent / 100.0)) if dao_percent > 0 else 0
    dao_amt = dao_raw if dao_raw >= 546 else 0
    if 0 < dao_raw < 546:
        fee += dao_raw

    remaining -= dao_amt

    # Change output (dust-safe)
    change_amt = remaining if remaining >= 546 else 0
    if 0 < remaining < 546:
        fee += remaining
        remaining = 0

    return TxEconomics(
        total_in=total_in,
        vsize=vsize,
        fee=fee,
        remaining=remaining,
        dao_amt=dao_amt,
        change_amt=change_amt,
    )



def encode_varint(i: int) -> bytes:
    if i < 0xfd: return bytes([i])
    if i < 0x10000: return b'\xfd' + i.to_bytes(2, 'little')
    if i < 0x100000000: return b'\xfe' + i.to_bytes(4, 'little')
    return b'\xff' + i.to_bytes(8, 'little')



@dataclass
class TxIn:
    prev_tx: bytes
    prev_index: int
    script_sig: bytes = b''
    sequence: int = 0xfffffffd   # Enable RBF

    def serialize(self) -> bytes:
        return (
            self.prev_tx[::-1] +
            self.prev_index.to_bytes(4, 'little') +
            encode_varint(len(self.script_sig)) + self.script_sig +
            self.sequence.to_bytes(4, 'little')
        )

@dataclass
class TxOut:
    amount: int
    script_pubkey: bytes

    def serialize(self) -> bytes:
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

def create_psbt(tx_hex: str) -> str:
    tx = bytes.fromhex(tx_hex)
    psbt = (
        b'psbt\xff' +
        b'\x00' +                                    # global unsigned tx
        encode_varint(len(tx)) + tx +
        b'\x00' +                                    # separator
        b'\xff'                                      # end of global
    )
    return base64.b64encode(psbt).decode()



# =========================
# UTXO Fetching Functions
# =========================

def get_utxos(addr: str, dust: int = 546) -> List[dict]:
    """Fetch UTXOs for a given address from multiple public APIs."""
    addr = addr.strip()
    if not addr:
        return []

    apis = [
        f"{MEMPOOL_API}/address/{addr}/utxo",
        BLOCKSTREAM_API.format(addr=addr),
        BITCOINER_API.format(addr=addr),
    ]

    for url in apis:
        r = safe_get(url)
        if not r:
            continue
        try:
            data = r.json()
            if not data:
                continue

            utxos = []
            iter_u = data if isinstance(data, list) else data.get("utxos", [])
            for u in iter_u:
                val = int(u.get("value", 0))
                if val > dust:
                    utxos.append({
                        "txid": u["txid"],
                        "vout": u["vout"],
                        "value": val,
                        "address": addr,
                        "confirmed": (
                            u.get("status", {}).get("confirmed", True)
                            if isinstance(u.get("status"), dict) else True
                        )
                    })
            if utxos:
                return utxos
        except Exception as e:
            log.warning(f"Error parsing UTXOs from {url}: {e}")
        time.sleep(0.05)  # Be polite to APIs

    return []

# =========================
# xPub Scanning (Optional HD Wallet Support)
# =========================
try:
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC as HDWALLET_BTC
    HDWALLET_AVAILABLE = True
except Exception:  # pragma: no cover
    HDWALLET_AVAILABLE = False

def scan_xpub(xpub: str, dust: int = 546, gap_limit: int = 20) -> Tuple[List[dict], str]:
    """Scan an xpub/ypub/zpub for UTXOs across receive/change addresses."""
    if not HDWALLET_AVAILABLE:
        return [], "Install: pip install hdwallet"

    all_utxos = []
    try:
        hdw = HDWallet(symbol=HDWALLET_BTC).from_xpublic_key(xpub.strip())
        purpose = 84 if xpub.startswith(("zpub", "vpub")) else 49 if xpub.startswith(("ypub", "upub")) else 44

        for change in [0, 1]:
            empty = 0
            for i in range(200):
                path = f"m/{purpose}'/0'/{change}/{i}"
                hdw.from_path(path)
                addr = (
                    hdw.p2wpkh_address() if purpose == 84 else
                    hdw.p2sh_p2wpkh_address() if purpose == 49 else
                    hdw.p2pkh_address()
                )
                utxos = get_utxos(addr, dust)
                if utxos:
                    all_utxos.extend(utxos)
                    empty = 0
                else:
                    empty += 1
                    if empty >= gap_limit:
                        break
                time.sleep(0.02)
        all_utxos.sort(key=lambda x: x["value"], reverse=True)
        addresses_used = len(set(u['address'] for u in all_utxos))
        return all_utxos, f"Found {len(all_utxos)} UTXOs across {addresses_used} addresses"
    except Exception as e:
        return [], f"Error: {str(e)[:120]}"


def analyze(addr_input, strategy, dust_threshold, dest_addr, fee_rate_slider, dao_slider, future_fee_slider, offline_mode, manual_utxo_input):
    # === SAFE INPUT CLAMPING ===
    fee_rate = max(1, min(500, int(float(fee_rate_slider or 15))))
    future_fee_rate = max(1, min(1000, int(float(future_fee_slider or 60))))
    dao_percent = max(0.0, min(10.0, float(dao_slider or 0.5)))
    dust_threshold = max(0, min(10000, int(float(dust_threshold or 546))))

    all_enriched = []
    successful_sources = 0
    info = ""

    if offline_mode:
        if not manual_utxo_input.strip():
            return "<div style='color:#ff3366;'>No manual UTXOs provided</div>", [], [], "", gr.update(visible=False), gr.update(visible=False)

        for line in manual_utxo_input.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')
            if len(parts) < 3:
                continue
            txid, vout_str, value_str = parts[0], parts[1], parts[2]
            addr = parts[3].strip() if len(parts) >= 4 else "unknown (manual)"
            try:
                vout = int(vout_str)
                value = int(value_str)
                if value <= dust_threshold:
                    continue

                # Estimate weight (fallback if no address)
                spk, meta = address_to_script_pubkey(addr)
                input_wu = meta["input_vb"] * 4

                enriched = {
                    "txid": txid,
                    "vout": vout,
                    "value": value,
                    "address": addr,
                    "input_weight": input_wu,
                    "health": "MANUAL",           # Fixed: only once
                    "recommend": "REVIEW",        # Fixed: only once
                    "script_type": meta["type"] if addr != "unknown (manual)" else "Unknown",
                    "source": "Manual Offline",
                }
                all_enriched.append(enriched)
            except Exception:
                continue

        if not all_enriched:
            return "<div style='color:#ff3366;'>No valid UTXOs parsed</div>", [], [], "", gr.update(visible=False), gr.update(visible=False)

        info = f"Offline manual mode • No API calls • {len(all_enriched):,} UTXOs loaded"
        successful_sources = 1

    else:
        # Online batch/single mode
        entries = [e.strip() for e in addr_input.strip().splitlines() if e.strip()]
        if not entries:
            return "<div style='color:#ff3366;'>No addresses provided</div>", [], [], "", gr.update(visible=False), gr.update(visible=False)

        for entry in entries:
            is_xpub = entry[:4] in ("xpub", "ypub", "zpub", "tpub", "upub", "vpub")
            source_label = f"xpub ({entry[:12]}...{entry[-6:]})" if is_xpub else (entry if len(entry) <= 34 else f"{entry[:31]}...")

            if is_xpub:
                utxos_raw, _ = scan_xpub(entry, dust_threshold)
            else:
                utxos_raw = get_utxos(entry, dust_threshold)

            if not utxos_raw:
                continue

            successful_sources += 1

            for u in utxos_raw:
                _, meta = address_to_script_pubkey(u["address"])
                input_wu = meta["input_vb"] * 4
                value = u["value"]

                # Same health logic as above
                if input_wu <= 228:
                    script_type = "Taproot"
                    health, recommend = "OPTIMAL", "KEEP"
                elif input_wu <= 272:
                    script_type = "Native SegWit"
                    health, recommend = "OPTIMAL", "KEEP"
                elif input_wu <= 364:
                    script_type = "Nested SegWit"
                    health, recommend = "MEDIUM", "OPTIONAL"
                else:
                    script_type = "Legacy"
                    health, recommend = "HEAVY", "PRUNE"

                if value < 10_000:
                    health, recommend = "DUST", "PRUNE"
                if value > 100_000_000 and script_type in ("Nested SegWit", "Legacy"):
                    health, recommend = "CAREFUL", "OPTIONAL"

                enriched = {
                    **u,
                    "input_weight": input_wu,
                    "health": health,
                    "recommend": recommend,
                    "script_type": script_type,
                    "source": source_label,
                }
                all_enriched.append(enriched)

        if not all_enriched:
            return f"No UTXOs found across {len(entries)} source(s)", [], [], "", gr.update(visible=False), gr.update(visible=False)

        info = f"Batch mode • {successful_sources} sources • {len(all_enriched):,} UTXOs loaded" if successful_sources > 1 else f"Single source • {len(all_enriched):,} UTXOs"

    # ===============================
    # CANONICAL ORDERING & PRUNING LOGIC
    # ===============================
    # Sort all enriched by value (largest first) for determinism
    all_enriched.sort(key=lambda u: (u["value"], u["txid"], u["vout"]), reverse=True)

    # Pre-prune vsize estimate
    all_input_weight = sum(u["input_weight"] for u in all_enriched)
    pre_vsize = max(
        (all_input_weight + 43*4 + 31*4 + 10*4 + len(all_enriched)) // 4 + 10,
        (all_input_weight + 150 + len(all_enriched) * 60) // 4,
    )

    # Pruning strategy
    ratio = {
        "Privacy First — ~30% pruned (lowest CIOH risk)": 0.30,
        "Recommended — ~40% pruned (balanced savings & privacy)": 0.40,
        "More Savings — ~50% pruned (stronger fee reduction)": 0.50,
        "NUCLEAR PRUNE — ~90% pruned (maximum savings, highest CIOH)": 0.90,
    }.get(strategy, 0.40)
    keep_count = max(MIN_KEEP_UTXOS, int(len(all_enriched) * (1 - ratio)))

    # Sort by health: worst first (so worst appear at the top of the table)
    enriched_sorted = sorted(all_enriched, key=lambda u: HEALTH_PRIORITY[u["health"]])

    # Number of UTXOs to pre-select for pruning (the worst ones)
    prune_count = len(enriched_sorted) - keep_count

    # Build dataframe rows
    df_rows = []
    for idx, u in enumerate(enriched_sorted):
        # Pre-check the first prune_count rows (the worst health)
        is_prune = idx < prune_count
        health_html = f'<div class="health health-{u["health"].lower()}">{u["health"]}<br><small>{u["recommend"]}</small></div>'
        df_rows.append([
            is_prune,
            u.get("source", "Single"),
            u["txid"],
            u["vout"],
            u["value"],
            u["address"],
            u["input_weight"],
            u["script_type"],
            health_html,
        ])



    summary_html = generate_summary(df_rows, enriched_sorted, fee_rate, future_fee_rate, dao_percent)

    return "", gr.update(value=df_rows), enriched_sorted, summary_html, gr.update(visible=True), gr.update(visible=True)
    
def generate_summary(
    df_rows: List[list],
    enriched_state: List[dict],
    fee_rate: int = 15,
    future_fee_rate: int = 60,
    dao_percent: float = 0.5,
    current_strategy: str = "Recommended — ~40% pruned (balanced savings & privacy)",
) -> Tuple[str, gr.update, str]:

    total_utxos = len(enriched_state)

    # === CRITICAL GUARD: No UTXOs loaded yet (pre-ANALYZE state) ===
    if total_utxos == 0:
        return "", gr.update(visible=False), ""

    # === Resolve current selection ===
    selected_utxos = _resolve_selected(df_rows, enriched_state)
    pruned_count = len(selected_utxos)

    # === Privacy score ===
    privacy_score = (
        calculate_privacy_score(selected_utxos, total_utxos)
        if selected_utxos else 100
    )
    score_color = (
        "#0f0" if privacy_score >= 70
        else "#ff9900" if privacy_score >= 40
        else "#ff3366"
    )

    # === Banner — always built when UTXOs exist ===
    banner_html = f"""
    <div style="text-align:center;margin:30px 0;padding:20px;background:#000;
                border:3px solid #f7931a;border-radius:16px;
                box-shadow:0 0 60px rgba(247,147,26,0.6);">
      <div style="color:#0f0;font-size:2.4rem;font-weight:900;
                  letter-spacing:2px;text-shadow:0 0 30px #0f0;">
        READY TO PRUNE
      </div>
      <div style="color:#f7931a;font-size:1.8rem;font-weight:900;margin:16px 0;">
        {total_utxos:,} UTXOs • Strategy:
        <span style="color:#00ff9d;">{current_strategy}</span>
      </div>
      <div style="color:#fff;font-size:1.4rem;font-weight:700;">
        Will prune <span style="color:#ff6600;font-weight:800;">
        {pruned_count:,}</span> inputs
      </div>
      <div style="color:#fff;font-size:1.6rem;font-weight:800;margin:20px 0;">
        Privacy Score:
        <span style="color:{score_color};
                     text-shadow:0 0 20px {score_color};">
            {privacy_score}/100
        </span>
      </div>
    </div>
    """

    # === Button visibility — single source of truth ===
    button_visibility = gr.update(visible=bool(selected_utxos))

    # === Case: UTXOs loaded, but none selected yet ===
    if not selected_utxos:
        return (
            banner_html,
            button_visibility,  # False
            "<div style='text-align:center;margin:60px 0;color:#888;font-size:1.1rem;'>"
            "No UTXOs selected yet — check the boxes in the table to begin pruning"
            "</div>"
        )

    # === Economics calculation ===
    try:
        econ = estimate_tx_economics(selected_utxos, fee_rate, dao_percent)
    except ValueError:
        # Shouldn't happen with valid selection, but safety
        return banner_html, gr.update(visible=False), ""

    # === Invalid transaction (fee too high) ===
    if econ.remaining <= 0:
        invalid_html = (
            "<div style='text-align:center;margin:20px;padding:20px;background:#330000;"
            "border:2px solid #ff3366;border-radius:14px;box-shadow:0 0 40px rgba(255,51,102,0.6);'>"
            "<div style='color:#ff3366;font-size:1.25rem;font-weight:700;'>Transaction Invalid</div>"
            "<div style='color:#fff;margin-top:12px;line-height:1.7;'>"
            "Fee exceeds available balance after pruning.<br>"
            "<strong style='color:#ff9966;'>Reduce fee rate</strong> or "
            "<strong style='color:#ff9966;'>select more UTXOs</strong>."
            "</div></div>"
        )
        return banner_html, gr.update(visible=False), invalid_html

    # === Size & savings calculations ===
    input_weight = sum(u["input_weight"] for u in selected_utxos)
    all_input_weight = sum(u["input_weight"] for u in enriched_state)

    pre_vsize = max(
        (all_input_weight + 172 + total_utxos) // 4 + 10,  # more accurate base
        (all_input_weight + 150 + total_utxos * 60) // 4 + 10,
    )

    savings_pct = round(100 * (1 - econ.vsize / pre_vsize), 1) if pre_vsize > econ.vsize else 0
    savings_pct = max(0, min(100, savings_pct))

    savings_color = "#0f0" if savings_pct >= 70 else "#00ff9d" if savings_pct >= 50 else "#ff9900" if savings_pct >= 30 else "#ff3366"
    savings_label = "NUCLEAR" if savings_pct >= 70 else "EXCELLENT" if savings_pct >= 50 else "GOOD" if savings_pct >= 30 else "WEAK"

    # === Future savings ===
    sats_saved = max(0, econ.vsize * (future_fee_rate - fee_rate))
    savings_text = ""
    if sats_saved >= 100_000:
        savings_text = (
            f"<span style='color:#0f0;font-size:1.18rem;font-weight:800;text-shadow:0 0 20px #0f0;'>"
            f"+{sats_saved:,} sats saved</span> "
            f"<span style='color:#0f0;font-weight:800;letter-spacing:3px;text-transform:uppercase;"
            f"text-shadow:0 0 30px #0f0;'>NUCLEAR MOVE</span>"
        )
    elif sats_saved > 0:
        savings_text = f"<span style='color:#0f0;font-weight:800;'>+{sats_saved:,} sats saved</span>"

    # === Privacy warnings ===
    distinct_addrs = len({u["address"] for u in selected_utxos})
    cioh_warning = get_cioh_warning(len(selected_utxos), distinct_addrs, privacy_score)

    bad_ratio = len([u for u in selected_utxos if u.get("health") in ("DUST", "HEAVY")]) / len(selected_utxos)
    extra_warning = (
        "<div style='margin-top:12px;color:#ae2029;font-weight:900;'>"
        "CAUTION: Heavy consolidation — strong fee savings.<br>"
        "Consider CoinJoin afterward.</div>"
        if bad_ratio > 0.8 else
        "<div style='margin-top:12px;color:#fcf75e;'>"
        "High dusty/heavy ratio — good savings, privacy trade-off.</div>"
        if bad_ratio > 0.6 else ""
    )

    # === Final details block ===
    details_html = f"""
    <div style='text-align:center;margin:10px;padding:14px;background:#111;
                border:2px solid #f7931a;border-radius:14px;max-width:100%;
                font-size:1rem;line-height:2.1;'>
      <b style='color:#fff;'>Selected Inputs:</b> <span style='color:#0f0;font-weight:800;'>{len(selected_utxos):,}</span><br>
      <b style='color:#fff;'>Total Value Pruned:</b> <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(econ.total_in)}</span><br>
      <b style='color:#fff;'>Post-prune tx size:</b> 
      <span style='color:#0f0;font-weight:800;'>{econ.vsize:,} vB</span>
      <span style='color:{savings_color};font-weight:800;text-shadow:0 0 20px {savings_color};'>
        {' ' + savings_label} (-{savings_pct}%)
      </span><br>
      {f"<b style='color:#fff;'>Sats saved →</b> {savings_text}<br>" if savings_text else ""}
      <b style='color:#fff;'>Fee:</b> 
      <span style='color:#0f0;font-weight:800;'>{econ.fee:,} sats @ {fee_rate} s/vB</span><br>
      <b style='color:#fff;'>Privacy Score:</b> 
      <span style='color:{score_color};font-weight:800;font-size:1.6rem;text-shadow:0 0 20px {score_color};'>
        {privacy_score}/100
      </span>
      <div style='margin-top:12px;'>{cioh_warning}</div>
      <div style='margin-top:6px;'>{extra_warning}</div>
    </div>
    """

    # === Single return point ===
    return banner_html, button_visibility, details_html


def generate_summary_safe(df_rows, enriched_state, fee_rate, future_fee_rate, dao_percent, locked, current_strategy):
    if locked:
        return None, None, None
    return generate_summary(df_rows, enriched_state, fee_rate, future_fee_rate, dao_percent, current_strategy)


def generate_psbt(
    dest_addr: str,
    fee_rate: int,
    future_fee_rate: int,
    dao_percent: float,
    df_rows: list,
    enriched_state: list
):
    if not df_rows or not enriched_state:
        return "<div style='color:#ff3366; text-align:center; padding:30px;'>Run Analyze first.</div>", gr.update(), gr.update(), "",gr.update(visible=False), gr.update(visible=False), None, []   

    # === SAFELY EXTRACT SELECTED UTXOs ===
    selected_utxos = _resolve_selected(df_rows, enriched_state)
    snapshot = _selection_snapshot(selected_utxos)
    fingerprint = snapshot["fingerprint"]  # for display

    # === CREATE TEMP JSON FILE FOR DOWNLOAD ===
    json_str = json.dumps(snapshot, indent=2, ensure_ascii=False)
    date_str = datetime.now().strftime("%Y%m%d")

    prefix = f"omega_selection_{date_str}_{fingerprint[:8]}_" if fingerprint != "none" else f"omega_selection_{date_str}_"
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=prefix,
        delete=False
    )
    tmp_file.write(json_str)
    tmp_file.close()
    json_filepath = tmp_file.name


    if not selected_utxos:
        return "<div style='color:#ff3366; text-align:center; padding:30px;'>No UTXOs selected for pruning!</div>", gr.update(), gr.update(), "",gr.update(visible=False), gr.update(visible=False), None, []    

    distinct_addrs = len(set(u["address"] for u in selected_utxos))
    privacy_score = calculate_privacy_score(selected_utxos, len(enriched_state))
    cioh_warning = get_cioh_warning(len(selected_utxos), distinct_addrs, privacy_score)
    score_color = "#0f0" if privacy_score >= 70 else "#ff9900" if privacy_score >= 40 else "#ff3366"


    # === CANONICAL ECONOMICS — SINGLE SOURCE OF TRUTH ===
    try:
        econ = estimate_tx_economics(selected_utxos, fee_rate, dao_percent)
    except ValueError:
        return "<div style='color:#ff3366; text-align:center; padding:30px;'>Invalid selection.</div>", gr.update(), gr.update(), "", gr.update(visible=False), gr.update(visible=False), None, []    

    # Unpack for readability and UI
    total_in = econ.total_in
    vsize = econ.vsize
    fee = econ.fee
    change_amt = econ.change_amt
    dao_amt = econ.dao_amt
    input_count = len(selected_utxos)
    input_weight = sum(u["input_weight"] for u in selected_utxos)

    # === PRE-PRUNE SIZE & SAVINGS ===
    all_input_weight = sum(u["input_weight"] for u in enriched_state)
    pre_vsize = max(
        (all_input_weight + 43 * 4 + 31 * 4 + 10 * 4 + len(enriched_state)) // 4 + 10,
        (all_input_weight + 150 + len(enriched_state) * 60) // 4,
    )
    savings_pct = round(100 * (1 - vsize / pre_vsize), 1) if pre_vsize > 0 else 0

    if savings_pct >= 70:
        savings_color, savings_label = "#0f0", "NUCLEAR"
    elif savings_pct >= 50:
        savings_color, savings_label = "#00ff9d", "EXCELLENT"
    elif savings_pct >= 30:
        savings_color, savings_label = "#ff9900", "GOOD"
    else:
        savings_color, savings_label = "#ff3366", "WEAK"

    sats_saved_by_pruning_now = max(0, vsize * future_fee_rate - vsize * fee_rate)

    # === DESTINATION ADDRESS VALIDATION ===
    dest = (dest_addr or "").strip()
    if not dest:
        dest = selected_utxos[0]["address"]  # fallback

    try:
        dest_spk, _ = address_to_script_pubkey(dest)
    except Exception:
        return (
            "<div style='color:#ff3366;text-align:center;padding:30px;'>Invalid or unsupported destination address.</div>",
            gr.update(), gr.update(), "", gr.update(visible=False), gr.update(visible=False), None, []
        )                      

    dest_spk, _ = address_to_script_pubkey(dest)
    dao_spk, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)

    # === BUILD UNSIGNED TRANSACTION ===
    tx = Tx()
    for u in selected_utxos:
        tx.tx_ins.append(TxIn(bytes.fromhex(u["txid"]), int(u["vout"])))

    if dao_amt >= 546:
        tx.tx_outs.append(TxOut(dao_amt, dao_spk))
    if change_amt >= 546:
        tx.tx_outs.append(TxOut(change_amt, dest_spk))

    raw_tx = (
        tx.version.to_bytes(4, 'little') +
        b'\x00\x01' +  # SegWit marker + flag
        encode_varint(len(tx.tx_ins)) +
        b''.join(i.serialize() for i in tx.tx_ins) +
        encode_varint(len(tx.tx_outs)) +
        b''.join(o.serialize() for o in tx.tx_outs) +
        tx.locktime.to_bytes(4, 'little')
    )

    psbt_b64 = create_psbt(raw_tx.hex())

    # === QR CODE GENERATION ===
    box_size = 6
    error_correction = qrcode.constants.ERROR_CORRECT_L if len(psbt_b64) > 2800 else qrcode.constants.ERROR_CORRECT_M
    qr = qrcode.QRCode(version=None, error_correction=error_correction, box_size=box_size, border=4)
    qr.add_data(f"bitcoin:?psbt={psbt_b64}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#f7931a", back_color="#000000")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    qr_img_html = f'<img src="{qr_uri}" style="width:100%;height:auto;display:block;image-rendering:crisp-edges;border-radius:12px;" alt="NUCLEAR QR"/>'

    qr_warning_html = ""
    if len(psbt_b64) > 2800:
        qr_warning_html = """
        <div style="margin-top:12px;padding:10px 14px;border-radius:8px;background:#331a00;color:#ffb347;
                     font-family:monospace;font-size:0.95rem;line-height:1.4;border:1px solid #ff9900aa;">
            Large PSBT detected.<br>
            If QR fails → tap <strong>COPY PSBT</strong> and paste directly.
        </div>
        """
    
     # OUTPUT CONTRACT (do not reorder without updating all .click() chains)
    # 0: html_out              → full PSBT + QR + summary UI
    # 1: gen_btn               → hide generate button (prevent re-click)
    # 2: generate_row          → hide the row
    # 3: summary               → clear old summary
    # 4: export_title_row      → show download title
    # 5: export_file_row       → show file download
    # 6: export_file           → tempfile path
    # 7: selection_snapshot_state → full snapshot dict for audit
    
    return f"""
    <div style="text-align:center;margin:60px auto 0px;max-width:960px;">
    <div style="display:inline-block;padding:55px;background:#000;border:14px solid #f7931a;border-radius:36px;
                box-shadow:0 0 140px rgba(247,147,26,0.95);
                background:radial-gradient(circle at center,#0a0a0a 0%,#000 100%);">

        <div style="margin:40px 0 60px;">
            <button disabled style="padding:30px 100px;font-size:2.4rem;font-weight:800;
                                    background:#000;color:#0f0;letter-spacing:8px;
                                    border:8px solid #0f0;border-radius:34px;
                                    box-shadow:0 0 100px #0f0;text-shadow:0 0 20px #0f0;cursor:not-allowed;">
                RAW UNSIGNED TRANSACTION
            </button>
        </div>
        <div style="margin:40px 0;padding:16px;background:#001100;border:2px solid #0f0;border-radius:12px;
            box-shadow:0 0 40px rgba(0,255,0,0.6);font-family:monospace;">
    <strong style="color:#0f0;font-size:1.1rem;">SELECTION FINGERPRINT</strong><br>
<span style="color:#00ff9d;font-size:1.6rem;letter-spacing:4px;font-family:monospace;">{fingerprint.upper()}</span><br>
<small style="color:#aaa;">Locked inputs • Verify this matches any exported selection</small>
</div>
<div style='text-align:center;margin:40px 0;padding:20px;background:#001133;border:3px solid #4488ff;border-radius:16px;
            box-shadow:0 0 50px rgba(68,136,255,0.5);'>
    <strong style='color:#44ccff;font-size:1.4rem;font-weight:800;text-shadow:0 0 15px #4488ff;'>
        🧊 Frozen Selection Snapshot (JSON)
    </strong><br><br>
    <span style='color:#aaddff;font-size:1.1rem;'>
        Your exact pruned inputs are locked below.<br>
        Download for backup, audit, or future verification.
    </span>
</div>

        <div style="margin:0 auto 40px;width:520px;max-width:96vw;padding:20px;background:#000;
                    border:8px solid #0f0;border-radius:24px;box-shadow:0 0 60px #0f0,inset 0 0 40px #0f0;">
            {qr_img_html}
        </div>

        {qr_warning_html}

        <!-- NUCLEAR SUMMARY BLOCK -->
        <div style='text-align:center;margin:40px 0;padding:18px;background:#111;border:2px solid #f7931a;
                    border-radius:14px;max-width:95%;font-size:1.4rem;line-height:2.1;'>
            <span style='color:#fff;font-weight:600;'>Inputs:</span> 
            <span style='color:#0f0;font-weight:800;'>{input_count:,}</span><br>

            <span style='color:#fff;font-weight:600;'>Total Value Pruned:</span> 
            <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(total_in)}</span><br>

            <span style='color:#fff;font-weight:600;'>Input Weight:</span> 
            <span style='color:#0f0;font-weight:800;'>{input_weight:,} wu ({input_weight//4:,} vB)</span><br>

            <span style='color:#fff;font-weight:600;'>Pre-prune tx size:</span> 
            <span style='color:#888;font-weight:700;'>{pre_vsize:,} vB</span><br>

            <span style='color:#fff;font-weight:600;'>Post-prune tx size:</span> 
            <span style='color:#0f0;font-weight:800;'>{vsize:,} vB</span>
            <span style='color:{savings_color};font-weight:800;font-size:1.2rem;letter-spacing:1.5px;
                        text-transform:uppercase;text-shadow:0 0 20px {savings_color}, 0 0 50px {savings_color};'>
                {' ' + savings_label} (-{savings_pct}%)
            </span><br>

            {(
                f"<span style='color:#fff;font-weight:600;'>Sats saved by pruning now:</span> "
                f"<span style='color:#0f0; font-size:1.45rem; font-weight:800; letter-spacing:1.8px; "
                f"text-shadow:0 0 12px #0f0, 0 0 30px #0f0;'>+{sats_saved_by_pruning_now:,}</span>"
                f" <span style='color:#0f0; font-weight:800; letter-spacing:3.5px; text-transform:uppercase; "
                f"text-shadow:0 0 20px #0f0, 0 0 50px #0f0;'>NUCLEAR MOVE</span>"
                if sats_saved_by_pruning_now >= 100_000 else
                f"<span style='color:#fff;font-weight:600;'>Sats saved by pruning now:</span> "
                f"<span style='color:#0f0; font-size:1.35rem; font-weight:800; letter-spacing:1.2px; "
                f"text-shadow:0 0 10px #0f0, 0 0 25px #0f0;'>+{sats_saved_by_pruning_now:,}</span>"
                if sats_saved_by_pruning_now > 0 else ""
            )}<br>

            <span style='color:#fff;font-weight:600;'>Final Fee:</span> 
            <span style='color:#0f0;font-weight:800;'>{fee:,} sats</span> 
            <span style='color:#0f0;font-weight:600;'>@</span> 
            <strong style='color:#0f0;font-weight:800;'>{fee_rate} s/vB</strong><br>
            <span style='color:#fff;font-weight:600;'>Privacy Score:</span> 
            <span style='color:{score_color};font-weight:800;font-size:1.6rem;text-shadow:0 0 20px {score_color};'>
                {privacy_score}/100
            </span>
              <div style="margin-top:12px; margin-bottom:12px;">{cioh_warning}</div>
            <span style='color:#fff;font-weight:600;'>Change:</span> 
            <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(change_amt)}</span>
            {f" <span style='color:#ff6600;'>(dust absorbed)</span>" if change_amt == 0 and econ.remaining > 0 else ""}

            {f" • <span style='color:#ff6600;'>DAO:</span> <span style='color:#0f0;font-weight:800;'>{sats_to_btc_str(dao_amt)}</span>" if dao_amt >= 546 else ""}
        <br>
        </div>

        <!-- PSBT + COPY BUTTON -->
        <div style="margin:60px auto 20px;width:92%;max-width:880px;">
            <div style="position:relative;background:#000;border:6px solid #f7931a;border-radius:18px;
                        box-shadow:0 0 40px #0f0;overflow:hidden;">
                <textarea id="psbt-output" readonly 
                    style="width:100%;height:180px;background:#000;color:#0f0;font-size:1rem;
                           padding:24px;padding-right:140px;border:none;outline:none;resize:none;
                           font-family:monospace;font-weight:700;box-sizing:border-box;">
{psbt_b64}</textarea>
                <button onclick="navigator.clipboard.writeText(document.getElementById('psbt-output').value).then(() => {{ 
                    this.innerText='COPIED'; setTimeout(() => this.innerText='COPY PSBT', 1500); 
                }})"
                    style="position:absolute;top:14px;right:14px;padding:12px 30px;background:#f7931a;
                           color:#000;border:none;border-radius:14px;font-weight:800;letter-spacing:1.5px;
                           font-size:1.12rem;text-transform:uppercase;cursor:pointer;
                           box-shadow:0 0 30px #f7931a, inset 0 0 20px rgba(0,0,0,0.4);z-index:10;">
                    COPY PSBT
                </button>
            </div>
            <div style="text-align:center;margin-top:12px;">
                <span style="color:#00f0ff;font-weight:700;text-shadow:0 0 10px #0f0;">RBF enabled</span>
                <span style="color:#888;"> • Raw PSBT • Tap COPY to clipboard</span>
            </div>
        </div>

        <div style='color:#ff9900;font-size:1rem;text-align:center;margin:40px 0 20px;padding:16px;
                    background:#220000;border:2px solid #f7931a;border-radius:12px;
                    box-shadow:0 0 40px rgba(247,147,26,0.4);'>
            <div style='color:#fff;font-weight:800;text-shadow:0 0 12px #f7931a;'>
                Important: Wallet must support <strong style='color:#0f0;text-shadow:0 0 15px #0f0;'>PSBT</strong>
            </div>
            <div style='color:#0f8;margin-top:8px;opacity:0.9;'>
                Sparrow • BlueWallet • Electrum • UniSat • OK
            </div>
        </div>

    </div>
    </div>
            """, gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=True), gr.update(visible=True), json_filepath, snapshot

# --------------------------
# Gradio UI
# --------------------------

with gr.Blocks(
    title="Ωmega Prunerv10.6 — BATCH NUCLEAR + OFFLINE MODE"
) as demo:
    # NUCLEAR SOCIAL PREVIEW — THIS IS ALL YOU NEED NOW
    gr.HTML("""
    <meta property="og:title" content="Ωmega Pruner v10.6 — BATCH NUCLEAR + OFFLINE MODE">
    <meta property="og:description" content="The cleanest open-source UTXO consolidator. Zero custody. Full coin-control. RBF. Taproot.">
    <meta property="og:image" content="https://omega-pruner.onrender.com/docs/omega_thumbnail.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:url" content="https://omega-pruner.onrender.com">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    """, visible=False)

    # Ωmega Background and Floating Banner
    gr.HTML("""
   <!-- Ωmega Background — full-screen animated Ω -->
    <div id="omega-bg" style="
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: -1;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: transparent;
    ">
        <span class="omega-wrap" style="
            display: inline-block;
            animation: gradient-pulse 18s infinite ease-in-out,
                       omega-spin 120s linear infinite;
        ">
            <span class="omega-symbol" style="
                font-size: 100vh;
                font-weight: 900;
                background: linear-gradient(135deg, rgba(247,147,26,0.28), rgba(247,147,26,0.15));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                color: transparent;
                text-shadow: 0 0 220px rgba(247,147,26,0.72);
                animation: omega-breath 28s infinite ease-in-out;
                user-select: none;
                line-height: 1;
                opacity: 0.96;
            ">Ω</span>
        </span>
    </div>

    <!-- Hero Welcome Banner — Semi-Transparent for Ω Visibility -->
    <div style="text-align:center;margin:100px 0 60px 0;padding:60px 40px;
                background:rgba(0,0,0,0.58);  /* ← Key: semi-transparent black */
                backdrop-filter: blur(8px);    /* Optional: subtle blur for depth */
                border:8px solid #f7931a;
                border-radius:32px;
                box-shadow:
                    0 0 100px rgba(247,147,26,0.5),
                    inset 0 0 80px rgba(247,147,26,0.1);
                max-width:900px;margin-left:auto;margin-right:auto;
                position:relative;z-index:1;">
        
        <div style="color:#f7931a;
                    font-size:4.8rem;
                    font-weight:900;
                    letter-spacing:12px;
                    text-shadow:
                        0 0 40px #f7931a,
                        0 0 80px #ffaa00,
                        0 0 120px rgba(247,147,26,0.9);
                    margin-bottom:30px;">
            ΩMEGA PRUNER
        </div>
        
        <div style="color:#0f0;
                    font-size:2.2rem;
                    font-weight:900;
                    letter-spacing:4px;
                    text-shadow:0 0 30px #0f0, 0 0 60px #0f0;
                    margin:40px 0;">
            NUCLEAR COIN CONTROL
        </div>
        
    <div style="color:#ddd;
            font-size:1.5rem;
            line-height:1.8;
            max-width:760px;
            margin:0 auto 50px auto;">
    Pruning isn't just about saving sats today — it's about <strong style="color:#0f0;">taking control</strong> of your coins for the long term.<br><br>
    
    By consolidating inefficient UTXOs, you:<br>
    • <strong style="color:#00ff9d;">Save significantly on fees</strong> when the network gets busy<br>
    • <strong style="color:#00ff9d;">Gain real coin control</strong> — know exactly what you're spending<br>
    • <strong style="color:#00ff9d;">Improve privacy</strong> when done thoughtfully<br>
    • <strong style="color:#00ff9d;">Future-proof your stack</strong> — stay spendable no matter what<br><br>

    <strong style="color:#f7931a;font-size:1.8rem;font-weight:900;letter-spacing:1px;">
        Prune now. Win forever.
    </strong><br><br>
    
    Paste one or more addresses (or xpubs) below and click 
    <strong style="color:#f7931a;font-size:1.7rem;">ANALYZE</strong> 
    to see your personalized pruning strategy.
</div>
        <div style="font-size:4rem;color:#f7931a;opacity:0.9;animation:pulse 2s infinite;">
            ↓
        </div>
    </div>

    <style>
    /* Pulsing arrow animation */
    @keyframes pulse {
        0%, 100% { transform: translateY(0); opacity: 0.8; }
        50% { transform: translateY(20px); opacity: 1; }
    }

    /* Ωmega Animations */
    @keyframes omega-breath {
        0%, 100% { opacity: 0.76; transform: scale(0.95) rotate(0deg); }
        50%      { opacity: 1.0;  transform: scale(1.05) rotate(180deg); }
    }

    @keyframes gradient-pulse {
        0%, 100% { transform: scale(0.97); }
        50%      { transform: scale(1.03); }
    }

    @keyframes omega-spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Container & body tweaks */
    .gradio-container { 
        position: relative;
        z-index: 0;
        background: transparent;
        overflow-y: auto;
    }

    body { 
        overflow-y: auto;
    }

    #omega-bg { 
        isolation: isolate;
        will-change: transform, opacity;
    }

    /* Fee button glow */
    .fee-btn button:not(:disabled),
    .fee-btn [role="button"]:not(:disabled) {
        box-shadow: 0 0 20px rgba(247,147,26,0.6);
        animation: fee-glow 3s infinite alternate;
    }

    .fee-btn button:disabled,
    .fee-btn [role="button"]:disabled {
        box-shadow: none;
        animation: none;
        opacity: 0.35;
    }

    @keyframes fee-glow {
        from { box-shadow: 0 0 20px rgba(247,147,26,0.6); }
        to   { box-shadow: 0 0 40px rgba(247,147,26,0.9); }
    }

    /* Slider halo */
    .gr-slider::after {
        content: '';
        position: absolute;
        inset: -8px;
        border-radius: 12px;
        pointer-events: none;
        opacity: 0;
        box-shadow: 0 0 30px rgba(247,147,26,0.8);
        animation: slider-halo 3.5s infinite alternate;
        transition: opacity 0.5s;
    }

    .gr-slider:not(:has(input:disabled))::after { opacity: 1; }
    .gr-slider:has(input:disabled)::after { opacity: 0; animation: none; }

    @keyframes slider-halo {
        from { box-shadow: 0 0 25px rgba(247,147,26,0.7); }
        to   { box-shadow: 0 0 45px rgba(247,147,26,1); }
    }

    /* Locked badge */
    @keyframes badge-pulse {
        0%   { transform: scale(1);   box-shadow: 0 0 60px rgba(0,255,0,0.7); }
        50%  { transform: scale(1.1); box-shadow: 0 0 120px rgba(0,255,0,1); }
        100% { transform: scale(1);   box-shadow: 0 0 60px rgba(0,255,0,0.7); }
    }

    @keyframes badge-entry {
        0%   { opacity: 0; transform: scale(0.3) translateY(-30px); }
        70%  { transform: scale(1.15); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    .locked-badge {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        padding: 18px 48px;
        background: #000;
        border: 7px solid #f7931a;
        border-radius: 28px;
        box-shadow: 0 0 140px rgba(247,147,26,1);
        color: #ffaa00;
        text-shadow: 
            0 0 8px #ffaa00,
            0 0 16px #ffaa00,
            0 0 32px #ffaa00,
            0 0 60px #ffaa00;
        font-weight: 900;
        font-size: 2.4rem;
        letter-spacing: 12px;
        pointer-events: none;
        opacity: 0;
        animation: badge-entry 0.9s cubic-bezier(0.175, 0.885, 0.32, 1.4) forwards,
                   badge-pulse 2.5s infinite alternate 0.9s;
    }
    </style>
    """)

    gr.HTML("""
    <style>
/* Health badge styling — clean and beautiful */
.health {
    font-weight: 900;
    text-align: center;
    padding: 6px 10px;
    border-radius: 4px;
    min-width: 70px;
    display: inline-block;
}
.health-dust     { color: #ff3366; background: rgba(255, 51, 102, 0.1); }
.health-heavy    { color: #ff6600; background: rgba(255, 102, 0, 0.1); }
.health-careful  { color: #ff00ff; background: rgba(255, 0, 255, 0.1); }
.health-medium   { color: #ff9900; background: rgba(255, 153, 0, 0.1); }
.health-optimal  { color: #00ff9d; background: rgba(0, 255, 157, 0.1); }
.health-manual   { color: #bb86fc; background: rgba(187, 134, 252, 0.1); }


.health small {
    display: block;
    color: #aaa;
    font-weight: normal;
    font-size: 0.8em;
    margin-top: 2px;
}
/* Make disabled textboxes obviously grayed out */
.gr-textbox input:disabled {
    background-color: #111 !important;
    color: #555 !important;
    opacity: 0.6;
    cursor: not-allowed;
}
</style>
        """)
    # =============================
    # — BACKGROUND FEE CACHE REFRESH —
    # =============================
    def refresh_fees_periodically():
        while True:
            time.sleep(30)
            try:
                get_live_fees()  # Keeps the internal cache warm
            except Exception as e:
                log.warning(f"Error during background fee refresh: {e}")

    # Start the daemon thread immediately on import — with guard against multiple starts (e.g., hot-reload)
    if not hasattr(threading, "_fee_refresh_started"):
        fee_refresh_thread = threading.Thread(target=refresh_fees_periodically, daemon=True)
        fee_refresh_thread.start()
        threading._fee_refresh_started = True

    # =============================
    # — LOCK-SAFE FEE PRESET FUNCTION —
    # =============================
    def apply_fee_preset_locked(
        df_rows,
        enriched_state,
        future_fee_slider,
        thank_you_slider,
        locked,
        current_strategy,
        preset: str
    ):
        if locked:
            return gr.update(), gr.update()  # No change to slider or summary when locked

        # Safely extract current values (handles both direct value and component)
        future_fee = (
            future_fee_slider.value if hasattr(future_fee_slider, "value") else int(future_fee_slider or 60)
        )
        thank_you = (
            thank_you_slider.value if hasattr(thank_you_slider, "value") else float(thank_you_slider or 0.5)
        )

        future_fee = max(5, min(500, future_fee))
        thank_you = max(0, min(5, thank_you))

        # Live fees with safe fallback
        fees = get_live_fees() or {
            "fastestFee": 10,
            "halfHourFee": 6,
            "hourFee": 3,
            "economyFee": 1,
            "minimumFee": 1,
        }

        # Correct keys + defensive .get()
        rate_map = {
            "fastest": fees.get("fastestFee", 10),
            "half_hour": fees.get("halfHourFee", 6),
            "hour": fees.get("hourFee", 3),
            "economy": fees.get("economyFee", 1),
        }

        new_rate = rate_map.get(preset, 3)

        # Update summary with new rate
        banner_html, button_vis, details_html = generate_summary_safe(
            df_rows,
            enriched_state,
            new_rate,
            future_fee,
            thank_you,
            locked,
            current_strategy
        )

        # CRITICAL: Explicit slider update + details block
        return gr.update(value=new_rate), details_html
   

    # =================================================================
    # ========================= UI STARTS HERE ========================
    # =================================================================
    with gr.Column():
        # Mode status — big, bold, impossible to miss
        mode_status = gr.Markdown(
            value="**Online mode** • API calls enabled",
            elem_classes="mode-status"  # optional: for extra styling
        )

        with gr.Row():
            offline_toggle = gr.Checkbox(
                label="🔒 Offline mode — no internet / API calls (fully air-gapped)",
                value=False,
                interactive=True,
                info="Disables all network requests. Safe for air-gapped or privacy-focused use.",
            )

        addr_input = gr.Textbox(
            label="Address or xpub (one per line for batch mode) — 100% non-custodial, keys never entered",
            placeholder="Paste one or many addresses/xpubs (one per line)\nClick ANALYZE when ready",
            lines=6,
        )

        with gr.Row(visible=False) as manual_box_row:
            manual_utxo_input = gr.Textbox(
                label="🔒 OFFLINE MODE • ACTIVE INPUT • Paste raw UTXOs (one per line) • Format: txid:vout:value_in_sats  (address optional at end)",
                placeholder="""Paste raw UTXOs — one per line

Format: txid:vout:value_in_sats[:address]

Examples:
abc123...000:0:125000:bc1qexample...
def456...789:1:5000000          ← 0.05 BTC, address optional
txidhere:2:999999

No API calls • Fully air-gapped safe""",
                lines=10,
            )

        # === Seamless mode switching with guidance ===
        offline_toggle.change(
            fn=lambda x: gr.update(visible=x, value="" if not x else gr.update()),
            inputs=offline_toggle,
            outputs=manual_box_row,
        ).then(
            # Clear main address box when entering offline
            fn=lambda x: gr.update(value="") if x else gr.update(),
            inputs=offline_toggle,
            outputs=addr_input,
        ).then(
            # Gray out + change placeholder to guide user to the correct box
            fn=lambda x: gr.update(
                interactive=not x,
                placeholder="🔒 Offline mode active — paste raw UTXOs in the box below 👇" if x
                else "Paste one or many addresses/xpubs (one per line)\nClick ANALYZE when ready"
            ),
            inputs=offline_toggle,
            outputs=addr_input,
        ).then(
            # Update mode status banner
            fn=lambda x: 
                "**🔒 Offline mode** • No API calls • Fully air-gapped" if x 
                else "**Online mode** • API calls enabled",
            inputs=offline_toggle,
            outputs=mode_status,
        )

    # === Now outside the Column — back to base indentation ===
    strategy_state = gr.State("Recommended — ~40% pruned (balanced savings & privacy)")

    with gr.Row():
        strategy = gr.Dropdown(
            choices=[
                "Privacy First — ~30% pruned (lowest CIOH risk)",
                "Recommended — ~40% pruned (balanced savings & privacy)",
                "More Savings — ~50% pruned (stronger fee reduction)",
                "NUCLEAR PRUNE — ~90% pruned (maximum savings, highest CIOH)",
            ],
            value="Recommended — ~40% pruned (balanced savings & privacy)",
            label="Pruning Strategy — fee savings vs privacy (Common Input Ownership Heuristic)",
        )

    strategy.change(fn=lambda x: x, inputs=strategy, outputs=strategy_state)

    with gr.Row():
        dust = gr.Slider(0, 5000, 546, step=1, label="Dust Threshold (sats)")
        dest = gr.Textbox(
            label="Change Address (optional)",
            placeholder="Leave blank → reuse first input",
        )

    with gr.Row():
        fee_rate = gr.Slider(
            1, 300, 15, step=1, label="Fee Rate now (sat/vB)", scale=3,
        )
        future_fee = gr.Slider(
            5,
            500,
            60,
            step=1,
            label="Future fee rate in 3–6 months (sat/vB)",
            scale=3,
        )
        thank_you = gr.Slider(
            0, 5, 0.5, step=0.1, label="Thank-You / DAO Donation (%)", scale=2,
        )

    # Fee preset buttons
    with gr.Row():
        economy_btn = gr.Button("Economy", size="sm", elem_classes="fee-btn")
        hour_btn = gr.Button("1 hour", size="sm", elem_classes="fee-btn")
        halfhour_btn = gr.Button("30 min", size="sm", elem_classes="fee-btn")
        fastest_btn = gr.Button("Fastest", size="sm", elem_classes="fee-btn")
        
    html_out = gr.HTML()

    
    with gr.Row(visible=False) as generate_row:
        gen_btn = gr.Button("2. GENERATE NUCLEAR PSBT", variant="primary", elem_id="generate-btn")
    

    # Rest of outputs & State
   
    summary = gr.HTML()
    enriched_state = gr.State([])
    locked = gr.State(False)
    locked_badge = gr.HTML("")  # Starts hidden
    selection_snapshot_state = gr.State({})   # dict


    analyze_btn = gr.Button("1. ANALYZE & LOAD UTXOs", variant="primary")

    # Export button and file output
    # First row: the title — centered and prominent
    with gr.Row(visible=False) as export_title_row:
        gr.HTML("""
        <div style='text-align:center;padding:0 0 20px 0;'>
            <strong style='color:#44ccff;font-size:1.8rem;font-weight:900;
                          text-shadow: 0 3px 8px #000000ee, 0 0 20px #000000aa;'>
                📄 Your Frozen Selection
                </strong><br>
            <span style='color:#aaddff;font-size:1.1rem;text-shadow: 0 3px 8px #000000ee, 0 0 20px #000000aa;'>
                Download the JSON below for backup or audit
                </span>
            </div>
        """)

    # Second row: the actual file download
    with gr.Row(visible=False) as export_file_row:
        export_file = gr.File(
            label="",                     # no duplicate label
            interactive=False
        )

    df = gr.DataFrame(
        headers=[
            "PRUNE",
            "Source",
            "TXID",
            "vout",
            "Value (sats)",
            "Address",
            "Weight (wu)",
            "Type",
            "Health",
        ],
        datatype=["bool", "str", "str", "number", "number", "str", "number", "str", "html"],
        type="array",
        interactive=True,
        wrap=True,
        row_count=(50, "dynamic"),
        max_height=500,
        max_chars=None,
        label="CHECK TO PRUNE • Pre-checked = recommended • OPTIMAL = ideal • DUST/HEAVY = prune",
        static_columns=[1, 2, 3, 4, 5, 6, 7],
        column_widths=["90px", "160px", "200px", "70px", "140px", "160px", "130px", "90px", "100px"]
    )

    reset_btn = gr.Button("NUCLEAR RESET · START OVER", variant="secondary")
    # =============================
    # — FEE PRESET BUTTONS WIRING —
    # =============================
    for btn, preset in [
        (economy_btn, "economy"),
        (hour_btn, "hour"),
        (halfhour_btn, "half_hour"),
        (fastest_btn, "fastest"),
    ]:
        btn.click(
            fn=partial(apply_fee_preset_locked, preset=preset),
            inputs=[df, enriched_state, future_fee, thank_you, locked, strategy_state],
            outputs=[fee_rate, summary],
        )
    # =============================
    # — ANALYZE BUTTON —
    # =============================
    analyze_btn.click(
        fn=analyze,
        inputs=[
            addr_input,          # renamed from addr
            strategy,
            dust,
            dest,
            fee_rate,
            thank_you,           # dao_slider
            future_fee,
            offline_toggle,      # new
            manual_utxo_input],        # new
        outputs=[html_out, df, enriched_state, summary, gen_btn, generate_row],
    ).then(lambda: gr.update(visible=False), outputs=analyze_btn)

    # =============================
    # — GENERATE BUTTON (NUCLEAR LOCK + ANIMATED BADGE) —
    # =============================
    gen_btn.click(
        fn=generate_psbt,
        inputs=[dest, fee_rate, future_fee, thank_you, df, enriched_state],
        outputs=[html_out, gen_btn, generate_row, summary, export_title_row, export_file_row, export_file, selection_snapshot_state],
    ).then(
        lambda: gr.update(interactive=False),  # Gray out dataframe checkboxes
        outputs=df,
    ).then(
        lambda: True,  # Set locked = True
        outputs=locked,
    ).then(
        lambda: [
            gr.update(interactive=False),  # addr_input (main address box)
            gr.update(interactive=False),  # strategy
            gr.update(interactive=False),  # dust
            gr.update(interactive=False),  # dest
            gr.update(interactive=False),  # fee_rate
            gr.update(interactive=False),  # future_fee
            gr.update(interactive=False),  # thank_you
            gr.update(interactive=False),  # offline_toggle
            gr.update(interactive=False),  # manual_utxo_input (disable even if visible)
            gr.update(interactive=False),  # fastest_btn
            gr.update(interactive=False),  # halfhour_btn
            gr.update(interactive=False),  # hour_btn
            gr.update(interactive=False),  # economy_btn
            "<div class='locked-badge'>LOCKED</div>",
        ],
        outputs=[
            addr_input,          # ← updated
            strategy,
            dust,
            dest,
            fee_rate,
            future_fee,
            thank_you,
            offline_toggle,      # ← new: lock the toggle
            manual_utxo_input,        # ← new: lock the manual box
            fastest_btn,
            halfhour_btn,
            hour_btn,
            economy_btn,
            locked_badge,
        ],
    )

    # =============================
    # — NUCLEAR RESET BUTTON —
    # =============================
    def nuclear_reset():
        return (
            "",                                          # html_out
            [],                                          # df
            [],                                          # enriched_state
            "",                                          # summary
            gr.update(visible=True),                     # ANALYZE button visible
            gr.update(visible=False),                    # GENERATE row hidden
            gr.update(value="", interactive=True),       # addr_input (main address box)
            gr.update(value="Recommended — ~40% pruned (balanced savings & privacy)", interactive=True),  # strategy
            gr.update(value=546, interactive=True),      # dust
            gr.update(value="", interactive=True),       # dest
            gr.update(value=15, interactive=True),       # fee_rate
            gr.update(value=60, interactive=True),       # future_fee
            gr.update(value=0.5, interactive=True),      # thank_you
            gr.update(value=False, interactive=True),    # offline_toggle = unchecked
            gr.update(value="", visible=False, interactive=True),  # manual_utxo_input = cleared + hidden
            False,                                       # locked state
            "",                                          # clear locked badge
            gr.update(interactive=True),                 # fastest_btn
            gr.update(interactive=True),                 # halfhour_btn
            gr.update(interactive=True),                 # hour_btn
            gr.update(interactive=True),                 # economy_btn
            gr.update(visible=False),                       # export_title_row
            gr.update(visible=False),                      # export_file_row
            None,                                          # export_file
            {}                                         # selection_snapshot_state
        )

    reset_btn.click(
        fn=nuclear_reset,
        inputs=None,
        outputs=[
            html_out,
            df,
            enriched_state,
            summary,
            analyze_btn,
            generate_row,
            addr_input,            
            strategy,
            dust,
            dest,
            fee_rate,
            future_fee,
            thank_you,
            offline_toggle,         
            manual_utxo_input,        
            locked,
            locked_badge,
            fastest_btn,
            halfhour_btn,
            hour_btn,
            economy_btn,
            export_title_row,
            export_file_row,             
            export_file,
            selection_snapshot_state
        ],
    )

    # =============================
    # — LIVE SUMMARY UPDATES —
    # =============================
    df.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate, future_fee, thank_you, locked, strategy_state],
        outputs=[html_out, generate_row, summary],
    )
    fee_rate.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate, future_fee, thank_you, locked, strategy_state],
        outputs=[html_out, generate_row, summary],
    )
    future_fee.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate, future_fee, thank_you, locked, strategy_state],
        outputs=[html_out, generate_row, summary],
    )
    thank_you.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate, future_fee, thank_you, locked, strategy_state],
        outputs=[html_out, generate_row, summary],
    )
    strategy.change(fn=lambda x: x, inputs=strategy, outputs=strategy_state)

    strategy.change(
        fn=analyze,
        inputs=[addr_input, strategy, dust, dest, fee_rate, thank_you, future_fee, offline_toggle, manual_utxo_input],
        outputs=[html_out, df, enriched_state, summary, gen_btn, generate_row],
    )
    dust.change(
        fn=analyze,
        inputs=[addr_input, strategy, dust, dest, fee_rate, thank_you, future_fee, offline_toggle, manual_utxo_input],
        outputs=[html_out, df, enriched_state, summary, gen_btn, generate_row],
    )


    # CRITICAL: DO NOT use .change() on fee_rate/future_fee for anything else
    # 5. FOOTER
    gr.HTML(
    """
    <div style="
        margin: 60px auto 6px auto !important;
        padding: 16px 0 12px 0 !important;
        text-align: center;
        font-size: 0.94rem;
        color: #888;
        opacity: 0.96;
        max-width: 720px;
        line-height: 1.7;
    ">

        <!-- VERSION -->
        <strong style="
            color: #f7931a;
            font-size: 1.08rem;
            letter-spacing: 0.5px;
            text-shadow: 0 0 12px rgba(247,147,26,0.65);
        ">
            Ωmega Pruner v10.6 — BATCH NUCLEAR + OFFLINE MODE
        </strong><br>

        <!-- GITHUB LINK -->
        <a href="https://github.com/babyblueviper1/Viper-Stack-Omega"
           target="_blank"
           rel="noopener"
           style="
               color: #f7931a;
               text-decoration: none;
               font-weight: 600;
               text-shadow: 0 0 10px rgba(247,147,26,0.55);
           ">
            GitHub • Open Source • Apache 2.0
        </a><br><br>

        <!-- CUSTOM BUILDS SECTION — NEON Ω-GREEN -->
        <div style="margin: 6px auto 10px auto; max-width: 650px;">
            <a href="https://www.babyblueviper.com/p/omega-pruner-custom-builds"
               target="_blank"
               style="
                   font-size: 0.96rem;
                   color: #dcdcdc;
                   font-weight: 700;
                   text-decoration: none;
                   letter-spacing: 0.3px;
                   text-shadow: 0 0 18px rgba(0,255,0,0.85);
               ">
                <strong>This build is engineered for speed and clarity.</strong><br>
                <strong>For extended capabilities or tailored integrations, custom versions can be commissioned.</strong>
            </a>
        </div><br>

        <!-- TAGLINE — NUCLEAR BLACK SHADOW, READABLE ON ANY DISPLAY -->
        <span style="
            color: #0f0;
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.6px;
            text-shadow:
                0 0 12px #0f0,          /* neon glow */
                0 0 24px #0f0,          /* outer glow */
                0 0 6px #000,           /* sharp black drop */
                0 4px 10px #000,        /* deep shadow */
                0 8px 20px #000000e6;   /* massive black halo */
        ">
            Prune today. Win forever. • Ω
        </span>
    </div>
    """,
    elem_id="omega_footer",
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False, debug=False)
