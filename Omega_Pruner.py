"""
CANONICAL STATE MODEL (AUTHORITATIVE — DO NOT VIOLATE)

User Inputs (mutable via UI):
- addr_input
- strategy (pruning strategy dropdown)
- dust_threshold
- dest_addr
- fee_rate_slider
- dao_slider
- future_fee_slider
- offline_mode
- manual_utxo_input

Derived State (write-once per analyze()):
- enriched_state: List[dict] — full UTXO set with health, weights, script_type, source
  → ONLY written by analyze()
  → NEVER mutated after creation
  → Single source of truth for all downstream logic

Selection (user intent):
- df_rows: checkbox state from Dataframe
  → Resolved via _resolve_selected(df_rows, enriched_state)

Phase (derived — do not store):
- "init": no enriched_state
- "analyzed": enriched_state present, locked = False
- "locked": locked = True (after generate_psbt)

RULES:
1. Only analyze() may assign enriched_state
2. No function may mutate enriched_state contents
3. Strategy changes require re-running analyze()
4. After lock, no economic recomputation — use frozen values only
5. All summary/economics must derive from enriched_state + current sliders (pre-lock) or frozen (post-lock)

Violations will be treated as bugs.
"""
# Omega_Pruner.py
import gradio as gr
import requests, time, base64, io, qrcode, json, os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import warnings, logging
from functools import partial
import threading
import hashlib
import tempfile
from datetime import datetime
import copy

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
HEALTH_COL   = 3 
VALUE_COL    = 4
ADDRESS_COL  = 5
WEIGHT_COL   = 6
TYPE_COL     = 7
VOUT_COL     = 8


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


from threading import Lock
_fee_cache_lock = Lock()


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
    # Safe cache check — avoid AttributeError on first call
    if (
        hasattr(get_live_fees, "cache")
        and hasattr(get_live_fees, "cache_time")
        and time.time() - get_live_fees.cache_time < 30
    ):
        return get_live_fees.cache

    try:
        r = session.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        if r.status_code == 200:
            data = r.json()
            fees = {
                "fastest": int(data["fastestFee"]),
                "half_hour": int(data["halfHourFee"]),
                "hour": int(data["hourFee"]),
                "economy": int(data["economyFee"]),
                "minimum": int(data["minimumFee"]),
            }
            now = time.time()
            with _fee_cache_lock:
                get_live_fees.cache = fees
                get_live_fees.cache_time = now
            return fees
    except requests.exceptions.RequestException as e:
        log.warning(f"Failed to fetch live fees: {e}")

    # Fallback conservative defaults
    return {
        "fastest": 10,
        "half_hour": 6,
        "hour": 3,
        "economy": 1,
        "minimum": 1,
    }

def build_psbt_snapshot(
    enriched_state: tuple,
    fee_rate: int,
    future_fee_rate: int,
    dao_percent: float,
    dest_addr_override: str | None,
    scan_source: str,
) -> dict:
    """
    Create an immutable transaction snapshot — point of no return.
    Records user intent only. No destination resolution or PSBT construction here.
    
    Canonical-compliant: does not mutate inputs.
    """
    # Select only UTXOs that were marked by the user
    selected_utxos = [u for u in enriched_state if u.get("selected", False)]
    if not selected_utxos:
        raise ValueError("No UTXOs selected for pruning")

    # Build snapshot dict
    snapshot = {
        "version": 1,
        "timestamp": int(time.time()),
        "scan_source": scan_source.strip(),  # original user input
        "dest_addr_override": dest_addr_override.strip() if dest_addr_override and dest_addr_override.strip() else None,
        "fee_rate": fee_rate,
        "future_fee_rate": future_fee_rate,
        "dao_percent": dao_percent,
        "inputs": copy.deepcopy(selected_utxos),  # prevent external mutation
    }

	# Deterministic fingerprint — computed over snapshot excluding fingerprint fields
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    snapshot["fingerprint"] = fingerprint
    snapshot["fingerprint_short"] = fingerprint[:16].upper()

    return snapshot

def _coerce_int(value, default: int) -> int:
    """
    Safely coerce a value to int, falling back to default on failure.
    Handles None, "", malformed strings, floats, etc.
    Used for slider inputs that can arrive as None or empty in Gradio edge cases.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default

def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def sync_selection(df_rows, enriched_state, locked):
    """Sync checkbox changes back to canonical enriched_state — disabled when locked."""
    if locked:
        return enriched_state  # ignore all checkbox changes post-lock

    if not enriched_state or not df_rows:
        return enriched_state

    if len(df_rows) != len(enriched_state):
        log.warning("Row count mismatch during sync — ignoring")
        return enriched_state

    updated = []
    for row, u in zip(df_rows, enriched_state):
        new_u = dict(u)
        new_u["selected"] = bool(row[CHECKBOX_COL])
        updated.append(new_u)

    return tuple(updated)

def update_enriched_from_df(df_rows: List[list], enriched_state: tuple, locked: bool) -> tuple:
    """
    Live-sync checkbox changes into enriched_state.
    Returns a new immutable tuple (preserves your frozen model).
    Ignores changes if locked.
    """
    if locked or not enriched_state:
        return enriched_state
    
    if len(df_rows) != len(enriched_state):
        log.warning("Row count mismatch in live selection sync — ignoring update")
        return enriched_state
    
    # Use your existing battle-tested sync_selection
    updated_list = sync_selection(df_rows, list(enriched_state), locked=False)  # locked already checked above
    return tuple(updated_list)
        
def load_selection(json_file, current_enriched):
    if not json_file:
        return current_enriched, "No file selected"
    
    try:
        with open(json_file.name, "r") as f:
            snapshot = json.load(f)
        
        if not isinstance(snapshot, dict) or "inputs" not in snapshot:
            return current_enriched, "Invalid Ωmega Pruner selection file"
        
        selected_keys = {
            (u["txid"], u["vout"]) 
            for u in snapshot.get("inputs", []) 
            if isinstance(u, dict) and "txid" in u and "vout" in u
        }
        
        if not current_enriched:
            return current_enriched, (
                "<div style='color:#00ff9d;padding:20px;background:#002200;border-radius:12px;text-align:center;'>"
                "<strong>Selection file loaded!</strong><br><br>"
                "Now paste the same addresses/xpubs → click <strong>ANALYZE</strong><br>"
                "Then upload this file again to restore your exact checkboxes."
                "</div>"
            )
        
        updated = []
        matched_count = 0
        for u in current_enriched:
            new_u = dict(u)
            is_selected = (u["txid"], u["vout"]) in selected_keys
            new_u["selected"] = is_selected
            if is_selected:
                matched_count += 1
            updated.append(new_u)
        
        if matched_count == 0:
            message = (
                "<div style='color:#ff9900;padding:20px;background:#331100;border-radius:12px;text-align:center;'>"
                "<strong>Selection loaded — no matching UTXOs found</strong><br><br>"
                f"File contains {len(selected_keys)} UTXOs.<br>"
                "They don't match current analysis (different addresses?).<br>"
                "Checkboxes not restored."
                "</div>"
            )
        else:
            message = f"Selection loaded — {matched_count}/{len(selected_keys)} UTXOs restored"
        
        return tuple(updated), message
    
    except Exception as e:
        return current_enriched, f"Failed to load: {str(e)}"

def rebuild_df_rows(enriched_state):
    if not enriched_state:
        return []
    
    # Sort same as in analyze() for consistency
    enriched_sorted = sorted(enriched_state, key=lambda u: HEALTH_PRIORITY[u["health"]])
    
    df_rows = []
    for u in enriched_sorted:
        health_html = f'<div class="health health-{u["health"].lower()}">{u["health"]}<br><small>{u["recommend"]}</small></div>'
        df_rows.append([
            u.get("selected", False),
            u.get("source", "Single"),
            u["txid"],
            health_html,
            u["value"],
            u["address"],
            u["input_weight"],
            u["script_type"],
            u["vout"],
        ])
    return df_rows


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
            "<div style='margin-top:16px;padding:16px;background:#440000;border:3px solid #ff3366;border-radius:14px;"
            "box-shadow:0 0 50px rgba(255,51,102,0.9);font-size:1.22rem;line-height:1.7;'>"
            "<strong style='color:#ff3366;font-size:1.45rem;font-weight:900;'>EXTREME CIOH LINKAGE</strong><br>"
            "<strong style='color:#ff6688;font-size:1.15rem;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            "This consolidation strongly proves common ownership of many inputs/addresses.<br>"
            "Privacy significantly reduced. Consider CoinJoin, PayJoin, or silent payments afterward."
            "</div>"
        )
    elif privacy_score <= 50:
        return (
            "<div style='margin-top:14px;padding:14px;background:#331100;border:2px solid #ff8800;border-radius:12px;"
            "font-size:1.18rem;line-height:1.6;'>"
            "<strong style='color:#ff9900;font-size:1.35rem;font-weight:900;'>High CIOH Risk</strong><br>"
            "<strong style='color:#ffaa44;font-size:1.12rem;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            f"Merging {input_count} inputs from {distinct_addrs} address(es) → analysts will cluster them as yours.<br>"
            "Good fee savings, but real privacy trade-off."
            "</div>"
        )
    elif privacy_score <= 70:
        return (
            "<div style='margin-top:12px;padding:12px;background:#113300;border:1px solid #00ff9d;border-radius:10px;"
            "color:#aaffaa;font-size:1.15rem;line-height:1.6;'>"
            "<strong style='color:#00ff9d;font-size:1.3rem;font-weight:900;'>Moderate CIOH</strong><br>"
            "<strong style='color:#66ffaa;font-size:1.1rem;'>Common Input Ownership Heuristic (CIOH)</strong><br>"
            "Some linkage created, but not extreme. Acceptable during low-fee periods."
            "</div>"
        )
    else:
        return (
            "<div style='margin-top:10px;color:#aaffaa;font-size:1.05rem;line-height:1.5;'>"
            "Low CIOH impact <strong style='color:#00ffdd;font-size:1.1rem;'>(Common Input Ownership Heuristic)</strong> — minimal new linkage."
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

    # === Legacy P2PKH (starts with '1') ===
    if addr.startswith('1'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x00:  # version byte for mainnet P2PKH
                hash160 = dec[1:21]
                return b'\x76\xa9\x14' + hash160 + b'\x88\xac', {
                    'input_vb': 148,
                    'output_vb': 34,
                    'type': 'P2PKH'
                }
        except Exception:
            pass  # Invalid Base58 → fall through

    # === P2SH (starts with '3') ===
    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:  # version byte for mainnet P2SH
                hash160 = dec[1:21]
                return b'\xa9\x14' + hash160 + b'\x87', {
                    'input_vb': 91,
                    'output_vb': 32,
                    'type': 'P2SH'
                }
        except Exception:
            pass

    # === Bech32 / Bech32m (bc1q... or bc1p...) ===
    if addr.startswith('bc1'):
        data_part = addr[4:]

        # Early validation: all characters must be in CHARSET
        if any(c not in CHARSET for c in data_part):
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        # Convert to 5-bit data array (no filtering needed — already validated)
        data = [CHARSET.find(c) for c in data_part]

        # Witness version is data[0]
        if len(data) < 1:
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        witness_version = data[0]

        # === P2WPKH / P2WSH (Bech32, witness v0) ===
        if addr.startswith('bc1q') and witness_version == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog:
                if len(prog) == 20:
                    return b'\x00\x14' + bytes(prog), {
                        'input_vb': 68,
                        'output_vb': 31,
                        'type': 'P2WPKH'
                    }
                elif len(prog) == 32:
                    return b'\x00\x20' + bytes(prog), {
                        'input_vb': 69,
                        'output_vb': 43,
                        'type': 'P2WSH'
                    }

        # === Taproot (Bech32m, witness v1) ===
        elif addr.startswith('bc1p') and witness_version == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {
                    'input_vb': 57,
                    'output_vb': 43,
                    'type': 'Taproot'
                }

        # Invalid Bech32/Bech32m (bad checksum, wrong version, etc.)
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

    # === Final fallback for any unsupported or malformed address ===
    return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'fallback'}

DEFAULT_DAO_SCRIPT_PUBKEY, _ = address_to_script_pubkey(DEFAULT_DAO_ADDR)
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
    """Estimate transaction economics — now prioritizes user change output."""
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

    dao_amt = 0
    change_amt = remaining

    if dao_percent > 0 and remaining > 0:
        dao_raw = int(remaining * (dao_percent / 100.0))

        # Special case: 100% donation → user explicitly wants full amount to DAO
        if dao_percent >= 99.9:  # Handles 100.0 and minor float rounding
            dao_amt = dao_raw if dao_raw >= 546 else 0
            if dao_amt == 0 and dao_raw > 0:
                fee += dao_raw
                remaining -= dao_raw
        else:
            # Normal safe mode: only allow DAO if room for BOTH non-dust outputs
            if remaining >= 1092 and dao_raw >= 546 and (remaining - dao_raw) >= 546:
                dao_amt = dao_raw
            else:
                fee += dao_raw
                remaining -= dao_raw

        remaining -= dao_amt

    # Final change dust handling
    change_amt = remaining if remaining >= 546 else 0
    if remaining > 0 and change_amt == 0:
        fee += remaining
        remaining = 0
        change_amt = 0

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
    sequence: int = 0xfffffffd   # Enable RBF

    def serialize(self) -> bytes:
        return (
            self.prev_tx[::-1] +
            self.prev_index.to_bytes(4, 'little') +
            b'\x00' +  # empty scriptSig length — critical for unsigned tx
            self.sequence.to_bytes(4, 'little')
        )

@dataclass
class TxOut:
    amount: int
    script_pubkey: bytes

    def serialize(self) -> bytes:
        return (
            self.amount.to_bytes(8, 'little') +
            encode_varint(len(self.script_pubkey)) +
            self.script_pubkey
        )

@dataclass
class Tx:
    version: int = 2
    tx_ins: List[TxIn] = field(default_factory=list)
    tx_outs: List[TxOut] = field(default_factory=list)
    locktime: int = 0

    def serialize_unsigned(self) -> bytes:
        """Serialize in legacy format (no witness marker, no scriptSig/witness) for PSBT"""
        return (
            self.version.to_bytes(4, 'little') +
            encode_varint(len(self.tx_ins)) +
            b''.join(tx_in.serialize() for tx_in in self.tx_ins) +
            encode_varint(len(self.tx_outs)) +
            b''.join(tx_out.serialize() for tx_out in self.tx_outs) +
            self.locktime.to_bytes(4, 'little')
        )


def create_psbt(tx: Tx, include_witness_utxo: bool = False) -> str:
    """
    Create a minimal, standards-compliant PSBT.
    
    Args:
        tx: Unsigned transaction object (Tx).
        include_witness_utxo: If True, attach dummy witness UTXOs for hardware wallets.
        
    Returns:
        Base64-encoded PSBT string.
    """
    raw_tx = tx.serialize_unsigned()

    # Parse input/output counts from raw tx (for reference)
    pos = 4
    input_count, shift = _read_varint(raw_tx, pos)
    pos += shift
    for _ in range(input_count):
        pos += 36  # prev txid + vout
        script_len, shift = _read_varint(raw_tx, pos)
        pos += shift + script_len + 4  # script + sequence
    output_count, _ = _read_varint(raw_tx, pos)

    # PSBT header
    psbt = b'psbt\xff'

    # Global unsigned tx (key 0x01)
    psbt += b'\x01\x00' + encode_varint(len(raw_tx)) + raw_tx + b'\x00'

    # Input maps
    for txin in tx.tx_ins:
        if include_witness_utxo:
            # Placeholder witness UTXO for hardware wallets
            # Real values can be filled later by scanning the blockchain
            dummy_value = (0).to_bytes(8, 'little')
            dummy_script = b''
            # Key type 0x01 = non-witness utxo, 0x02 = witness utxo
            psbt += b'\x02' + encode_varint(len(dummy_value + dummy_script)) + dummy_value + dummy_script + b'\x00'
        else:
            psbt += b'\x00'

    # Output maps
    for _ in tx.tx_outs:
        psbt += b'\x00'

    return base64.b64encode(psbt).decode()



def _read_varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    val = data[pos]
    if val < 0xfd:
        return val, 1
    elif val == 0xfd:
        return int.from_bytes(data[pos+1:pos+3], 'little'), 3
    elif val == 0xfe:
        return int.from_bytes(data[pos+1:pos+5], 'little'), 5
    else:
        return int.from_bytes(data[pos+1:pos+9], 'little'), 9

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


def analyze(
    addr_input,
    strategy,
    dust_threshold,
    dest_addr,
    fee_rate_slider,
    thank_you_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
):
    # === SAFE INPUT CLAMPING ===
    fee_rate        = max(1,   min(300,  _coerce_int(fee_rate_slider, 15)))      # matches slider max
    future_fee_rate = max(5,   min(500,  _coerce_int(future_fee_slider, 60)))   # matches slider max
    dust_threshold  = max(0,   min(10000, _coerce_int(dust_threshold, 546)))
    dao_percent     = max(0.0, min(100.0, float(thank_you_slider or 5.0)))      # direct clamp

    all_enriched = []

    # Capture the original user input — this is the scan source
    scan_source_out = addr_input.strip()

    if offline_mode:
        manual_str = (manual_utxo_input or "").strip()
        if not manual_str:
            return _analyze_empty(scan_source_out)

        for line in manual_str.splitlines():
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

                spk, meta = address_to_script_pubkey(addr)
                input_wu = meta["input_vb"] * 4

                enriched = {
                    "txid": txid,
                    "vout": vout,
                    "value": value,
                    "address": addr,
                    "input_weight": input_wu,
                    "health": "MANUAL",
                    "recommend": "REVIEW",
                    "script_type": "Manual" if addr == "unknown (manual)" else meta["type"],
                    "source": "Manual Offline",
                    "selected": False,
                }
                all_enriched.append(enriched)
            except Exception:
                continue

        if not all_enriched:
            return _analyze_empty(scan_source_out)

    else:  # online mode
        addr_str = (addr_input or "").strip()
        if not addr_str:
            return _analyze_empty(scan_source_out)

        entries = [e.strip() for e in addr_str.splitlines() if e.strip()]

        for entry in entries:
            is_xpub = entry[:4] in ("xpub", "ypub", "zpub", "tpub", "upub", "vpub")
            source_label = f"xpub ({entry[:12]}...{entry[-6:]})" if is_xpub else (entry if len(entry) <= 34 else f"{entry[:31]}...")

            if is_xpub:
                utxos_raw, _ = scan_xpub(entry, dust_threshold)
            else:
                utxos_raw = get_utxos(entry, dust_threshold)

            if not utxos_raw:
                continue

            for u in utxos_raw:
                _, meta = address_to_script_pubkey(u["address"])
                input_wu = meta["input_vb"] * 4
                value = u["value"]

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
                    "selected": False,
                }
                all_enriched.append(enriched)

        if not all_enriched:
            return _analyze_empty(scan_source_out)

    # ===============================
    # CANONICAL ORDERING & PRUNING LOGIC
    # ===============================
    all_enriched.sort(key=lambda u: (u["value"], u["txid"], u["vout"]), reverse=True)

    ratio = {
        "Privacy First — ~30% pruned (lowest CIOH risk)": 0.30,
        "Recommended — ~40% pruned (balanced savings & privacy)": 0.40,
        "More Savings — ~50% pruned (stronger fee reduction)": 0.50,
        "NUCLEAR PRUNE — ~90% pruned (maximum savings, highest CIOH)": 0.90,
    }.get(strategy, 0.40)
    keep_count = max(MIN_KEEP_UTXOS, int(len(all_enriched) * (1 - ratio)))

    enriched_sorted = sorted(all_enriched, key=lambda u: HEALTH_PRIORITY[u["health"]])

    prune_count = len(enriched_sorted) - keep_count
    for idx, u in enumerate(enriched_sorted):
        u["selected"] = idx < prune_count

    df_rows = []
    for u in enriched_sorted:
        health_html = f'<div class="health health-{u["health"].lower()}">{u["health"]}<br><small>{u["recommend"]}</small></div>'
        df_rows.append([
            u["selected"],            # PRUNE
            u.get("source", "Single"),# Source
            u["txid"],                # TXID
            health_html,              # Health
            u["value"],               # Value
            u["address"],             # Address
            u["input_weight"],        # Weight
            u["script_type"],         # Type
            u["vout"],                # vout
        ])

    import copy
    frozen_enriched = tuple(copy.deepcopy(u) for u in enriched_sorted)

	
    return (
        gr.update(value=df_rows),
        frozen_enriched,
        gr.update(visible=True),   # generate_row
        gr.update(visible=True),   # import_file
        scan_source_out,           # pass the original scan source forward
    )
    
def _analyze_empty(scan_source: str = ""):
	"""Common return for all empty/failure states in analyze()."""
	return (
    	gr.update(value=[]),              # df — empty table
        tuple(),                          # enriched_state — empty
        gr.update(visible=False),         # generate_row — hide
        gr.update(visible=False),          # import_file — hide
		scan_source,
    )


def generate_summary_safe(
    df,
    enriched_state,
    fee_rate,
    future_fee_rate,
    dao_percent,
    locked,
    strategy,  # selected strategy string (value)
):
    """Builds the single unified status box after analysis."""
    if locked:
        return (
            "<div style='text-align:center;padding:60px;color:#888;font-size:1.4rem;'>"
            "SELECTION LOCKED — Ready to sign PSBT"
            "</div>",
            gr.update(visible=False)
        )

    total_utxos = len(enriched_state)
    if total_utxos == 0:
        return (
            "<div style='text-align:center;padding:60px;color:#ff9900;font-size:1.4rem;font-weight:700;'>"
            "No UTXOs found<br><br>"
            "Try different addresses, lower dust threshold, or paste manual UTXOs"
            "</div>",
            gr.update(visible=False)
        )

    # Extract strategy label
    strategy_label = (
        strategy.split(" — ")[0]
        if isinstance(strategy, str) and " — " in strategy
        else "Recommended"
    )

    # Resolve selected UTXOs
    df_data = df if isinstance(df, list) else (df.value if hasattr(df, "value") else [])
    selected_utxos = _resolve_selected(df_data, enriched_state)
    pruned_count = len(selected_utxos)

    # Privacy score
    privacy_score = calculate_privacy_score(selected_utxos, total_utxos) if selected_utxos else 100
    score_color = "#0f0" if privacy_score >= 70 else "#ff9900" if privacy_score >= 40 else "#ff3366"

    # Economics
    try:
        econ = estimate_tx_economics(selected_utxos, fee_rate, dao_percent)
    except ValueError:
        econ = None

    if not selected_utxos:
        return (
            "<div style='text-align:center;padding:60px;color:#ff9900;font-size:1.4rem;'>"
            "Select UTXOs in the table to begin"
            "</div>",
            gr.update(visible=False)
        )

    if econ.remaining <= 0:
        return (
            "<div style='text-align:center;padding:40px;background:#330000;border:2px solid #ff3366;border-radius:16px;"
            "box-shadow:0 0 40px rgba(255,51,102,0.5);font-size:1.3rem;color:#ffaa88;'>"
            "<strong style='color:#ff3366;font-size:1.5rem;'>Transaction Invalid</strong><br><br>"
            f"Current fee ({econ.fee:,} sats @ {fee_rate} s/vB) exceeds available balance.<br><br>"
            "<strong>Lower the fee rate</strong> or select more UTXOs."
            "</div>",
            gr.update(visible=False)
        )

    # Pre-prune size estimate
    all_input_weight = sum(u["input_weight"] for u in enriched_state)
    pre_vsize = max(
        (all_input_weight + 172 + total_utxos) // 4 + 10,
        (all_input_weight + 150 + total_utxos * 60) // 4 + 10,
    )

    savings_pct = round(100 * (1 - econ.vsize / pre_vsize), 1) if pre_vsize > econ.vsize else 0
    savings_pct = max(0, min(100, savings_pct))
    savings_color = "#0f0" if savings_pct >= 70 else "#00ff9d" if savings_pct >= 50 else "#ff9900" if savings_pct >= 30 else "#ff3366"
    savings_label = "NUCLEAR" if savings_pct >= 70 else "EXCELLENT" if savings_pct >= 50 else "GOOD" if savings_pct >= 30 else "WEAK"

    # Future savings
    sats_saved = max(0, econ.vsize * (future_fee_rate - fee_rate))
    savings_line = ""
    if sats_saved >= 100_000:
        savings_line = (
            f"<b style='color:#fff;'>Unlocking now saves:</b> "
            f"<span style='color:#0f0;font-size:1.45rem;font-weight:800;letter-spacing:1.8px;"
            f"text-shadow:0 0 12px #0f0, 0 0 30px #0f0;'>"
            f"+{sats_saved:,} sats</span> "
            f"<span style='color:#0f0;font-weight:800;letter-spacing:3.5px;text-transform:uppercase;"
            f"text-shadow:0 0 20px #0f0, 0 0 50px #0f0;'>NUCLEAR MOVE</span>"
            f"<br><span style='color:#00ff88;font-size:0.9rem;'>(at {future_fee_rate} s/vB future rate)</span>"
        )
    elif sats_saved > 0:
        savings_line = (
            f"<b style='color:#fff;'>Pruning now saves:</b> "
            f"<span style='color:#0f0;font-weight:800;'>+{sats_saved:,} sats</span>"
            f"<br><span style='color:#00ff88;font-size:0.9rem;'>(at {future_fee_rate} s/vB future rate)</span>"
        )

    # DAO visual feedback (in current fee line)
    dao_raw = int((econ.total_in - econ.fee) * (dao_percent / 100.0)) if dao_percent > 0 else 0
    dao_line = ""
    if dao_percent > 0:
        if econ.dao_amt >= 546:
            dao_line = (
                f" • <span style='color:#00ff88;font-weight:800;text-shadow:0 0 20px #00ff88;'>"
                f"DAO: {sats_to_btc_str(econ.dao_amt)}</span><br>"
                f"<span style='color:#00ffaa;font-size:0.95rem;font-style:italic;'>"
                f"Thank you. Your support keeps Ωmega Pruner free, sovereign, and evolving. • Ω</span>"
            )
        elif dao_raw > 0:
            dao_line = (
                f" • <span style='color:#ff3366;font-weight:800;text-shadow:0 0 20px #ff3366;'>"
                f"DAO: {sats_to_btc_str(dao_raw)} → absorbed into fee</span><br>"
                f"<span style='color:#ff6688;font-size:0.9rem;font-style:italic;'>(below 546 sat dust threshold)</span>"
            )

    # === CALCULATIONS FOR NEW BLOCKS ===
    total_in = econ.total_in
    current_fee = econ.fee
    remainder_after_fee = total_in - current_fee
    remaining_utxos_after_prune = total_utxos - pruned_count

    # === PRUNING EXPLANATION (always shown) ===
    pruning_explanation_html = f"""
    <!-- === PRUNING SUMMARY === -->
    <div style="margin:32px 0;padding:28px;background:#001a00;border:4px solid #00ff9d;border-radius:18px;
                box-shadow:0 0 60px rgba(0,255,157,0.7), inset 0 0 40px rgba(0,255,157,0.1);
                font-size:1.25rem;line-height:1.9;color:#ccffe6;">
      <strong style="color:#00ff9d;font-size:1.65rem;text-shadow:0 0 20px #00ff9d, 0 0 40px #00ff9d;">
        🧹 WHAT PRUNING ACTUALLY DOES
      </strong><br><br>
      
      Pruning <strong style="color:#aaffff;">removes inefficient UTXOs</strong> (dust, legacy, or heavy) from your address.<br><br>
      
      • You pay a fee now to delete <strong style="color:#00ffff;">{pruned_count}</strong> bad inputs.<br>
      • The <strong style="color:#00ffff;">{remaining_utxos_after_prune}</strong> remaining UTXOs are now easier and cheaper to spend later.<br>
      • <strong style="color:#ffff88;">If no change output is created:</strong> the pruned value is absorbed into fees — but your wallet is <strong style="color:#aaffff;">cleaner forever</strong>.<br><br>
      
      <strong style="color:#00ffaa;">Goal:</strong> Healthier address → lower future fees.<br>
      Pruning is often worth it during low-fee periods, even if you don’t get change back.<br><br>
      
      <small style="color:#88ffcc;font-style:italic;">
        💡 Tip: If your goal is to get change, only prune when total value > ~10–20× the current expected fee.
      </small>
    </div>
    """

    # === DAO/CHANGE INFO (shown only if DAO % > 0, placed BEFORE warning) ===
    dao_info_html = ""
    if dao_percent > 0:
        intended_dao = int(remainder_after_fee * (dao_percent / 100.0))

        full_donation_note = ""
        if dao_percent >= 99.9:
            full_donation_note = f"<br><strong style='color:#ffcc00;font-size:1.2rem;text-shadow:0 0 15px #ffaa00, 0 0 30px #ff8800;'>☢️ 100% mode active → no change output (full donation to DAO)</strong>"

        change_note = ""
        if econ.change_amt == 0 and remainder_after_fee > 0:
            change_note = " <span style='color:#ff8888;font-style:italic;'>(likely absorbed — see warning below)</span>"

        dao_info_html = f"""
        <!-- === DAO/CHANGE INFO === -->
        <div style="margin:18px 0;padding:18px;background:#112200;border:3px solid #ffaa00;border-radius:14px;
                    box-shadow:0 0 40px rgba(255,170,0,0.6);font-size:1.12rem;line-height:1.7;color:#ffffee;">
          🔹 DAO Donation set to {dao_percent}% (~{sats_to_btc_str(intended_dao)} intended){full_donation_note}<br><br>
          
          • Final DAO output: <strong style="color:#ffffff;text-shadow:0 0 10px #aaffaa;">{sats_to_btc_str(econ.dao_amt)}</strong><br>
          • Expected change back to you: <strong style="color:#ffffff;text-shadow:0 0 10px #aaffff;">{sats_to_btc_str(econ.change_amt)}</strong>{change_note}<br><br>
          
          <small style="color:#ddddaa;">
            Values depend on current fee rate and dust rules. Tiny amounts may be absorbed into fees.
          </small>
        </div>
        """

    # === SMALL PRUNE WARNING (shown only when needed) ===
    small_prune_warning_html = ""
    if remainder_after_fee < 15000:
        if remainder_after_fee < 8000:
            warning_title = "⚠️ Warning: No change output expected"
            warning_color = "#ff3366"
            warning_bg = "#330000"
            warning_border = "#ff3366"
        else:
            warning_title = "⚠️ Caution: Change output may be absorbed"
            warning_color = "#ff8800"
            warning_bg = "#331100"
            warning_border = "#ff8800"

        # Dynamic recommendation — updated to 5× for consistency with HTML
        min_recommended = max(5 * current_fee, 100000)

        small_prune_warning_html = f"""
        <!-- === SMALL PRUNE WARNING === -->
        <div style="margin:26px 0;padding:24px;background:{warning_bg};border:4px solid {warning_border};border-radius:16px;
                    box-shadow:0 0 50px rgba(255,100,100,0.8);font-size:1.28rem;line-height:1.8;color:#ffeeee;">

          <div style="color:{warning_color};font-size:1.55rem;text-shadow:0 0 15px {warning_color}, 0 0 30px {warning_color};margin-bottom:20px;">
            {warning_title}
          </div>
          
          Post-fee remainder (~{remainder_after_fee:,} sats) is small.<br>
          The pruned value will likely be fully or partially absorbed into miner fees.<br><br>
          
          <div style="color:#ffff88;font-size:1.35rem;text-shadow:0 0 12px #ffff99, 0 0 25px #ffaa00;line-height:1.8;">
            Only proceed if your goal is wallet cleanup
          </div>
          <div style="color:#ffdd88;font-size:1.1rem;margin-top:8px;">
            — not expecting significant change back.
          </div><br><br>
          
          <div style="color:#ffaaaa;font-size:1.05rem;line-height:1.6;">
            💡 For a <span style="color:#ffffff;font-size:1.1rem;">reliable change output</span>, aim for:<br>
            • Value Pruned > ~<span style="color:#ffffff;">5× Current Fee</span> (good change back)<br>
            • Value Pruned > ~<span style="color:#ffffff;">10× Current Fee</span> (very comfortable)<br><br>
            Current fee: {current_fee:,} sats → recommended: <span style="color:#ffffff;">{min_recommended:,}+ sats</span>
          </div><br>
          
          <small style="color:#88ffcc;">
            💡 Pro tip: The bigger the prune (relative to fee), the more you get back as change. Small prunes = cleanup only.
          </small>
        </div>
        """

    # CIOH warning
    distinct_addrs = len({u["address"] for u in selected_utxos})
    cioh_warning = get_cioh_warning(len(selected_utxos), distinct_addrs, privacy_score)

    # Final status box
    status_box_html = f"""
    <div style="text-align:center;margin:40px auto 30px auto;padding:28px;background:#000;
                border:3px solid #f7931a;border-radius:20px;max-width:960px;
                box-shadow:0 0 80px rgba(247,147,26,0.5);">

      <div style="color:#0f0;font-size:2.6rem;font-weight:900;
                  letter-spacing:3px;text-shadow:0 0 35px #0f0, 0 0 70px #0f0;margin-bottom:24px;">
        SELECTION READY
      </div>

      <div style="color:#f7931a;font-size:1.8rem;font-weight:800;margin:20px 0;">
        {total_utxos:,} UTXOs • <span style="color:#00ff9d;">{strategy_label}</span> Strategy Active
      </div>

      <div style="color:#fff;font-size:1.5rem;font-weight:700;margin:16px 0;">
        Pruning <span style="color:#ff6600;font-weight:900;">{pruned_count:,}</span> inputs
      </div>

      <div style="color:#fff;font-size:1.7rem;font-weight:800;margin:24px 0;">
        Privacy Score: 
        <span style="color:{score_color};font-size:2.3rem;margin-left:12px;
                     text-shadow:0 0 25px {score_color};">
          {privacy_score}/100
        </span>
      </div>

      <hr style="border:none;border-top:1px solid rgba(247,147,26,0.3);margin:32px 0;">

      <div style="font-size:1.1rem;line-height:2.1;">
        <div style="margin:12px 0;">
          <b style="color:#fff;">Value Pruned:</b> 
          <span style="color:#0f0;font-weight:800;">{sats_to_btc_str(econ.total_in)}</span>
        </div>
        <div style="margin:12px 0;">
          <b style="color:#fff;">Pre-Prune Size (if sent today):</b> 
          <span style="color:#ff9900;font-weight:800;">{pre_vsize:,} vB</span>
        </div>
        <div style="margin:12px 0;">
          <b style="color:#fff;">Post-Prune Size:</b> 
          <span style="color:#0f0;font-weight:800;">{econ.vsize:,} vB</span>
          <span style="color:{savings_color};font-weight:900;margin-left:12px;">
            {savings_label} (-{savings_pct}%)
          </span>
        </div>
        <div style="margin:16px 0;">
          {savings_line}
        </div>
        <div style="margin:12px 0;">
          <b style="color:#fff;">Current Fee:</b> 
          <span style="color:#0f0;font-weight:800;">{econ.fee:,} sats @ {fee_rate} s/vB</span>{dao_line}
        </div>
      </div>

      <hr style="border:none;border-top:1px solid rgba(247,147,26,0.3);margin:32px 0;">

      <div style="margin:32px 0 40px 0;line-height:1.7;">
        {cioh_warning}
      </div>

      <hr style="border:none;border-top:1px solid rgba(247,147,26,0.3);margin:32px 0;">

      {pruning_explanation_html}

      {dao_info_html}

      {small_prune_warning_html}

       <hr style="border:none;border-top:1px solid rgba(247,147,26,0.3);margin:40px 0 60px 0;">

      <div style="color:#aaffaa;font-size:1.15rem;line-height:1.8;padding-bottom:40px;">
        <span style="color:#00ff9d;font-weight:900;">Full coin control:</span> Review table below<br>
        Check/uncheck UTXOs • Adjust as needed
        <br><br>
        <span style="color:#00ffff;font-weight:700;">
          Previous selection? Upload .json to restore
        </span>
      </div>

      <!-- SEPARATOR LINE BEFORE FINAL CALL TO ACTION -->
      <hr style="border:none;border-top:2px solid #f7931a;margin:40px auto 80px auto;width:60%;
                  box-shadow:0 0 15px rgba(247,147,26,0.8);">

      <!-- FINAL CALL TO ACTION -->
      <div style="margin:40px 0 60px 0;">
        <div style="color:#0f0;font-size:2.8rem;font-weight:900;
                    letter-spacing:4px;text-shadow:0 0 40px #0f0, 0 0 80px #0f0;">
          READY TO PRUNE?
        </div>

        <div style="margin-top:50px;font-size:5.5rem;color:#f7931a;opacity:0.95;
                    animation:pulse 2.2s infinite ease-in-out;text-align:center;">
          ↓
        </div>
      </div>

    </div>
    """

    return status_box_html, gr.update(visible=pruned_count > 0)
  
def on_generate(
    dest_addr: str,
    fee_rate: int,
    future_fee_rate: int,
    dao_percent: float,
    enriched_state: tuple,
    scan_source: str,
):
    """
    Freeze user intent into an immutable PSBT snapshot.
    No destination resolution or PSBT construction happens here.
    Called when user clicks "GENERATE NUCLEAR PSBT" after final review.
    """

    # 1. No analysis performed yet
    if not enriched_state:
        return None, gr.update(value=False), None

    # 2. Resolve selected UTXOs — this is the canonical user intent
    selected_utxos = [u for u in enriched_state if u.get("selected", False)]

    # 3. Must have at least one input to prune/consolidate
    if not selected_utxos:
        log.info("Generate attempted with no UTXOs selected")
        return None, gr.update(value=False), None

    try:
        # Freeze everything needed for full deterministic replay
        snapshot = build_psbt_snapshot(
            enriched_state=selected_utxos,           # only pruned inputs
            scan_source=scan_source.strip(),         # frozen fallback destination
            dest_addr_override=dest_addr.strip() or None,
            fee_rate=fee_rate,
            future_fee_rate=future_fee_rate,
            dao_percent=dao_percent,
        )

        # Export as human-readable, offline-safe JSON
        json_str = json.dumps(snapshot, indent=2, ensure_ascii=False)

        date_str = datetime.now().strftime("%Y%m%d")
        fingerprint_short = snapshot["fingerprint_short"]

        filename_prefix = f"omega_selection_{date_str}_{fingerprint_short[:8]}_"

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=filename_prefix,
            delete=False,
        )
        tmp_file.write(json_str)
        tmp_file.close()

        # Success: lock UI and provide immutable export
        return snapshot, gr.update(value=True), tmp_file.name

    except Exception as e:
        log.warning(f"Failed to generate snapshot: {e}")
        return None, gr.update(value=False), None

def generate_psbt(psbt_snapshot: dict):
    """Generate PSBT from frozen snapshot — no live reads."""
    pruned_utxos = psbt_snapshot["inputs"]
    scan_source = psbt_snapshot["scan_source"]
    dest_addr_override = psbt_snapshot.get("dest_addr_override")
    
    if not psbt_snapshot:
        return (
            "<div style='color:#ff3366;text-align:center;padding:30px;'>"
            "No snapshot — run Generate first."
            "</div>"
        )

    pruned_count = len(pruned_utxos)
    fingerprint = psbt_snapshot["fingerprint_short"]
    fee_rate = psbt_snapshot["fee_rate"]
    dao_percent = psbt_snapshot["dao_percent"]

    if not pruned_utxos:
        return (
            "<div style='color:#ff3366;text-align:center;padding:30px;'>"
            "No UTXOs selected for pruning!"
            "</div>"
        )
    
    # Economics from snapshot values
    try:
        econ = estimate_tx_economics(pruned_utxos, fee_rate, dao_percent)
    except ValueError:
        return (
            "<div style='color:#ff3366;text-align:center;padding:30px;'>"
            "Invalid snapshot economics."
            "</div>"
        )

    dao_spk = DEFAULT_DAO_SCRIPT_PUBKEY

    # Resolve destination — explicit override or original scan source
    if dest_addr_override:
        final_dest = dest_addr_override
    else:
        final_dest = scan_source.strip() if scan_source else ""

    # Early error if no destination at all
    if not final_dest:
        return (
            "<div style='color:#ff3366;text-align:center;padding:40px;background:#440000;border-radius:16px;"
            "box-shadow:0 0 40px rgba(255,51,102,0.4);font-size:1.3rem;line-height:1.7;'>"
            "<strong>No destination address available.</strong><br><br>"
            "Please enter a destination address or ensure your scan source is valid."
            "</div>"
        )

    # Guardrail for xpub (future-proof)
    if final_dest.startswith(("xpub", "ypub", "zpub", "tpub", "upub", "vpub")):
        return (
            "<div style='color:#ffcc00;text-align:center;padding:40px;background:#332200;border-radius:16px;"
            "box-shadow:0 0 40px rgba(255,204,0,0.4);font-size:1.3rem;line-height:1.7;'>"
            "<strong>xpub detected as scan source.</strong><br><br>"
            "Please specify a destination address.<br>"
            "Automatic derivation coming soon."
            "</div>"
        )

    # Validate address
    try:
        dest_spk, _ = address_to_script_pubkey(final_dest)
    except Exception:
        return (
            "<div style='color:#ff3366;text-align:center;padding:40px;background:#440000;border-radius:16px;"
            "box-shadow:0 0 40px rgba(255,51,102,0.4);font-size:1.3rem;line-height:1.7;'>"
            "<strong>Invalid destination address.</strong><br><br>"
            "Please check the address format."
            "</div>"
        )

    # Build transaction
    tx = Tx()
    for u in pruned_utxos:
        tx.tx_ins.append(TxIn(bytes.fromhex(u["txid"]), int(u["vout"])))

    # Add outputs — dust-safe (already enforced in economics)
    if econ.dao_amt > 0:
        tx.tx_outs.append(TxOut(econ.dao_amt, dao_spk))
    if econ.change_amt > 0:
        tx.tx_outs.append(TxOut(econ.change_amt, dest_spk))

    # Legacy serialization
    raw_tx = (
        tx.version.to_bytes(4, 'little') +
        encode_varint(len(tx.tx_ins)) +
        b''.join(i.serialize() for i in tx.tx_ins) +
        encode_varint(len(tx.tx_outs)) +
        b''.join(o.serialize() for o in tx.tx_outs) +
        tx.locktime.to_bytes(4, 'little')
    )

    psbt_b64 = create_psbt(tx)
    # QR generation
    error_correction = qrcode.constants.ERROR_CORRECT_L if len(psbt_b64) > 2800 else qrcode.constants.ERROR_CORRECT_M
    qr = qrcode.QRCode(version=None, error_correction=error_correction, box_size=6, border=4)
    qr.add_data(f"bitcoin:?psbt={psbt_b64}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#f7931a", back_color="#000000")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    qr_img_html = f'<img src="{qr_uri}" style="width:100%;height:auto;display:block;image-rendering:crisp-edges;border-radius:12px;" alt="PSBT QR Code"/>'

    qr_warning_html = ""
    if len(psbt_b64) > 2800:
        qr_warning_html = """
        <div style="margin-top:12px;padding:10px 14px;border-radius:8px;background:#331a00;color:#ffb347;
                     font-family:monospace;font-size:0.95rem;line-height:1.4;border:1px solid #ff9900aa;">
            Large PSBT detected.<br>
            If QR fails → tap <strong>COPY PSBT</strong> and paste directly.
        </div>
        """

    psbt_html = f"""
	<div style="height: 60px;"></div>
<div style="text-align:center;margin:60px auto 0px;max-width:960px;">
<div style="display:inline-block;padding:55px;background:#000;border:14px solid #f7931a;border-radius:36px;
            box-shadow:0 0 140px rgba(247,147,26,0.95);
            background:radial-gradient(circle at center,#0a0a0a 0%,#000 100%);">

     <!-- Selection Fingerprint — Provable Intent -->
    <div style="margin:40px 0;padding:24px;background:#001100;border:4px solid #0f0;border-radius:18px;
        box-shadow:0 0 80px rgba(0,255,0,0.8);font-family:monospace;"
        title="This deterministic hash proves your exact input selection. Identical selection = identical hash. Verify against exported JSON.">
        <div style="color:#0f0;font-size:1.4rem;font-weight:900;letter-spacing:3px;
                    text-shadow:0 0 20px #0f0, 0 3px 6px #000, 0 6px 16px #000000dd;
                    margin-bottom:16px;">
            FINGERPRINT
        </div>
        <div style="color:#00ff9d;font-size:2.2rem;font-weight:900;letter-spacing:8px;
                    text-shadow:0 0 30px #00ff9d, 0 0 60px #00ff9d,
                                0 4px 8px #000, 0 8px 20px #000000ee;">
            {fingerprint}
        </div>
        <div style="margin-top:20px;color:#00ffaa;font-size:1.1rem;line-height:1.6;font-weight:800;
                text-shadow:0 0 16px #0f0,
                            0 2px 4px #000, 0 4px 12px #000000cc, 0 8px 20px #000000aa;">
            Cryptographic proof of your pruning selection<br>
            Deterministic • Audit-proof • Never changes
        </div>
    <button onclick="
            navigator.clipboard.writeText('{fingerprint}')
                .then(() => {{ 
                    this.innerText = 'COPIED'; 
                    setTimeout(() => this.innerText = 'COPY FINGERPRINT', 1500); 
                }})
                .catch(() => alert('Copy failed — select manually'));
        "
            style="margin-top:16px;padding:8px 20px;background:#000;color:#0f0;border:2px solid #0f0;border-radius:12px;
                   font-size:1.1rem;font-weight:800;cursor:pointer;box-shadow:0 0 20px #0f0;">
            COPY FINGERPRINT
        </button>
	
    <!-- QR -->
    <div style="margin:40px auto;width:520px;max-width:96vw;padding:20px;background:#000;
                border:8px solid #0f0;border-radius:24px;box-shadow:0 0 60px #0f0,inset 0 0 40px #0f0;">
        {qr_img_html}
    </div>

    {qr_warning_html}

  <!-- PSBT + COPY + HINT -->
    <div style="margin:60px auto 20px;width:92%;max-width:880px;">
        <div style="position:relative;background:#000;border:6px solid #f7931a;border-radius:18px;
                    box-shadow:0 0 40px #0f0;overflow:hidden;">
            <textarea id="psbt-output" readonly 
                style="width:100%;height:180px;background:#000;color:#0f0;font-size:1rem;
                       padding:24px;padding-right:140px;border:none;outline:none;resize:none;
                       font-family:monospace;font-weight:700;box-sizing:border-box;">
{psbt_b64}</textarea>
                       <button onclick="
                navigator.clipboard.writeText(document.getElementById('psbt-output').value)
                    .then(() => {{ 
                        this.innerText = 'COPIED'; 
                        setTimeout(() => this.innerText = 'COPY PSBT', 1500); 
                    }})
                    .catch(() => alert('Copy failed — select and copy manually'));
            "
                style="position:absolute;top:14px;right:14px;padding:12px 30px;background:#f7931a;
                       color:#000;border:none;border-radius:14px;font-weight:800;letter-spacing:1.5px;
                       font-size:1.12rem;text-transform:uppercase;cursor:pointer;
                       box-shadow:0 0 30px #f7931a, inset 0 0 20px rgba(0,0,0,0.4);z-index:10;">
                COPY PSBT
            </button>
        </div>
        <div style="text-align:center;margin-top:12px;">
            <span style="color:#00f0ff;font-weight:700;text-shadow:0 0 10px #0f0;">RBF enabled</span>
            <span style="color:#888;"> • Raw PSBT • </span>
            <span style="color:#666;font-size:0.9rem;">Inspect before signing</span>
        </div>
    </div>
								 
    <!-- Wallet support -->
    <div style='color:#ff9900;font-size:1rem;text-align:center;margin:40px 0 20px;padding:16px;
                background:#220000;border:2px solid #f7931a;border-radius:12px;
                box-shadow:0 0 40px rgba(247,147,26,0.4);'>
        <div style='color:#fff;font-weight:800;text-shadow:0 0 12px #f7931a;'>
            Important: Wallet must support <strong style='color:#0f0;text-shadow:0 0 15px #0f0;'>PSBT</strong>
        </div>
        <div style='color:#0f8;margin-top:8px;opacity:0.9;'>
            Sparrow • BlueWallet • Electrum • UniSat • Nunchuk • OKX
        </div>
    </div>
	</div>
</div>
"""



    return psbt_html

# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks(
    title="Ωmega Pruner v10.7 — FLOW STATE"
) as demo:
    # Social / OpenGraph Preview
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

    # Full-screen animated Ωmega background + Hero Banner
    gr.HTML("""
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

    <div style="text-align:center;margin:100px 0 60px 0;padding:60px 40px;
                background:rgba(0,0,0,0.62);
                backdrop-filter: blur(10px);
                border:8px solid #f7931a;
                border-radius:32px;
                box-shadow:0 0 100px rgba(247,147,26,0.5), inset 0 0 80px rgba(247,147,26,0.1);
                max-width:900px;margin-left:auto;margin-right:auto;
                position:relative;z-index:1;">
        
        <div style="color:#f7931a;font-size:4.8rem;font-weight:900;letter-spacing:12px;
                    text-shadow:0 0 40px #f7931a, 0 0 80px #ffaa00, 0 0 120px rgba(247,147,26,0.9);
                    margin-bottom:30px;">
            ΩMEGA PRUNER
        </div>
        
        <div style="color:#0f0;font-size:2.4rem;font-weight:900;letter-spacing:5px;
                    text-shadow:0 0 30px #0f0, 0 0 60px #0f0;margin:40px 0;">
            NUCLEAR COIN CONTROL
        </div>
        
        <div style="color:#ddd;font-size:1.5rem;line-height:1.8;max-width:760px;margin:0 auto 50px auto;">
            Pruning isn't just about saving sats today — it's about <strong style="color:#0f0;">taking control</strong> of your coins for the long term.<br><br>
            
            By consolidating inefficient UTXOs, you:<br>
            • <strong style="color:#00ff9d;">Save significantly on fees</strong> during high network congestion<br>
            • <strong style="color:#00ff9d;">Gain true coin control</strong> — know exactly what you're spending<br>
            • <strong style="color:#00ff9d;">Improve privacy</strong> when done thoughtfully<br>
            • <strong style="color:#00ff9d;">Future-proof your stack</strong> — remain spendable forever<br><br>

            <strong style="color:#f7931a;font-size:1.8rem;font-weight:900;letter-spacing:1px;">
                Prune now. Win forever.
            </strong><br><br>
            
            Paste addresses or xpubs below and click 
            <strong style="color:#f7931a;font-size:1.7rem;">ANALYZE</strong> 
            to unlock your personalized strategy.
        </div>
        
        <div style="font-size:4rem;color:#f7931a;opacity:0.9;animation:pulse 2s infinite;">
            ↓
        </div>
    </div>

    <style>
    @keyframes pulse {
        0%, 100% { transform: translateY(0); opacity: 0.8; }
        50% { transform: translateY(20px); opacity: 1; }
    }

    @keyframes omega-breath {
        0%, 100% { opacity: 0.76; transform: scale(0.95) rotate(0deg); }
        50% { opacity: 1.0; transform: scale(1.05) rotate(180deg); }
    }

    @keyframes gradient-pulse {
        0%, 100% { transform: scale(0.97); }
        50% { transform: scale(1.03); }
    }

    @keyframes omega-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .gradio-container { 
        position: relative;
        z-index: 0;
        background: transparent;
        overflow-y: auto;
    }

    #omega-bg { 
        isolation: isolate;
        will-change: transform, opacity;
    }

    /* Fee preset glow */
    .fee-btn button:not(:disabled) {
        box-shadow: 0 0 20px rgba(247,147,26,0.6);
        animation: fee-glow 3s infinite alternate;
    }

    .fee-btn button:disabled {
        box-shadow: none;
        animation: none;
        opacity: 0.35;
    }

    @keyframes fee-glow {
        from { box-shadow: 0 0 20px rgba(247,147,26,0.6); }
        to { box-shadow: 0 0 40px rgba(247,147,26,0.9); }
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
        to { box-shadow: 0 0 45px rgba(247,147,26,1); }
    }

    /* Locked badge — more dramatic */
    @keyframes badge-pulse {
        0%   { transform: scale(1);   box-shadow: 0 0 80px rgba(0,255,0,0.8); }
        50%  { transform: scale(1.15); box-shadow: 0 0 160px rgba(0,255,0,1); }
        100% { transform: scale(1);   box-shadow: 0 0 80px rgba(0,255,0,0.8); }
    }

    @keyframes badge-entry {
        0%   { opacity: 0; transform: scale(0.4) translateY(-40px); }
        70%  { transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    .locked-badge {
        position: fixed;
        top: 24px;
        right: 24px;
        z-index: 9999;
        padding: 20px 56px;
        background: #000;
        border: 8px solid #f7931a;
        border-radius: 32px;
        box-shadow: 0 0 160px rgba(247,147,26,1);
        color: #ffaa00;
        text-shadow: 0 0 10px #ffaa00, 0 0 20px #ffaa00, 0 0 40px #ffaa00, 0 0 80px #ffaa00;
        font-weight: 900;
        font-size: 2.8rem;
        letter-spacing: 14px;
        pointer-events: none;
        opacity: 0;
        animation: badge-entry 1s cubic-bezier(0.175, 0.885, 0.32, 1.4) forwards,
                   badge-pulse 2.8s infinite alternate 1s;
    }
    </style>
    """)

    # Health badges + disabled textbox styling
    gr.HTML("""
    <style>
    .health {
        font-weight: 900;
        text-align: center;
        padding: 6px 10px;
        border-radius: 4px;
        min-width: 70px;
        display: inline-block;
    }
    .health-dust     { color: #ff3366; background: rgba(255, 51, 102, 0.12); }
    .health-heavy    { color: #ff6600; background: rgba(255, 102, 0, 0.12); }
    .health-careful  { color: #ff00ff; background: rgba(255, 0, 255, 0.12); }
    .health-medium   { color: #ff9900; background: rgba(255, 153, 0, 0.12); }
    .health-optimal  { color: #00ff9d; background: rgba(0, 255, 157, 0.12); }
    .health-manual   { color: #bb86fc; background: rgba(187, 134, 252, 0.12); }

    .health small {
        display: block;
        color: #aaa;
        font-weight: normal;
        font-size: 0.8em;
        margin-top: 2px;
    }

    .gr-textbox input:disabled {
        background-color: #111 !important;
        color: #555 !important;
        opacity: 0.6;
        cursor: not-allowed;
    }
    </style>
    """)

  # Global CSS for dark mode (pure black)
    gr.HTML("""
<style>
    .dark-mode {
        background: #000 !important;
    }
    .dark-mode .gradio-container,
    .dark-mode .gr-panel,
    .dark-mode .gr-form,
    .dark-mode .gr-box,
    .dark-mode textarea,
    .dark-mode input,
    .dark-mode .gr-button {
        background: #000 !important;
        color: #0f0 !important;
        border-color: #f7931a !important;
    }
    .dark-mode .gr-button:hover {
        background: #f7931a !important;
        color: #000 !important;
    }
        /* Visible, glowing checkbox tick in dark mode */
      /* Nuclear checkbox — visible tick + green fill from the start */
    input[type="checkbox"] {
        width: 28px !important;
        height: 28px !important;
        accent-color: #0f0 !important;
        background: #000 !important;
        border: 3px solid #f7931a !important;
        border-radius: 8px !important;
        cursor: pointer;
        box-shadow: 0 0 20px rgba(247,147,26,0.6) !important;
        appearance: none;                    /* Remove native look */
        position: relative;
    }

    /* Green fill when checked — works immediately */
    input[type="checkbox"]:checked {
        background: #0f0 !important;
        box-shadow: 0 0 30px #0f0 !important;
    }

    /* Custom black checkmark */
    input[type="checkbox"]:checked::after {
        content: '✓';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #000;
        font-size: 20px;
        font-weight: 900;
        pointer-events: none;
    }

    /* Light mode fallback */
    :not(.dark-mode) input[type="checkbox"] {
        accent-color: #f7931a !important;
        background: #fff !important;
        border-color: #0f0 !important;
    }
    :not(.dark-mode) input[type="checkbox"]:checked {
        background: #f7931a !important;
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
    def apply_fee_preset_locked(locked: bool, preset: str):
        if locked:
            return gr.update(), gr.update()

        fees = get_live_fees() or {
            "fastestFee": 10,
            "halfHourFee": 6,
            "hourFee": 3,
            "economyFee": 1,
        }

        rate_map = {
            "fastest": fees.get("fastestFee", 10),
            "half_hour": fees.get("halfHourFee", 6),
            "hour": fees.get("hourFee", 3),
            "economy": fees.get("economyFee", 1),
        }

        new_rate = rate_map.get(preset, 3)

        return gr.update(value=new_rate), gr.update()

    def finalize_generate_ui():
        return (
            gr.update(visible=False),                    # gen_btn
            gr.update(visible=False),                    # generate_row
            gr.update(visible=True),                     # export_title_row
            gr.update(visible=True),                     # export_file_row
            gr.update(visible=False, interactive=False), # import_file
            "<div class='locked-badge'>LOCKED</div>",    # locked_badge
        )


    # =================================================================
    # ========================= UI STARTS HERE ========================
    # =================================================================
    with gr.Column():
        # Live status banner
        mode_status = gr.Markdown(
            value="**🌙 Dark • Online mode** • API calls enabled"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                offline_toggle = gr.Checkbox(
                    label="🔒 Offline / Air-Gapped Mode",
                    value=False,
                    interactive=True,
                    info="No API calls • Paste raw UTXOs below • True cold wallet prep",
                )
            with gr.Column(scale=1, min_width=220):
                theme_toggle = gr.Checkbox(
                    label="🌙 Dark Mode (pure black)",
                    value=True,
                    interactive=True,
                    info="Retinal protection • Nuclear glow preserved • Recommended",
                )

        with gr.Row():
            addr_input = gr.Textbox(
                label="Scan Address / xpub (one per line)",
                placeholder="Paste address(es) or xpub — 100% non-custodial",
                lines=6,
                scale=2,
            )
            dest = gr.Textbox(
                label="Destination (optional)",
                placeholder="Blank = back to scanned address",
                info="Your pruned coins return here. Default = original scan source. Required for xpub scans.",
                scale=1,
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

        # === Seamless mode switching + dark mode + live status ===
        def update_status_and_ui(offline, dark):
            theme_icon = "🌙 Dark" if dark else "☀️ Light"
            mode = "Offline 🔒" if offline else "Online"
            api_note = "No API calls • Fully air-gapped" if offline else "API calls enabled"
            return f"**{theme_icon} • {mode}** • {api_note}"

        offline_toggle.change(
            fn=lambda x: gr.update(visible=x),
            inputs=offline_toggle,
            outputs=manual_box_row,
        ).then(
            fn=lambda x: gr.update(value="") if x else gr.update(),
            inputs=offline_toggle,
            outputs=addr_input,
        ).then(
            fn=lambda x: gr.update(
                interactive=not x,
                placeholder="🔒 Offline mode active — paste raw UTXOs in the box below 👇" if x
                else "Paste one or many addresses/xpubs (one per line)\nClick ANALYZE when ready"
            ),
            inputs=offline_toggle,
            outputs=addr_input,
        ).then(
            fn=update_status_and_ui,
            inputs=[offline_toggle, theme_toggle],
            outputs=mode_status
        )

        theme_toggle.change(
            fn=update_status_and_ui,
            inputs=[offline_toggle, theme_toggle],
            outputs=mode_status
        )

        # Strategy dropdown + Dust
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
            dust = gr.Slider(0, 5000, 546, step=1, label="Dust Threshold (sats)")

        # Fee sliders
        with gr.Row():
            fee_rate_slider = gr.Slider(
                1, 300, 15, step=1, label="Fee Rate now (sat/vB)", scale=3,
            )
            future_fee_slider = gr.Slider(
                5, 500, value=60, step=1, label="Future fee rate in 3–6 months (sat/vB)", scale=3,
            )
            thank_you_slider = gr.Slider(
                0, 100, value=5.0, step=0.1, label="Thank You / DAO Donation (%)", info="Applied to amount remaining after miner fee. 100% = full donation (no change output). Below ~546 sats will be absorbed into fee.", scale=2,
            )

        # Fee preset buttons
        with gr.Row():
            economy_btn = gr.Button("Economy", size="sm", elem_classes="fee-btn")
            hour_btn = gr.Button("1 hour", size="sm", elem_classes="fee-btn")
            halfhour_btn = gr.Button("30 min", size="sm", elem_classes="fee-btn")
            fastest_btn = gr.Button("Fastest", size="sm", elem_classes="fee-btn")
       
        

        analyze_btn = gr.Button("1. ANALYZE & LOAD UTXOs", variant="primary")
        status_output = gr.HTML("")

        # States (invisible)
        scan_source = gr.State("")
        enriched_state = gr.State([])
        locked = gr.State(False)
        psbt_snapshot = gr.State(None)
        locked_badge = gr.HTML("")  # Starts hidden
        selection_snapshot_state = gr.State({})

        # Generate row — hidden until analysis complete
        with gr.Row(visible=False) as generate_row:
            gen_btn = gr.Button(
                "2. GENERATE NUCLEAR PSBT",
                variant="primary",
                elem_id="generate-btn"
                # No visible=False here — the whole row controls it
            )

        # PSBT output — placed right below the generate row
        psbt_output = gr.HTML("")

        # Import file last
        import_file = gr.File(
            label="Restore Previous Ωmega Selection: Upload your saved .json file",
            file_types=[".json"],
            type="filepath",
            visible=False,
        )


        # Export sections
        with gr.Row(visible=False) as export_title_row:
            gr.HTML("""
            <div style='text-align:center;padding:40px 0 30px 0;'>
                <!-- Main Header — FROZEN = icy blue theme -->
                <div style='color:#00ddff;font-size:2.6rem;font-weight:900;
                            letter-spacing:8px;
                            text-shadow:0 0 40px #00ddff, 0 0 80px #00ddff,
                                        0 4px 8px #000, 0 8px 20px #000000ee,
                                        0 12px 32px #000000cc;
                            margin-bottom:20px;'>
                    🔒 SELECTION FROZEN
                </div>
                
                <!-- Core message — back to signature green -->
                <div style='color:#aaffaa;font-size:1.4rem;font-weight:700;
                            text-shadow:0 0 20px #0f0,
                                        0 3px 6px #000, 0 6px 16px #000000dd,
                                        0 10px 24px #000000bb;
                            max-width:720px;margin:0 auto 16px auto;
                            line-height:1.6;'>
                    Your pruning intent is now immutable • Permanent audit trail secured
                </div>
                
                <!-- Extra reassurance — bright cyan for clarity -->
                <div style='color:#00ffdd;font-size:1.1rem;opacity:0.9;font-weight:700;
                            text-shadow:0 2px 4px #000, 0 4px 12px #000000cc,
                                        0 8px 20px #000000aa;
                            max-width:640px;margin:0 auto;'>
                    The file below includes:<br>
                    • Full selection fingerprint • All selected UTXOs • Transaction parameters<br><br>
                    Download for backup, offline verification, or future reference
                </div>
            </div>
            """)

        with gr.Row(visible=False) as export_file_row:
            export_file = gr.File(
                label="",
                interactive=False
            )


        df = gr.DataFrame(
            headers=[
                "PRUNE",
                "Source",
                "TXID",
                "Health",
                "Value (sats)",
                "Address",
                "Weight (wu)",
                "Type",
                "vout",
            ],
            datatype=["bool", "str", "str", "html", "number", "str", "number", "str", "number"],
            type="array",
            interactive=True,
            wrap=True,
            row_count=(5, "dynamic"),
            max_height=500,
            max_chars=None,
            label="CHECK TO PRUNE • Pre-checked = recommended • OPTIMAL = ideal • DUST/HEAVY = prune",
            static_columns=[1, 2, 3, 4, 5, 6, 7],
            column_widths=["90px", "160px", "200px", "120px", "140px", "160px", "130px", "90px", "80px"]
        )

        with gr.Column():
            reset_btn = gr.Button("NUCLEAR RESET · START OVER", variant="secondary")
            gr.HTML("""
            <div style="text-align:center;margin-top:8px;">
                <small style="color:#888;font-style:italic;">
                    Clears everything • No funds affected • Safe to use anytime
                </small>
            </div>
            """)
  # =============================
    # — FEE PRESET BUTTONS (pure parameter change) —
    # =============================
    for btn, preset in [
        (economy_btn, "economy"),
        (hour_btn, "hour"),
        (halfhour_btn, "half_hour"),
        (fastest_btn, "fastest"),
    ]:
        btn.click(
            fn=partial(apply_fee_preset_locked, preset=preset),
            inputs=[locked],
            outputs=[fee_rate_slider],  # ← current fee rate slider
        )

    # =============================
    # — Import File (pure state mutation) —
    # =============================
    import_file.change(
        fn=load_selection,
        inputs=[import_file, enriched_state],
        outputs=[enriched_state],
    ).then(
        fn=rebuild_df_rows,
        inputs=[enriched_state],
        outputs=[df]
    ).then(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row]
    )

    # =============================
    # — ANALYZE BUTTON (pure data loading + affordances) —
    # =============================
    analyze_btn.click(
        fn=analyze,
        inputs=[
            addr_input,
            strategy,
            dust,
            dest,
            fee_rate_slider,
            thank_you_slider,
            future_fee_slider,
            offline_toggle,
            manual_utxo_input,
        ],
        outputs=[
            df,
            enriched_state,
            generate_row,
            import_file,
			scan_source,
        ],
    ).then(lambda: gr.update(visible=False), outputs=analyze_btn)

    # =============================
    # — GENERATE BUTTON (pure execution + PSBT render) —
    # =============================
    gen_btn.click(
        fn=on_generate,
        inputs=[dest, fee_rate_slider, future_fee_slider, thank_you_slider, enriched_state, scan_source],
        outputs=[psbt_snapshot, locked, export_file],
    ).then(
        fn=generate_psbt,
        inputs=[psbt_snapshot],
        outputs=[psbt_output],
    ).then(
        fn=finalize_generate_ui,
        outputs=[
            gen_btn,
            generate_row,
            export_title_row,
            export_file_row,
            import_file,
            locked_badge,
        ],
    ).then(
        lambda: gr.update(interactive=False),
        outputs=df,
    ).then(
        lambda: True,
        outputs=locked,
    )
    # =============================
    # — NUCLEAR RESET BUTTON —
    # =============================
    def nuclear_reset():
        """NUCLEAR RESET — silent wipe of state and affordances."""
        return (
            gr.update(value=[]),                                     # df — clear table
            tuple(),                                                 # enriched_state — empty
            gr.update(visible=True),                                 # analyze_btn — show
            gr.update(visible=False),                                # generate_row — hide
            None,                                                    # psbt_snapshot — wipe
            False,                                                   # locked — unlock
            "",                                                      # locked_badge — clear
            gr.update(value="", interactive=True),                   # addr_input
			gr.update(value="", interactive=True),				     # dest — clear destination
            gr.update(interactive=True),                             # strategy
            gr.update(interactive=True),                             # dust
            gr.update(interactive=True),                             # dest
            gr.update(interactive=True),                             # fee_rate_slider
            gr.update(interactive=True),                             # future_fee_slider
            gr.update(interactive=True),                             # thank_you_slider
            gr.update(value=False, interactive=True),                # offline_toggle
            gr.update(value="", visible=False, interactive=True),    # manual_utxo_input
            gr.update(interactive=True),                             # fastest_btn
            gr.update(interactive=True),                             # halfhour_btn
            gr.update(interactive=True),                             # hour_btn
            gr.update(interactive=True),                             # economy_btn
            gr.update(visible=False),                                # export_title_row
            gr.update(visible=False),                                # export_file_row
            None,                                                    # export_file
            gr.update(value=None, visible=False, interactive=True),  # import_file
            "",                                                      # psbt_output — clear PSBT
        )

    reset_btn.click(
        fn=nuclear_reset,
        inputs=None,
        outputs=[
            df,
            enriched_state,
            analyze_btn,
            generate_row,
            psbt_snapshot,
            locked,
            locked_badge,
            addr_input,
			dest,
            strategy,
            dust,
            dest,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            offline_toggle,
            manual_utxo_input,
            fastest_btn,
            halfhour_btn,
            hour_btn,
            economy_btn,
            export_title_row,
            export_file_row,
            export_file,
            import_file,
            psbt_output,
        ],
    ).then(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row]
    )

    # =============================
    # — LIVE INTERPRETATION (single source of truth) —
    # =============================
    df.change(
        fn=update_enriched_from_df,
        inputs=[df, enriched_state, locked],
        outputs=enriched_state,
    ).then(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row],
    )

    fee_rate_slider.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row],
    )

    future_fee_slider.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row],
    )

    thank_you_slider.change(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row],
    )

    demo.load(
        fn=generate_summary_safe,
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy],
        outputs=[status_output, generate_row]
    )

    # Re-analyze on strategy/dust change (hard reset)
    strategy.change(
        fn=analyze,
        inputs=[addr_input, strategy, dust, dest, fee_rate_slider, thank_you_slider, future_fee_slider, offline_toggle, manual_utxo_input],
        outputs=[df, enriched_state, generate_row, import_file, scan_source],
    )

    dust.change(
        fn=analyze,
        inputs=[addr_input, strategy, dust, dest, fee_rate_slider, thank_you_slider, future_fee_slider, offline_toggle, manual_utxo_input],
        outputs=[df, enriched_state, generate_row, import_file, scan_source],
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
            Ωmega Pruner v10.7 — Flow State
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
        <span style="color: #00ff9d; font-weight: 900; text-shadow: 0 0 20px #00ff9d, 0 0 40px #00ff9d;">
            This build is engineered for speed and clarity.
        </span><br>
        <span style="color: #00ff88; font-weight: 900; text-shadow: 0 0 18px #00ff88, 0 0 36px #00ff88;">
            For extended capabilities or tailored integrations, custom versions can be commissioned.
        </span>
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

    # === DARK MODE PERSISTENCE JS — MUST BE INSIDE BLOCKS ===
    demo.load(
        fn=None,
        js="""
    () => {
        const key = 'omega_dark_mode';
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        const themeCheckbox = checkboxes.length >= 2 ? checkboxes[1] : checkboxes[0];
        
        // Force dark mode on initial load
        document.body.classList.add('dark-mode');
        localStorage.setItem(key, 'true');
        
        const saved = localStorage.getItem(key);
        let isDark = saved === 'true';
        if (saved === null) isDark = true;
        
        themeCheckbox.checked = isDark;
        
        themeCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem(key, 'true');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem(key, 'false');
            }
        });
    }
    """
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False, debug=False)
