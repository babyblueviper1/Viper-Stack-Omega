"""
CANONICAL STATE MODEL (AUTHORITATIVE — DO NOT VIOLATE)

User Inputs (mutable via UI):
- addr_input
- strategy (pruning strategy dropdown)
- dust_threshold
- dest_addr
- fee_rate_slider
- dao_slider (thank_you_slider)
- future_fee_slider
- offline_mode
- manual_utxo_input


Derived State (write-once per analyze()):
- enriched_state: tuple(meta: dict, utxos: tuple[dict]) 
  → Full UTXO set with health, weights, script_type, source, and initial 'selected' flags
  → ONLY written by analyze()
  → NEVER mutated after creation (new tuples returned on selection changes)
  → Single source of truth for all downstream logic
  → Format: (metadata_dict, frozen_utxos_tuple)

# NOTE:
# enriched_state is now a frozen tuple: (meta, utxos)
# Post-analyze functions may only update the 'selected' field in UTXOs,
# returning a new immutable tuple with same meta and updated utxos.

Selection (user intent):
- df_rows: checkbox state from Dataframe
  → Resolved via _resolve_selected(df_rows, enriched_state[1])  # uses utxos only

Phase (derived — do not store):
- "init": no enriched_state
- "analyzed": enriched_state present, locked = False
- "locked": locked = True (after successful generate_psbt)

RULES:
1. Only analyze() may assign enriched_state
2. No function may mutate enriched_state contents (utxos are immutable after analyze)
3. Strategy changes require re-running analyze()
4. After lock, no economic recomputation — use frozen values only
5. All summary/economics must derive from enriched_state[1] (utxos) + current sliders (pre-lock) or frozen snapshot (post-lock)

Violations will be treated as bugs.
"""

# Omega_Pruner.py
import gradio as gr
import requests, time, base64, io, qrcode, json, os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings, logging
from functools import partial
import threading
import hashlib
import tempfile
from datetime import datetime
import copy
import urllib.parse
import pandas as pd

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

no_utxos_msg = (
    "<div style='text-align:center;padding:60px;color:#ff9900;"
    "font-size:1.4rem;font-weight:700;'>"
    "No UTXOs found<br><br>"
    "Try different addresses, lower dust threshold, or paste manual UTXOs"
    "</div>"
)

select_msg = (
    "<div style='text-align:center;padding:60px;color:#ff9900;"
    "font-size:1.4rem;'>"
    "Select UTXOs in the table to begin"
    "</div>"
)


CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

PRUNING_RATIOS = {
    "Privacy First — ~30% pruned (lowest CIOH risk)": 0.30,
    "Recommended — ~40% pruned (balanced savings & privacy)": 0.40,
    "More Savings — ~50% pruned (stronger fee reduction)": 0.50,
    "NUCLEAR PRUNE — ~90% pruned (maximum savings, highest CIOH)": 0.90,
}

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
    """Resolve selected UTXOs via row-order matching (Gradio-safe)."""

    if not df_rows or not enriched_state:
        return []

    # Extract utxos from frozen state
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        _, utxos = enriched_state
    else:
        utxos = enriched_state or []

    if len(df_rows) != len(utxos):
        print(f"Warning: row mismatch df={len(df_rows)} utxos={len(utxos)}")
        return []

    selected = []

    for idx, row in enumerate(df_rows):
        if not row or len(row) <= CHECKBOX_COL:
            continue

        checkbox_val = row[CHECKBOX_COL]

        checked = checkbox_val in (True, 1, "true", "True", "1") or bool(checkbox_val)
        if not checked:
            continue

        selected.append(utxos[idx])

    print(f">>> _resolve_selected: {len(selected)} UTXOs detected via row index")
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

def update_enriched_from_df(df_rows: List[list], enriched_state: tuple, locked: bool) -> tuple:
    if locked or not enriched_state:
        return enriched_state

    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        meta, utxos = enriched_state
    else:
        meta = {}
        utxos = enriched_state or ()

    if len(df_rows) != len(utxos):
        return enriched_state

    updated_utxos = []
    for row, u in zip(df_rows, utxos):
        new_u = dict(u)
        script_type = u.get("script_type", "")

        # THIS IS THE UNBREAKABLE DEFENSE
        if script_type not in ("P2WPKH", "Taproot"):
            new_u["selected"] = False  # ALWAYS force off for legacy/nested
        else:
            new_u["selected"] = bool(row[CHECKBOX_COL])

        updated_utxos.append(new_u)

    return (meta, tuple(updated_utxos))
        
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

def rebuild_df_rows(enriched_state) -> tuple[List[List], bool]:
    """
    Rebuild dataframe rows from current enriched_state.
    Used when loading a saved selection JSON.
    Safely handles invalid input and enforces unsupported type rules.
    """
    # Handle invalid enriched_state
    if not enriched_state:
        return [], False

    # Extract utxos from frozen tuple
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        _, utxos = enriched_state
        if not isinstance(utxos, (list, tuple)):
            return [], False
        state_list = list(utxos)
    elif isinstance(enriched_state, (list, tuple)):
        state_list = list(enriched_state)
    else:
        print(f"Warning: rebuild_df_rows received invalid enriched_state type: {type(enriched_state)}")
        return [], False

    rows = []
    has_unsupported = False

    for u in state_list:
        if not isinstance(u, dict):
            continue

        script_type = u.get("script_type", "")
        selected = u.get("selected", False)

        # Supported only: Native SegWit and Taproot
        supported_in_psbt = script_type in ("P2WPKH", "Taproot")

        if not supported_in_psbt:
            has_unsupported = True
            selected = False  # Force unselected

            # Unified legacy detection (catches both "P2PKH" and "Legacy")
            if script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;">'
                    '⚠️ LEGACY<br><small>Not supported for PSBT</small>'
                    '</div>'
                )
            elif script_type == "P2SH-P2WPKH":
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;">'
                    '⚠️ NESTED<br><small>Not supported yet</small>'
                    '</div>'
                )
            else:
                health_html = (
                    f'<div class="health health-{u.get("health", "unknown").lower()}">'
                    f'{u.get("health", "UNKNOWN")}<br><small>Cannot prune</small></div>'
                )
        else:
            health_html = (
                f'<div class="health health-{u.get("health", "OPTIMAL").lower()}">'
                f'{u.get("health", "OPTIMAL")}<br><small>{u.get("recommend", "")}</small></div>'
            )

        # Friendly display name
        display_type = {
            "P2WPKH": "Native SegWit",
            "Taproot": "Taproot",
            "P2SH-P2WPKH": "Nested SegWit",
            "P2PKH": "Legacy",
            "Legacy": "Legacy",  # ← catches classification output
        }.get(script_type, script_type)

        rows.append([
            selected,
            u.get("source", "Single"),
            u.get("txid", "unknown")[:8] + "..." + u.get("txid", "unknown")[-8:],
            health_html,
            u.get("value", 0),
            u.get("address", "unknown"),
            u.get("input_weight", 0),
            display_type,
            u.get("vout", 0),
        ])

    return rows, has_unsupported
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
        return f"""
              <div style="
        margin-top:16px !important;
        padding:16px !important;
        background:#440000 !important;
        border:3px solid #ff3366 !important;
        border-radius:14px !important;
        box-shadow:0 0 50px rgba(255,51,102,0.9) !important;
        font-size:1.22rem !important;
        line-height:1.7 !important;
        color:#ffcccc !important;
    ">
        <div style="
            color:#ff3366 !important;
            font-size:1.6rem !important;
            font-weight:900 !important;
            text-shadow:0 0 30px #ff3366 !important;
        ">
            EXTREME CIOH LINKAGE
        </div><br>
        <div style="color:#ff6688 !important; font-size:1.15rem !important;">
            Common Input Ownership Heuristic (CIOH)
        </div><br>
        This consolidation strongly proves common ownership of many inputs/addresses.<br><br>
        <div style="color:#ffaaaa !important;">Privacy state: Severely compromised</div><br>
        Maximum fee savings, but analysts will confidently cluster these addresses as yours.<br>
        Consider CoinJoin, PayJoin, or silent payments afterward to restore privacy.
    </div>
           """
    elif privacy_score <= 50:
        return f"""
             <div style="
        margin-top:14px !important;
        padding:14px !important;
        background:#331100 !important;
        border:2px solid #ff8800 !important;
        border-radius:12px !important;
        font-size:1.18rem !important;
        line-height:1.6 !important;
        color:#ffddaa !important;
    ">
        <div style="
            color:#ff9900 !important;
            font-size:1.6rem !important;
            font-weight:900 !important;
            text-shadow:0 0 30px #ff9900 !important;
        ">
            HIGH CIOH RISK
        </div><br>
        <div style="color:#ffaa44 !important; font-size:1.12rem !important;">
            Common Input Ownership Heuristic (CIOH)
        </div><br>
        Merging {input_count} inputs from {distinct_addrs} address(es) → analysts will cluster them as yours.<br><br>
        <div style="color:#ffcc88 !important;">Privacy state: Significantly reduced</div><br>
        Good fee savings, but real privacy trade-off.
    </div>
    """
    elif privacy_score <= 70:
        return f"""
            <div style="
        margin-top:12px !important;
        padding:12px !important;
        background:#113300 !important;
        border:1px solid #00ff9d !important;
        border-radius:10px !important;
        color:#aaffaa !important;
        font-size:1.15rem !important;
        line-height:1.6 !important;
    ">
        <div style="
            color:#00ff9d !important;
            font-size:1.6rem !important;
            font-weight:900 !important;
            text-shadow:0 0 30px #00ff9d !important;
        ">
            MODERATE CIOH
        </div><br>
        <div style="color:#66ffaa !important; font-size:1.1rem !important;">
            Common Input Ownership Heuristic (CIOH)
        </div><br>
        Spending multiple inputs together creates some on-chain linkage between them.<br>
        Analysts may assume they belong to the same person — but it's not definitive.<br><br>
        Privacy impact is moderate. Acceptable trade-off during low-fee periods when saving sats matters most.
    </div>
    """
    else:
        return f"""
         <div style="
        margin-top:10px !important;
        color:#aaffaa !important;
        font-size:1.05rem !important;
        line-height:1.5 !important;
    ">
        <div style="
            color:#00ffdd !important;
            font-size:1.6rem !important;
            font-weight:900 !important;
            text-shadow:0 0 30px #00ffdd !important;
        ">
            LOW CIOH IMPACT
        </div><br>
        <span style="color:#00ffdd !important; font-size:1.1rem !important;">
            (Common Input Ownership Heuristic)
        </span><br><br>
        Few inputs spent together — minimal new linkage created.<br>
        Your addresses remain well-separated on-chain.<br>
        Privacy preserved.
    </div>
    """

def estimate_coinjoin_mixes_needed(input_count: int, distinct_addrs: int, privacy_score: int) -> tuple[int, int]:
    """
    Estimate Whirlpool-style mixes needed to reasonably break CIOH linkage.
    Returns (min_mixes, max_mixes). Conservative and practical.
    """
    # Low impact — suggest optional hardening, not zero
    if privacy_score > 70:
        return 1, 1  # "Optional: 1 mix for extra caution"

    # Base from privacy score
    base = max(1, round(60 / privacy_score))

    # Input count penalty
    if input_count >= 30:
        base += 3
    elif input_count >= 15:
        base += 2
    elif input_count >= 8:
        base += 1

    # Distinct addresses penalty
    if distinct_addrs >= 10:
        base += 2
    elif distinct_addrs >= 5:
        base += 1

    min_mixes = max(1, base - 1)
    max_mixes = base + 1

    return min_mixes, max_mixes

def detect_payjoin_support(dest_input: str) -> tuple[bool, str | None]:
    """
    Check if destination supports PayJoin (BIP78).
    Returns (supports_payjoin: bool, payjoin_url: str | None)
    """
    dest_input = dest_input.strip().lower()
    
    if dest_input.startswith("bitcoin:"):
        parsed = urllib.parse.urlparse(dest_input)
        query = urllib.parse.parse_qs(parsed.query)
        pj_urls = query.get("pj", [])
        if pj_urls:
            pj_url = pj_urls[0]
            if pj_url.startswith("https://") and len(pj_url) > 10:
                return True, pj_url
    
    return False, None
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
            if len(dec) == 25 and dec[0] == 0x00:  # mainnet P2PKH version
                hash160 = dec[1:21]
                return b'\x76\xa9\x14' + hash160 + b'\x88\xac', {
                    'input_vb': 148,
                    'output_vb': 34,
                    'type': 'P2PKH'
                }
        except Exception:
            pass

        # Invalid legacy — fall to unknown, not fake P2WPKH
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 148, 'output_vb': 34, 'type': 'invalid'}

    # === P2SH (starts with '3') ===
    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:  # mainnet P2SH version
                hash160 = dec[1:21]
                return b'\xa9\x14' + hash160 + b'\x87', {
                    'input_vb': 91,
                    'output_vb': 32,
                    'type': 'P2SH'
                }
        except Exception:
            pass

        return b'\x00\x14' + b'\x00'*20, {'input_vb': 91, 'output_vb': 32, 'type': 'invalid'}

    # === Bech32 / Bech32m (bc1...) ===
    if addr.startswith('bc1'):
        data_part = addr[4:]

        if any(c not in CHARSET for c in data_part):
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        data = [CHARSET.find(c) for c in data_part]
        if len(data) < 1:
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        witness_version = data[0]

        # P2WPKH / P2WSH (v0, bc1q)
        if addr.startswith('bc1q') and witness_version == 0 and bech32_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) in (20, 32):
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

        # Taproot (v1, bc1p)
        if addr.startswith('bc1p') and witness_version == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {
                    'input_vb': 57,
                    'output_vb': 43,
                    'type': 'Taproot'
                }

        # Invalid bech32
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

    # === Final fallback ===
    return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'unknown'}

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


def create_psbt(tx: Tx, utxos: list[dict]) -> tuple[str, str]:
    """
    Build a Sparrow-compatible PSBT from a Tx object and enriched UTXOs.

    Args:
        tx: Unsigned transaction object with tx_ins and tx_outs.
        utxos: List of dicts, each must contain:
               - value (int, satoshis)
               - scriptPubKey (bytes)
               - script_type (str: "P2WPKH", "P2WSH", or "Taproot")
               - tap_internal_key (bytes, optional, only for Taproot)

    Returns:
        (base64-encoded PSBT string, "")
    """
    import base64

    def encode_varint(i: int) -> bytes:
        if i < 0xfd:
            return i.to_bytes(1, "little")
        elif i <= 0xffff:
            return b'\xfd' + i.to_bytes(2, "little")
        elif i <= 0xffffffff:
            return b'\xfe' + i.to_bytes(4, "little")
        else:
            return b'\xff' + i.to_bytes(8, "little")

    # ---- Input validation ----
    if not tx.tx_ins:
        raise ValueError("Transaction has no inputs")
    if len(tx.tx_ins) != len(utxos):
        raise ValueError(f"Input count mismatch: {len(tx.tx_ins)} inputs vs {len(utxos)} UTXOs")
    if not tx.tx_outs:
        raise ValueError("Transaction must have at least one output")

    raw_tx = tx.serialize_unsigned()
    psbt = b"psbt\xff"

    # ==== Global: unsigned transaction ====
    psbt += b"\x01"                          # key length = 1
    psbt += b"\x00"                          # key type: PSBT_GLOBAL_UNSIGNED_TX = 0x00
    psbt += encode_varint(len(raw_tx))
    psbt += raw_tx
    psbt += b"\x00"                          # field separator
    psbt += b"\x00"                          # end of global map

    # ==== Per-input maps ====
    for u in utxos:
        script_type = u.get("script_type")
        if script_type not in ("P2WPKH", "Taproot"):
            raise ValueError(f"Unsupported script type in PSBT: {script_type}. Only P2WPKH and Taproot are supported.")

        # ---- PSBT_IN_WITNESS_UTXO (key type 0x01) ----
        witness_utxo = (
            u["value"].to_bytes(8, "little") +
            encode_varint(len(u["scriptPubKey"])) +
            u["scriptPubKey"]
        )
        psbt += b"\x01"                          # key len
        psbt += b"\x01"                          # key type
        psbt += encode_varint(len(witness_utxo))
        psbt += witness_utxo

        # ---- Optional: Taproot internal key (0x17) ----
        if script_type == "Taproot":
            tapkey = u.get("tap_internal_key")
            if tapkey and len(tapkey) == 32:
                psbt += b"\x01"                  # key len
                psbt += b"\x17"                  # key type
                psbt += encode_varint(32)
                psbt += tapkey

        psbt += b"\x00"  # end input map

    # ==== Output maps (empty OK) ====
    for _ in tx.tx_outs:
        psbt += b"\x00"

    # ==== Final separator ====
    psbt += b"\x00"

	# ---- Debug: print input summary before returning PSBT ----
    print("=== PSBT Input Summary ===")
    for idx, u in enumerate(utxos):
        stype = u.get("script_type")
        value = u.get("value")
        tapkey = u.get("tap_internal_key")
        tap_info = f", Taproot key present" if stype == "Taproot" and tapkey else ""
        print(f"Input {idx}: type={stype}, value={value}{tap_info}")
    print("==========================")


    return base64.b64encode(psbt).decode("ascii"), ""

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

# =================
# Analyze functions
# ================


@dataclass(frozen=True)
class AnalyzeParams:
    fee_rate: int
    future_fee_rate: int
    dust_threshold: int
    dao_percent: float
    strategy: str
    offline_mode: bool
    addr_input: str
    manual_utxo_input: str
    scan_source: str

def _sanitize_analyze_inputs(
    addr_input,
    strategy,
    dust_threshold,
    fee_rate_slider,
    thank_you_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
) -> AnalyzeParams:
    """Normalize and clamp all analyze() inputs into a deterministic parameter object."""

    fee_rate = max(1, min(300, int(float(fee_rate_slider or 15))))
    future_fee_rate = max(5, min(500, int(float(future_fee_slider or 60))))
    dust_threshold = max(0, min(10000, int(float(dust_threshold or 546))))
    dao_percent = max(0.0, min(100.0, float(thank_you_slider or 5.0)))

    scan_source = (addr_input or "").strip()

    return AnalyzeParams(
        fee_rate=fee_rate,
        future_fee_rate=future_fee_rate,
        dust_threshold=dust_threshold,
        dao_percent=dao_percent,
        strategy=strategy,
        offline_mode=bool(offline_mode),
        addr_input=scan_source,
        manual_utxo_input=(manual_utxo_input or "").strip(),
        scan_source=scan_source,
    )

def _collect_manual_utxos(params: AnalyzeParams) -> List[Dict]:
    """Parse user-supplied offline UTXO lines into normalized UTXO dicts."""

    if not params.manual_utxo_input:
        return []

    utxos: List[Dict] = []

    for line in params.manual_utxo_input.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(":")
        if len(parts) < 3:
            continue

        try:
            txid, vout_str, value_str = parts[:3]
            addr = parts[3].strip() if len(parts) >= 4 else "unknown (manual)"

            vout = int(vout_str)
            value = int(value_str)

            if value <= params.dust_threshold:
                continue

            _, meta = address_to_script_pubkey(addr)
            input_weight = meta["input_vb"] * 4

            utxos.append({
                "txid": txid,
                "vout": vout,
                "value": value,
                "address": addr,
                "input_weight": input_weight,
                "health": "MANUAL",
                "recommend": "REVIEW",
                "script_type": meta.get("type", "Manual"),
                "source": "Manual Offline",
                "selected": False,
            })

        except Exception:
            continue

    return utxos

def _collect_online_utxos(params: AnalyzeParams) -> List[Dict]:
    """Scan addresses or xpubs and return raw UTXO dicts."""

    if not params.addr_input:
        return []

    utxos: List[Dict] = []
    entries = [e.strip() for e in params.addr_input.splitlines() if e.strip()]

    for entry in entries:
        is_xpub = entry[:4] in ("xpub", "ypub", "zpub", "tpub", "upub", "vpub")

        source_label = (
            f"xpub ({entry[:12]}...{entry[-6:]})"
            if is_xpub
            else (entry if len(entry) <= 34 else f"{entry[:31]}...")
        )

        raw_utxos = (
            scan_xpub(entry, params.dust_threshold)[0]
            if is_xpub
            else get_utxos(entry, params.dust_threshold)
        )

        if not raw_utxos:
            continue

        for u in raw_utxos:
            utxos.append({**u, "source": source_label})

    return utxos

def _classify_utxo(value: int, input_weight: int) -> Tuple[str, str, str]:
    # Base classification
    if input_weight <= 228:
        script_type = "Taproot"
        default_health = "OPTIMAL"
        default_rec = "KEEP"
    elif input_weight <= 272:
        script_type = "P2WPKH"
        default_health = "OPTIMAL"
        default_rec = "KEEP"
    elif input_weight <= 364:
        script_type = "P2SH-P2WPKH"
        default_health = "MEDIUM"
        default_rec = "CAUTION"
    else:
        script_type = "Legacy"
        default_health = "HEAVY"
        default_rec = "PRUNE"

    # Overrides (highest priority first)
    if value < 10_000:
        return script_type, "DUST", "PRUNE"

    if value > 100_000_000 and script_type in ("P2SH-P2WPKH", "Legacy"):
        return script_type, "CAREFUL", "OPTIONAL"

    # Fall back to base
    return script_type, default_health, default_rec

def _enrich_utxos(raw_utxos: list[dict], params: AnalyzeParams) -> list[dict]:
    """
    Attach script metadata, weight, health, recommendation, scriptPubKey,
    and Taproot internal key to UTXOs.
    """

    enriched: list[dict] = []

    for u in raw_utxos:
        # Get both scriptPubKey and metadata (input_vb, type, etc.)
        script_pubkey, meta = address_to_script_pubkey(u["address"])
        input_weight = meta["input_vb"] * 4

        # Classify UTXO
        script_type, health, recommend = _classify_utxo(u["value"], input_weight)

        # Taproot internal key extraction (if present)
        tap_internal_key = None
        if script_type == "Taproot":
            # Extract from meta if available; must be 32 bytes
            candidate = meta.get("tap_internal_key")
            if candidate and len(candidate) == 32:
                tap_internal_key = candidate

        enriched.append({
            **u,
            "input_weight": input_weight,
            "health": health,
            "recommend": recommend,
            "script_type": script_type,
            "scriptPubKey": script_pubkey,        # raw bytes, needed for PSBT
            "tap_internal_key": tap_internal_key, # None unless Taproot
            # "selected" is handled elsewhere
        })

    return enriched

def _apply_pruning_strategy(enriched: List[Dict], strategy: str) -> List[Dict]:
    """
    Apply deterministic pruning strategy.
    Returns a new list of UTXO dicts with 'selected' flags set.
    No mutation of input objects.
    """
	# prune lowest-health first

    ratio = PRUNING_RATIOS.get(strategy, 0.40)

    sorted_utxos = sorted(
        enriched,
        key=lambda u: (u["value"], u["txid"], u["vout"]),
        reverse=True,
    )

    keep_count = max(
        MIN_KEEP_UTXOS,
        int(len(sorted_utxos) * (1 - ratio)),
    )

    by_health = sorted(
        sorted_utxos,
        key=lambda u: HEALTH_PRIORITY[u["health"]],
    )

    prune_count = len(by_health) - keep_count

    result: List[Dict] = []
    for i, u in enumerate(by_health):
        new_u = dict(u)
        new_u["selected"] = i < prune_count
        result.append(new_u)

    return result

def _build_df_rows(enriched: List[Dict]) -> tuple[List[List], bool]:
    """
    Convert enriched UTXOs into dataframe rows.
    Handles unsupported script types by disabling selection and showing warnings.
    """
    rows: List[List] = []
    has_unsupported = False

    for u in enriched:
        script_type = u.get("script_type", "")
        selected = u.get("selected", False)

        # Supported types for PSBT generation
        supported_in_psbt = script_type in ("P2WPKH", "Taproot")

        if not supported_in_psbt:
            has_unsupported = True
            selected = False  # Force unselected

            # Unified warning badges
            if script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;">'
                    '⚠️ LEGACY<br><small>Not supported for PSBT</small>'
                    '</div>'
                )
            elif script_type == "P2SH-P2WPKH":
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;">'
                    '⚠️ NESTED<br><small>Not supported yet</small>'
                    '</div>'
                )
            else:
                # Fallback — should not happen
                health_html = (
                    f'<div class="health health-{u.get("health", "unknown").lower()}">'
                    f'{u.get("health", "UNKNOWN")}<br><small>Cannot prune</small></div>'
                )
        else:
            health_html = (
                f'<div class="health health-{u["health"].lower()}">'
                f'{u["health"]}<br><small>{u["recommend"]}</small></div>'
            )
            # Keep original selection for supported types

        # Friendly display name
        display_type = {
            "P2WPKH": "Native SegWit",
            "Taproot": "Taproot",
            "P2SH-P2WPKH": "Nested SegWit",
            "P2PKH": "Legacy",
            "Legacy": "Legacy",  # ← catch both
        }.get(script_type, script_type)

        rows.append([
            selected,
            u.get("source", "Single"),
            u["txid"][:8] + "..." + u["txid"][-8:],
            health_html,
            u["value"],
            u["address"],
            u["input_weight"],
            display_type,
            u["vout"],
        ])

    return rows, has_unsupported
def _freeze_enriched(
    enriched: List[Dict],
    *,
    strategy: str,
    scan_source: str,
) -> tuple:
    """Freeze enriched UTXOs with immutable metadata."""

    frozen_utxos = tuple(copy.deepcopy(u) for u in enriched)

    meta = {
        "strategy": strategy,
        "scan_source": scan_source,
        "timestamp": int(time.time()),   # ← Fixed here
    }

    return (meta, frozen_utxos)


def analyze(
    addr_input,
    strategy,
    dust_threshold,
    fee_rate_slider,
    thank_you_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
):
    """
    Main analysis entrypoint.
    Orchestrates input sanitization, UTXO collection, enrichment, pruning,
    and UI-safe deterministic outputs.
    """

    # --- 1. Sanitize & normalize inputs ---
    params = _sanitize_analyze_inputs(
        addr_input=addr_input,
        strategy=strategy,
        dust_threshold=dust_threshold,
        fee_rate_slider=fee_rate_slider,
        thank_you_slider=thank_you_slider,
        future_fee_slider=future_fee_slider,
        offline_mode=offline_mode,
        manual_utxo_input=manual_utxo_input,
    )

    # --- 2. Collect UTXOs ---
    if params.offline_mode:
        raw_utxos = _collect_manual_utxos(params)
    else:
        raw_utxos = _collect_online_utxos(params)

    # --- 3. Handle empty scan ---
    if not raw_utxos:
        return _analyze_empty(params.scan_source)

    # --- 4. Enrich UTXOs ---
    enriched = _enrich_utxos(raw_utxos, params)

    # Safety assertion
    missing_script_type = [u for u in enriched if "script_type" not in u]
    if missing_script_type:
        raise RuntimeError("Invariant violation: missing 'script_type' in enriched UTXOs")

    # --- 5. Apply pruning strategy ---
    enriched_pruned = _apply_pruning_strategy(enriched, params.strategy)

    # Safety assertions
    assert len(enriched_pruned) >= MIN_KEEP_UTXOS, "Pruning violated MIN_KEEP_UTXOS"
    assert any(not u["selected"] for u in enriched_pruned), "All UTXOs selected for pruning — rejected"
    assert params.strategy in PRUNING_RATIOS, "Unknown pruning strategy"

    # --- 6. Build table rows ---
    df_rows, has_unsupported = _build_df_rows(enriched_pruned)

    # --- 6.1 Warning banner for unsupported inputs ---
    if has_unsupported:
        warning_banner = (
            "<div style='color:#ff9900;background:#332200;padding:24px;border-radius:16px;"
            "font-weight:bold;text-align:center;font-size:1.3rem;margin:24px 0;box-shadow:0 0 40px rgba(255,153,0,0.4);'>"
            "⚠️ Some UTXOs Cannot Be Included in PSBT<br><br>"
            "Legacy (starting with 1...) and Nested SegWit (starting with 3...) inputs "
            "are not currently supported in the generated PSBT.<br><br>"
            "They are shown in the table for transparency but cannot be selected.<br>"
            "Please spend them separately or convert to Native SegWit (bc1q...) or Taproot (bc1p...) first."
            "</div>"
        )
    else:
        warning_banner = ""

    # --- 7. Freeze state ---
    frozen_state = _freeze_enriched(
        enriched_pruned,
        strategy=params.strategy,
        scan_source=params.scan_source,
    )

    # --- 8. Success return ---
    return _analyze_success(
        df_rows=df_rows,
        frozen_state=frozen_state,
        scan_source=params.scan_source,
        warning_banner=warning_banner,
    )

def _analyze_success(df_rows, frozen_state, scan_source, warning_banner=""):
    """Unified success return for analyze() — 7 outputs"""
    return (
        gr.update(value=df_rows),         # 0: df table
        frozen_state,                     # 1: enriched_state
        gr.update(value=warning_banner),  # 2: warning banner
        gr.update(visible=True),          # 3: generate_row
        gr.update(visible=True),          # 4: import_file
        scan_source,                      # 5: scan_source state
        "",                               # 6: placeholder if you added extra output — or remove if not
    )

def _analyze_empty(scan_source: str = ""):
    """Empty/failure state return — 7 outputs"""
    return (
        gr.update(value=[]),
        (),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
        scan_source,
        "",
    )


# ====================
# generate_summary_safe() — Refactored for Clarity
# ====================

def _render_locked_state() -> Tuple[str, gr.update]:
    """Early return when selection is locked."""
    return (
        "<div style='text-align:center;padding:60px;color:#888;font-size:1.4rem;'>"
        "SELECTION LOCKED — Ready to sign PSBT"
        "</div>",
        gr.update(visible=False)
    )

def _validate_utxos_and_selection(df, utxos: list[dict]):
    if not utxos:
        return None, 0, "NO_UTXOS"

    selected_utxos = _resolve_selected(df, utxos)
    pruned_count = len(selected_utxos)

    if pruned_count == 0:
        return None, 0, "NO_SELECTION"

    return selected_utxos, pruned_count, None


def _compute_privacy_metrics(selected_utxos: List[dict], total_utxos: int) -> Tuple[int, str]:
    privacy_score = calculate_privacy_score(selected_utxos, total_utxos)
    score_color = "#0f0" if privacy_score >= 70 else "#ff9900" if privacy_score >= 40 else "#ff3366"
    return privacy_score, score_color

def _compute_economics_safe(selected_utxos: List[dict], fee_rate: int, dao_percent: float) -> Optional[TxEconomics]:
    try:
        return estimate_tx_economics(selected_utxos, fee_rate, dao_percent)
    except ValueError:
        return None

def _render_dao_feedback(econ: TxEconomics, dao_percent: float) -> str:
    if dao_percent <= 0:
        return ""

    dao_raw = int((econ.total_in - econ.fee) * (dao_percent / 100.0))
    if econ.dao_amt >= 546:
        return (
            f" • <span style='color:#00ff88 !important;font-weight:800;text-shadow:0 0 20px #00ff88 !important;'>"
            f"DAO: {sats_to_btc_str(econ.dao_amt)}</span><br>"
            f"<span style='color:#00ffaa !important;font-size:0.95rem;font-style:italic;'>"
            f"Thank you. Your support keeps Ωmega Pruner free, sovereign, and evolving. • Ω</span>"
        )
    elif dao_raw > 0:
        return (
            f" • <span style='color:#ff3366 !important;font-weight:800;text-shadow:0 0 20px #ff3366 !important;'>"
            f"DAO: {sats_to_btc_str(dao_raw)} → absorbed into fee</span><br>"
            f"<span style='color:#ff6688 !important;font-size:0.9rem;font-style:italic;'>(below 546 sat dust threshold)</span>"
        )
    return ""
	

def _render_small_prune_warning(econ: TxEconomics, fee_rate: int) -> str:
    remainder_after_fee = econ.total_in - econ.fee
    current_fee = econ.fee

    if remainder_after_fee >= 15000:
        return ""

    if remainder_after_fee < 8000:
        title = "⚠️ Warning: No change output expected"
        color = "#ff3366"
        bg = "#330000"
        border = "#ff3366"
    else:
        title = "⚠️ Caution: Change output may be absorbed"
        color = "#ff8800"
        bg = "#331100"
        border = "#ff8800"

    ratio = round(econ.total_in / current_fee, 1) if current_fee > 0 else 0

    return f"""
    <div style="
        margin:26px 0 !important;
        padding:24px !important;
        background:{bg} !important;
        border:4px solid {border} !important;
        border-radius:16px !important;
        box-shadow:0 0 50px rgba(255,100,100,0.8) !important;
        font-size:1.28rem !important;
        line-height:1.8 !important;
        color:#ffeeee !important;
    ">
      <div style="
          color:{color} !important;
          font-size:1.55rem !important;
          text-shadow:0 0 15px {color} !important, 0 0 30px {color} !important;
          margin-bottom:20px !important;
      ">
        {title}
      </div>
      Post-fee remainder (~{remainder_after_fee:,} sats) is small.<br>
      The pruned value will likely be fully or partially absorbed into miner fees.<br><br>
      <div style="
          color:#ffff88 !important;
          font-size:1.35rem !important;
          text-shadow:0 0 12px #ffff99 !important, 0 0 25px #ffaa00 !important;
          line-height:1.8 !important;
      ">
        Only proceed if your goal is wallet cleanup
      </div>
      <div style="color:#ffdd88 !important; font-size:1.1rem !important; margin-top:8px !important;">
        — not expecting significant change back.
      </div><br>
      <div style="color:#ffaaaa !important; font-size:1.05rem !important; line-height:1.6 !important;">
        💡 For a reliable change output, aim for:<br>
        • Value Pruned > ~5× Current Fee (good change back)<br>
        • Value Pruned > ~10× Current Fee (very comfortable)<br><br>
        This prune: <span style="color:#ffffff !important; font-weight:800 !important;">{sats_to_btc_str(econ.total_in)}</span> value and <span style="color:#ffffff !important; font-weight:800 !important;">{current_fee:,} sats</span> fee<br>
        Ratio: <span style="color:#ffffff !important; font-weight:800 !important;">{ratio}×</span> current fee
      </div><br>
      <small style="color:#88ffcc !important;">
        💡 Pro tip: The bigger the prune (relative to fee), the more you get back as change. Small prunes = cleanup only.
      </small>
    </div>
    """

def _render_payjoin_badge(dest_value: str) -> str:
    if not dest_value:
        return ""

    supports_pj, _ = detect_payjoin_support(dest_value)
    if not supports_pj:
        return ""

    return f"""
    <div style="
        margin:24px 0 !important;
        padding:20px !important;
        background:#001100 !important;
        border:3px solid #00ff88 !important;
        border-radius:16px !important;
        box-shadow:0 0 60px rgba(0,255,136,0.8) !important;
        font-size:1.3rem !important;
        line-height:1.8 !important;
        color:#ccffcc !important;
    ">
      <div style="
          color:#00ff88 !important;
          font-size:1.6rem !important;
          font-weight:900 !important;
          text-shadow:0 0 30px #00ff88 !important;
      ">
        🟢 CIOH-PROTECTED SEND AVAILABLE
      </div><br>
      This destination supports <span style="color:#00ffdd !important;font-weight:900 !important;">PayJoin (BIP78)</span>.<br><br>
      PayJoin breaks the Common Input Ownership Heuristic by including receiver inputs — 
      preventing new address clustering from this transaction.<br><br>
      <div style="
          color:#aaffff !important;
          font-size:1.25rem !important;
          font-weight:900 !important;
      ">
        To complete a CIOH-free send:
      </div>
      <div style="
          color:#ffffff !important;
          font-size:1.1rem !important;
          margin-top:8px !important;
          line-height:1.6 !important;
      ">
        • Sign and broadcast with a PayJoin-compatible wallet<br>
        • Receiver must support PayJoin (e.g., BTCPay Server)<br>
        • Works great with: Sparrow ↔ BTCPay, JoinMarket, etc.
      </div><br>
      <div style="
          font-size:0.95rem !important;
          color:#88ffaa !important;
      ">
        Standard send = new CIOH created<br>
        PayJoin send = no new CIOH from this transaction
      </div>
    </div>
    """

def _render_pruning_explanation(pruned_count: int, remaining_utxos: int) -> str:
    return f"""
    <div style="
        margin:32px 0 !important;
        padding:28px !important;
        background:#001a00 !important;
        border:4px solid #00ff9d !important;
        border-radius:18px !important;
        box-shadow:0 0 60px rgba(0,255,157,0.7) !important, 
                   inset 0 0 40px rgba(0,255,157,0.1) !important;
        font-size:1.25rem !important;
        line-height:1.9 !important;
        color:#ccffe6 !important;
    ">
      <strong style="
          color:#00ff9d !important;
          font-size:1.65rem !important;
          text-shadow:0 0 20px #00ff9d !important, 0 0 40px #00ff9d !important;
      ">
        🧹 WHAT PRUNING ACTUALLY DOES
      </strong><br><br>
      Pruning <strong style="color:#aaffff !important;">removes inefficient UTXOs</strong> (dust, legacy, or heavy) from your address.<br><br>
      • You pay a fee now to delete <strong style="color:#00ffff !important;">{pruned_count}</strong> bad inputs.<br>
      • The <strong style="color:#00ffff !important;">{remaining_utxos}</strong> remaining UTXOs are now easier and cheaper to spend later.<br>
      • <strong style="color:#ffff88 !important;">If no change output is created:</strong> the pruned value is absorbed into fees — but your wallet is <strong style="color:#aaffff !important;">cleaner forever</strong>.<br><br>
      <strong style="color:#00ffaa !important;">Goal:</strong> Healthier address → lower future fees.<br>
      Pruning is often worth it during low-fee periods, even if you don’t get change back.<br><br>
      <small style="color:#88ffcc !important; font-style:italic !important;">
        💡 Tip: If your goal is to get change, only prune when total value pruned > ~10–20× the current expected fee.
      </small>
    </div>
    """

def generate_summary_safe(
    df,
    enriched_state,
    fee_rate,
    future_fee_rate,
    dao_percent,
    locked,
    strategy,
    dest_value: str,
) -> tuple:
    print(">>> generate_summary_safe CALLED")
    print(f"    locked = {locked}")
    print(f"    enriched_state type: {type(enriched_state)}, len: {len(enriched_state[1]) if isinstance(enriched_state, tuple) and len(enriched_state) == 2 else 'N/A'}")
    print(f"    df rows: {len(df) if df else 0}")
    print(f"    fee_rate = {fee_rate}, dao_percent = {dao_percent}")

    if locked:
        print("    → Returning locked state message")
        return _render_locked_state()

    # Extract actual UTXO list from frozen state
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        meta, utxos = enriched_state
    else:
        utxos = enriched_state or []
    
    total_utxos = len(utxos)

    # === No UTXOs at all ===
    if total_utxos == 0:
        return (
            no_utxos_msg,
            gr.update(visible=False)
        )

    # === Validate selection – now returns clean data only ===
    selected_utxos, pruned_count, error = _validate_utxos_and_selection(df, utxos)

    if error == "NO_UTXOS":
        return no_utxos_msg, gr.update(visible=False)
    if error == "NO_SELECTION":
        return select_msg, gr.update(visible=False)

    # === FINAL GUARD: Only count supported inputs in summary ===
    supported_selected = [
        u for u in selected_utxos
        if u.get("script_type") in ("P2WPKH", "Taproot")
    ]

    pruned_count = len(supported_selected)
    if pruned_count == 0:
        return select_msg, gr.update(visible=False)

    # Use supported_selected for everything below
    remaining_utxos = total_utxos - pruned_count
	
    privacy_score, score_color = _compute_privacy_metrics(selected_utxos, total_utxos)
    econ = _compute_economics_safe(selected_utxos, fee_rate, dao_percent)
    if econ is None or econ.remaining <= 0:
        return (
            "<div style='text-align:center !important;padding:40px !important;background:rgba(30,0,0,0.7) !important;backdrop-filter:blur(10px) !important;border:2px solid #ff3366 !important;border-radius:16px !important;"
            "box-shadow:0 0 40px rgba(255,51,102,0.5) !important;font-size:1.3rem !important;color:#ffaa88 !important;'>"
            "<strong style='color:#ff3366 !important;font-size:1.5rem !important;'>Transaction Invalid</strong><br><br>"
            f"Current fee ({econ.fee:,} sats @ {fee_rate} s/vB) exceeds available balance.<br><br>"
            "<strong>Lower the fee rate</strong> or select more UTXOs."
            "</div>",
            gr.update(visible=False)
        )

    all_input_weight = sum(u["input_weight"] for u in utxos)
    pre_vsize = max(
        (all_input_weight + 172 + total_utxos) // 4 + 10,
        (all_input_weight + 150 + total_utxos * 60) // 4 + 10,
    )
    savings_pct = round(100 * (1 - econ.vsize / pre_vsize), 1) if pre_vsize > econ.vsize else 0
    savings_label = "NUCLEAR" if savings_pct >= 70 else "EXCELLENT" if savings_pct >= 50 else "GOOD" if savings_pct >= 30 else "WEAK"

    sats_saved = max(0, econ.vsize * (future_fee_rate - fee_rate))

    # Component rendering (SP preview removed)
    dao_line = _render_dao_feedback(econ, dao_percent)
    small_warning = _render_small_prune_warning(econ, fee_rate)
    payjoin_badge = _render_payjoin_badge(dest_value)
    cioh_warning = get_cioh_warning(pruned_count, len({u["address"] for u in selected_utxos}), privacy_score)
    pruning_explanation = _render_pruning_explanation(pruned_count, remaining_utxos)

    strategy_label = strategy.split(" — ")[0] if " — " in strategy else "Recommended"

    status_box_html = f"""
    <div style="
        text-align:center !important;
        margin:40px auto 30px auto !important;
        padding:32px !important;
        background: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border:3px solid #f7931a !important;
        border-radius:24px !important;
        max-width:960px !important;
        box-shadow:
            0 0 100px rgba(247,147,26,0.7) !important,
            inset 0 0 60px rgba(247,147,26,0.15) !important;
        position: relative;
        z-index: 1;
    ">
      <div style="
          color:#0f0 !important;
          font-size:2.6rem !important;
          font-weight:900 !important;
          letter-spacing:3px !important;
          text-shadow:0 0 35px #0f0 !important, 0 0 70px #0f0 !important;
          margin-bottom:24px !important;
      ">
        SELECTION READY
      </div>
      <div style="
          color:#f7931a !important;
          font-size:1.8rem !important;
          font-weight:800 !important;
          margin:20px 0 !important;
      ">
        {total_utxos:,} UTXOs • <span style="color:#00ff9d !important;">{strategy_label}</span> Strategy Active
      </div>
      <div style="
          color:#fff !important;
          font-size:1.5rem !important;
          font-weight:700 !important;
          margin:16px 0 !important;
      ">
        Pruning <span style="color:#ff6600 !important;font-weight:900 !important;">{pruned_count:,}</span> inputs
      </div>
      <div style="
          color:#fff !important;
          font-size:1.7rem !important;
          font-weight:800 !important;
          margin:24px 0 !important;
      ">
        Privacy Score: 
        <span style="
            color:{score_color} !important;
            font-size:2.3rem !important;
            margin-left:12px !important;
            text-shadow:0 0 25px {score_color} !important;
        ">
          {privacy_score}/100
        </span>
      </div>
      <hr style="border:none !important;border-top:1px solid rgba(247,147,26,0.3) !important;margin:32px 0 !important;">
      <div style="font-size:1.1rem !important;line-height:2.1 !important;">
        <div style="margin:16px 0 !important;">
          <b style="color:#fff !important;">Full wallet spend size today (before pruning):</b> 
          <span style="color:#ff9900 !important;font-weight:800 !important;">~{pre_vsize:,} vB</span>
        </div>
        <div style="margin:16px 0 !important;">
          <b style="color:#fff !important;">Size of this one-time pruning cleanup transaction:</b> 
          <span style="color:#0f0 !important;font-weight:800 !important;">~{econ.vsize:,} vB</span>
        </div>
        <div style="margin:18px 0 !important;color:#88ffcc !important;font-size:1.05rem !important;line-height:1.6 !important;">
          💡 After pruning: your full wallet spend size drops to roughly <span style="color:#0f0 !important;font-weight:800 !important;">~{pre_vsize - econ.vsize + 200:,} vB</span>
        </div>
        <div style="
            margin:24px 0 !important;
            color:#0f0 !important;
            font-size:1.35rem !important;
            font-weight:900 !important;
            text-shadow:0 0 25px #0f0 !important;
            line-height:1.6 !important;
        ">
          {savings_label.upper()} WALLET CLEANUP!
        </div>
        <div style="margin:16px 0 !important;">
          <b style="color:#fff !important;">Current fee (paid now):</b> 
          <span style="color:#0f0 !important;font-weight:800 !important;">{econ.fee:,} sats @ {fee_rate} s/vB</span>{dao_line}
        </div>
        <div style="margin:20px 0 !important;color:#88ffcc !important;font-size:1.1rem !important;line-height:1.6 !important;">
          💧 Expected output: 
          <span style="color:#0f0 !important;font-weight:800 !important;">{econ.change_amt:,} sats</span>
          change sent to standard address
        </div>
        <div style="margin:20px 0 !important;color:#88ffcc !important;font-size:1.05rem !important;line-height:1.7 !important;">
          💡 Pruning now saves you <span style="color:#0f0 !important;font-weight:800 !important;">+{sats_saved:,} sats</span> versus pruning later if fees reach {future_fee_rate} s/vB
        </div>
      </div>
      {payjoin_badge}
      <hr style="border:none !important;border-top:1px solid rgba(247,147,26,0.3) !important;margin:32px 0 !important;">
      <div style="margin:32px 0 40px 0 !important;line-height:1.7 !important;">
        {cioh_warning}
      </div>
      <hr style="border:none !important;border-top:1px solid rgba(247,147,26,0.3) !important;margin:32px 0 !important;">
      <div style="
          margin:0 20px 40px 20px !important;
          padding:28px !important;
          background: rgba(0,20,0,0.6) !important;
          backdrop-filter: blur(10px) !important;
          border:3px solid #00ff9d !important;
          border-radius:18px !important;
          box-shadow:0 0 60px rgba(0,255,157,0.5) !important;
      ">
        {pruning_explanation}
      </div>
      {small_warning}
    </div>
    """

    return status_box_html, gr.update(visible=pruned_count > 0)
  
# ====================
# on_generate() & generate_psbt() — Refactored for Clarity
# ====================

def _extract_selected_utxos(enriched_state: tuple) -> List[dict]:
    """Safely extract selected UTXOs from frozen enriched_state."""
    # Extract the actual UTXOs list/tuple from the frozen state
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        _, utxos = enriched_state
    else:
        utxos = enriched_state or ()

    return [u for u in utxos if u.get("selected", False)]
	
def _create_psbt_snapshot(
    selected_utxos: List[dict],
    scan_source: str,
    dest_override: Optional[str],
    fee_rate: int,
    future_fee_rate: int,
    dao_percent: float,
) -> dict:
    if not selected_utxos:
        raise ValueError("No UTXOs selected")

    # Clean, JSON-serializable input list
    clean_inputs = [
        {
            "txid": u["txid"],
            "vout": u["vout"],
            "value": u["value"],
            "address": u.get("address"),
            "script_type": u.get("script_type"),
            "health": u.get("health"),
            "source": u.get("source"),
        }
        for u in sorted(selected_utxos, key=lambda x: (x["txid"], x["vout"]))
    ]

    snapshot = {
        "version": 1,
        "timestamp": int(time.time()),
        "scan_source": scan_source.strip(),
        "dest_addr_override": dest_override.strip() if dest_override else None,
        "fee_rate": fee_rate,
        "future_fee_rate": future_fee_rate,
        "dao_percent": dao_percent,
        "inputs": clean_inputs,
    }

    # Deterministic fingerprint
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    snapshot["fingerprint"] = fingerprint
    snapshot["fingerprint_short"] = fingerprint[:16].upper()

    return snapshot

def _persist_snapshot(snapshot: dict) -> str:
    """Write snapshot to temporary file for download."""
    date_str = datetime.now().strftime("%Y%m%d")
    fingerprint_short = snapshot["fingerprint_short"]
    filename_prefix = f"Ωmega_Prune_{date_str}_{fingerprint_short[:8]}"

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=filename_prefix,
        delete=False,
    )
    json_str = json.dumps(snapshot, indent=2, ensure_ascii=False)
    tmp_file.write(json_str)
    tmp_file.close()

    return tmp_file.name

def on_generate(
    dest_value: str,
    fee_rate: int,
    future_fee_rate: int,
    dao_percent: float,
    enriched_state: tuple,
    scan_source: str,
) -> tuple:
    """Freeze user intent and return both snapshot and full UTXOs for PSBT generation."""
    log.info("on_generate called")

    if not enriched_state:
        log.info("enriched_state is empty")
        return None, [], gr.update(value=False), None

    selected_utxos = _extract_selected_utxos(enriched_state)
    log.info(f"Selected UTXOs count: {len(selected_utxos)}")

    if not selected_utxos:
        log.info("Generate attempted with no UTXOs selected")
        return None, [], gr.update(value=False), None

    log.info(f"dest_value: {dest_value!r}")

    try:
        log.info("Creating snapshot...")
        snapshot = _create_psbt_snapshot(
            selected_utxos=selected_utxos,
            scan_source=scan_source,
            dest_override=dest_value,
            fee_rate=fee_rate,
            future_fee_rate=future_fee_rate,
            dao_percent=dao_percent,
        )

        log.info("Snapshot created, persisting to file...")
        file_path = _persist_snapshot(snapshot)

        log.info("Snapshot persisted — returning success")
        # Return: snapshot, full_selected_utxos, locked_update, file_path
        return snapshot, selected_utxos, gr.update(value=True), file_path

    except Exception as e:
        log.error(f"Snapshot creation failed: {e}", exc_info=True)
        return None, [], gr.update(value=False), None
		
# ====================
# generate_psbt() helpers
# ====================

def _render_no_snapshot() -> str:
    return (
        "<div style='color:#ff3366 !important;text-align:center !important;padding:30px !important;font-size:1.2rem !important;font-weight:700 !important;'>"
        "No snapshot — run Generate first."
        "</div>"
    )

def _render_no_inputs() -> str:
    return (
        "<div style='color:#ff3366 !important;text-align:center !important;padding:30px !important;font-size:1.2rem !important;font-weight:700 !important;'>"
        "No UTXOs selected for pruning!"
        "</div>"
    )

@dataclass(frozen=True)
class PsbtParams:
    inputs: List[dict]
    scan_source: str
    dest_override: Optional[str]
    fee_rate: int
    dao_percent: float
    fingerprint_short: str

def _extract_psbt_params(snapshot: dict) -> PsbtParams:
    return PsbtParams(
        inputs=snapshot["inputs"],
        scan_source=snapshot["scan_source"],
        dest_override=snapshot.get("dest_addr_override"),
        fee_rate=snapshot["fee_rate"],
        dao_percent=snapshot["dao_percent"],
        fingerprint_short=snapshot["fingerprint_short"],
    )

def _resolve_destination(dest_override: Optional[str], scan_source: str) -> Union[bytes, str]:
    """Validate and resolve final destination scriptPubKey or return error HTML."""
    final_dest = (dest_override or scan_source).strip()
    if not final_dest:
        return (
            "<div style='"
            "color:#ff3366 !important;"
            "text-align:center !important;"
            "padding:40px !important;"
            "background:#440000 !important;"
            "border-radius:16px !important;"
            "box-shadow:0 0 40px rgba(255,51,102,0.4) !important;"
            "font-size:1.3rem !important;"
            "line-height:1.7 !important;"
            "'>"
            "<div style='"
            "color:#ff3366 !important;"
            "font-size:1.5rem !important;"
            "font-weight:900 !important;"
            "'>No destination address available.</div><br><br>"
            "Please enter a destination address or ensure your scan source is valid."
            "</div>"
        )

    if final_dest.startswith(("xpub", "ypub", "zpub", "tpub", "upub", "vpub")):
        return (
            "<div style='"
            "color:#ffcc00 !important;"
            "text-align:center !important;"
            "padding:40px !important;"
            "background:#332200 !important;"
            "border-radius:16px !important;"
            "box-shadow:0 0 40px rgba(255,204,0,0.4) !important;"
            "font-size:1.3rem !important;"
            "line-height:1.7 !important;"
            "'>"
            "<div style='"
            "color:#ffcc00 !important;"
            "font-size:1.5rem !important;"
            "font-weight:900 !important;"
            "'>xpub detected as scan source.</div><br><br>"
            "Please specify a destination address.<br>"
            "Automatic derivation coming soon."
            "</div>"
        )

    try:
        spk, _ = address_to_script_pubkey(final_dest)
        return spk
    except Exception:
        return (
            "<div style='"
            "color:#ff3366 !important;"
            "text-align:center !important;"
            "padding:40px !important;"
            "background:#440000 !important;"
            "border-radius:16px !important;"
            "box-shadow:0 0 40px rgba(255,51,102,0.4) !important;"
            "font-size:1.3rem !important;"
            "line-height:1.7 !important;"
            "'>"
            "<div style='"
            "color:#ff3366 !important;"
            "font-size:1.5rem !important;"
            "font-weight:900 !important;"
            "'>Invalid destination address.</div><br><br>"
            "Please check the address format."
            "</div>"
        )

	
def _build_unsigned_tx(
    inputs: List[dict],
    econ: TxEconomics,
    dest_spk: bytes,
    params: PsbtParams,
) -> tuple[Tx, list[dict]]:
    """
    Construct unsigned transaction and return prepared UTXO info for PSBT.

    Returns:
        tx: Unsigned Tx object.
        utxos_for_psbt: List of dicts with 'value', 'scriptPubKey', 'script_type' per input.
    """
    tx = Tx()
    utxos_for_psbt: list[dict[str, any]] = []

    for u in inputs:
        tx.tx_ins.append(TxIn(bytes.fromhex(u["txid"]), int(u["vout"])))

        # Prepare UTXO info for PSBT
        utxos_for_psbt.append({
            "value": u["value"],
            "scriptPubKey": address_to_script_pubkey(u["address"])[0],
            "script_type": u.get("script_type", "unknown"),
        })

    # DAO output (if any)
    if econ.dao_amt > 0:
        tx.tx_outs.append(TxOut(econ.dao_amt, DEFAULT_DAO_SCRIPT_PUBKEY))

    # Change / remainder output
    change_amt = econ.change_amt
    if getattr(params, "silent_payment_full", False):
        change_amt = econ.total_in - econ.fee - econ.dao_amt

    if change_amt > 0:
        tx.tx_outs.append(TxOut(change_amt, dest_spk))

    return tx, utxos_for_psbt

def _generate_qr(psbt_b64: str) -> Tuple[str, str]:
    """Generate QR code for PSBT, with graceful fallback for large PSBTs."""
    # Safe threshold — QR version 40 max ~2953 chars
    if len(psbt_b64) > 2900:
        qr_html = ""
        qr_warning = (
            "<div style='"
            "margin:40px 0 !important;"
            "padding:32px !important;"
            "background:#221100 !important;"  # Dark amber background
            "border:4px solid #ff9900 !important;"
            "border-radius:18px !important;"
            "text-align:center !important;"
            "font-size:1.35rem !important;"
            "color:#ffeecc !important;"       # Warm light text — never gray
            "box-shadow:0 0 70px rgba(255,153,0,0.6) !important;"
            "'>"
            "<strong style='color:#ffff66 !important;font-size:1.7rem !important;text-shadow:0 0 30px #ffff00 !important;'>"
            "PSBT Too Large for QR Code"
            "</strong><br><br>"
            f"<span style='color:#ffddaa !important;'>Size: {len(psbt_b64):,} characters</span><br><br>"
            "Use the <strong style='color:#00ffff !important;font-size:1.4rem !important;text-shadow:0 0 25px #00ffff !important;'>"
            "COPY PSBT"
            "</strong> button below and paste directly into your wallet.<br><br>"
            "<span style='color:#aaffff !important;font-size:1.1rem !important;'>"
            "Sparrow • Coldcard • Electrum • Most wallets support direct paste"
            "</span>"
            "</div>"
        )
        return qr_html, qr_warning

    # Normal QR generation
    error_correction = qrcode.constants.ERROR_CORRECT_L if len(psbt_b64) > 2600 else qrcode.constants.ERROR_CORRECT_M
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=6,
        border=4,
    )
    qr.add_data(f"bitcoin:?psbt={psbt_b64}")
    qr.make(fit=True)

    img = qr.make_image(fill_color="#f7931a", back_color="#000000")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    qr_html = (
        f'<img src="{qr_uri}" style="'
        "width:100% !important;height:auto !important;display:block !important;"
        "image-rendering:crisp-edges !important;border-radius:12px !important;\" "
        "alt=\"PSBT QR Code\"/>"
    )

    qr_warning = ""
    if len(psbt_b64) > 2600:
        qr_warning = (
            "<div style='margin-top:16px;padding:12px;background:#331a00;border:1px solid #ff9900;border-radius:10px;"
            "color:#ffb347;font-size:0.95rem;line-height:1.4;'>"
            "Large PSBT — if QR scan fails, use <strong>COPY PSBT</strong> and paste manually."
            "</div>"
        )

    return qr_html, qr_warning

	
def _render_payjoin_note(dest_addr: str) -> str:
    if not dest_addr:
        return ""

    supports_pj, _ = detect_payjoin_support(dest_addr)
    if not supports_pj:
        return ""

    return f"""
    <div style="
        margin:30px 0 !important;
        padding:16px !important;
        background:#001100 !important;
        border:2px solid #00ff88 !important;
        border-radius:12px !important;
        box-shadow:0 0 40px rgba(0,255,136,0.5) !important;
        font-size:1.05rem !important;
        line-height:1.6 !important;
        color:#ccffcc !important;
    ">
      🟢 PayJoin possible with this receiver<br><br>
      <div style="
          color:#aaffff !important;
          font-size:1.1rem !important;
          font-weight:900 !important;
      ">
        Sign with a PayJoin-compatible wallet to complete the CIOH-free send
      </div><br>
      <div style="
          font-size:0.95rem !important;
          color:#88ffaa !important;
      ">
        Standard signing = normal CIOH created
      </div>
    </div>
    """

def _compose_psbt_html(
    fingerprint: str,
    qr_html: str,
    qr_warning: str,
    psbt_b64: str,
    payjoin_note: str,
	extra_note: str = "",
) -> str:
    return f"""
    <div style="height: 80px !important;"></div>
<div style="
    text-align:center !important;
    margin:60px auto 0px !important;
    max-width:960px !important;
    position: relative;
    z-index: 1;
">
<div style="
    display:inline-block !important;
    padding:55px !important;
    background: rgba(0, 0, 0, 0.42) !important;   /* Matches hero banner feel */
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(16px) saturate(140%) !important;
    border:14px solid #f7931a !important;
    border-radius:36px !important;
    box-shadow: 
        0 0 140px rgba(247,147,26,0.95) !important,
        inset 0 0 60px rgba(247,147,26,0.15) !important;
    position: relative;
">
     <!-- Selection Fingerprint -->
    <div style="
        margin:40px 0 !important;
        padding:28px !important;
        background: rgba(0, 30, 0, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        border:4px solid #0f0 !important;
        border-radius:18px !important;
        box-shadow:0 0 80px rgba(0,255,0,0.8) !important;
        font-family:monospace !important;
    ">
        <div style="
            color:#0f0 !important;
            font-size:1.4rem !important;
            font-weight:900 !important;
            letter-spacing:3px !important;
            text-shadow:0 0 20px #0f0 !important;
            margin-bottom:16px !important;
        ">
            Ω FINGERPRINT
        </div>
        <div style="
            color:#00ff9d !important;
            font-size:2.2rem !important;
            font-weight:900 !important;
            letter-spacing:8px !important;
            text-shadow:0 0 30px #00ff9d !important, 0 0 60px #00ff9d !important;
        ">
            {fingerprint}
        </div>
        <div style="
            margin-top:20px !important;
            color:#00ffaa !important;
            font-size:1.1rem !important;
            line-height:1.6 !important;
            font-weight:800 !important;
        ">
            Cryptographic proof of your pruning selection<br>
            Deterministic • Audit-proof • Never changes
        </div>
        <button onclick="navigator.clipboard.writeText('{fingerprint}').then(() => {{this.innerText='COPIED';setTimeout(()=>this.innerText='COPY FINGERPRINT',1500);}})"
            style="margin-top:16px;padding:8px 20px;background:#000;color:#0f0;border:2px solid #0f0;border-radius:12px;font-size:1.1rem;font-weight:800;cursor:pointer;box-shadow:0 0 20px #0f0;">
            COPY FINGERPRINT
        </button>
    </div>
        {extra_note}
    <div style="
    color:#aaffaa !important;
    font-size:1.05rem !important;
    margin:30px 0 !important;
    text-align:center !important;
">
  Change output sent to standard address
</div>
    {payjoin_note}

    <!-- QR -->
    <div style="
        margin:40px auto !important;
        width:520px !important;
        max-width:96vw !important;
        padding:20px !important;
        background: rgba(0,0,0,0.7) !important;
        backdrop-filter: blur(10px) !important;
        border:8px solid #0f0 !important;
        border-radius:24px !important;
        box-shadow:0 0 60px #0f0 !important;
    ">
        {qr_html}
    </div>

    {qr_warning}

    <!-- PSBT Output -->
    <div style="margin:60px auto 20px !important;width:92% !important;max-width:880px !important;">
        <div style="
            position:relative !important;
            background: rgba(0,0,0,0.75) !important;
            backdrop-filter: blur(12px) !important;
            border:6px solid #f7931a !important;
            border-radius:18px !important;
            box-shadow:0 0 40px #0f0 !important;
            overflow:hidden !important;
        ">
            <textarea id="psbt-output" readonly 
                style="
                    width:100% !important;
                    height:180px !important;
                    background:transparent !important;
                    color:#0f0 !important;
                    font-size:1rem !important;
                    padding:24px !important;
                    padding-right:140px !important;
                    border:none !important;
                    outline:none !important;
                    resize:none !important;
                    font-family:monospace !important;
                    font-weight:700 !important;
                ">
{psbt_b64}</textarea>
            <button onclick="navigator.clipboard.writeText(document.getElementById('psbt-output').value).then(() => {{this.innerText='COPIED';setTimeout(()=>this.innerText='COPY PSBT',1500);}})"
                style="
                    position:absolute !important;
                    top:14px !important;
                    right:14px !important;
                    padding:12px 30px !important;
                    background:#f7931a !important;
                    color:#000 !important;
                    border:none !important;
                    border-radius:14px !important;
                    font-weight:800 !important;
                    font-size:1.12rem !important;
                    cursor:pointer !important;
                    box-shadow:0 0 30px #f7931a !important;
                ">
                COPY PSBT
            </button>
        </div>
        <div style="text-align:center !important;margin-top:12px !important;">
            <span style="color:#00f0ff !important;font-weight:700 !important;">RBF enabled</span>
            <span style="color:#888 !important;"> • Raw PSBT • </span>
            <span style="color:#666 !important;font-size:0.9rem !important;">Inspect before signing</span>
        </div>
    </div>

    <!-- Wallet support -->
    <div style="
        color:#ff9900 !important;
        font-size:1rem !important;
        text-align:center !important;
        margin:40px 0 20px !important;
        padding:16px !important;
        background: rgba(30,0,0,0.6) !important;
        backdrop-filter: blur(8px) !important;
        border:2px solid #f7931a !important;
        border-radius:12px !important;
        box-shadow:0 0 40px rgba(247,147,26,0.4) !important;
    ">
        <div style='color:#fff !important;font-weight:800 !important;'>
            Important: Wallet must support <strong style='color:#0f0 !important;'>PSBT</strong>
        </div>
        <div style='color:#0f8 !important;margin-top:8px !important;'>
            Sparrow • BlueWallet • Electrum • UniSat • Nunchuk • OKX
        </div>
    </div>
</div>
</div>
"""

def generate_psbt(psbt_snapshot: dict, full_selected_utxos: list[dict]) -> str:
    """Orchestrate PSBT generation using both snapshot (params) and full enriched UTXOs."""
    print(f">>> generate_psbt received snapshot: {type(psbt_snapshot)}")
    print(f">>> generate_psbt received full_utxos: {len(full_selected_utxos) if full_selected_utxos else 0} items")
    print(f">>> First UTXO keys: {list(full_selected_utxos[0].keys()) if full_selected_utxos else 'N/A'}")

    if not psbt_snapshot:
        return _render_no_snapshot()

    if not full_selected_utxos:
        return _render_no_inputs()

    # Safe param extraction — critical guard
    try:
        params = _extract_psbt_params(psbt_snapshot)
    except Exception as e:
        log.error(f"Failed to extract PSBT params: {e}", exc_info=True)
        return (
            "<div style='color:#ff3366;background:#440000;padding:40px;border-radius:16px;text-align:center;"
            "font-size:1.4rem;box-shadow:0 0 40px rgba(255,51,102,0.5);'>"
            "Invalid or corrupted snapshot<br><br>"
            "Please click <strong>GENERATE</strong> again."
            "</div>"
        )

    # Use the full enriched UTXOs
    all_inputs = full_selected_utxos

    # Filter to supported types
    supported_inputs = [
        u for u in all_inputs
        if u.get("script_type") in ("P2WPKH", "Taproot")
    ]

    if not supported_inputs:
        return (
            "<div style='color:#ff9900;background:#332200;padding:40px;border-radius:16px;text-align:center;"
            "font-size:1.4rem;box-shadow:0 0 50px rgba(255,153,0,0.4);'>"
            "No supported inputs selected<br><br>"
            "Only <strong>Native SegWit (bc1q...)</strong> and <strong>Taproot (bc1p...)</strong> inputs can be pruned.<br>"
            "Legacy and Nested inputs are excluded from the PSBT."
            "</div>"
        )

    legacy_excluded = len(supported_inputs) < len(all_inputs)

    # Continue with your existing code...
    dest_result = _resolve_destination(params.dest_override, params.scan_source)
    if isinstance(dest_result, str):
        return dest_result
    dest_spk = dest_result

    try:
        econ = estimate_tx_economics(supported_inputs, params.fee_rate, params.dao_percent)
    except ValueError:
        return (
            "<div style='color:#ff3366;text-align:center;padding:30px;font-size:1.3rem;font-weight:700;'>"
            "Invalid transaction economics — please re-analyze."
            "</div>"
        )

    tx, utxos_for_psbt = _build_unsigned_tx(supported_inputs, econ, dest_spk, params)

    if len(utxos_for_psbt) != len(tx.tx_ins):
        return (
            "<div style='color:#ff3366;text-align:center;padding:40px;font-size:1.4rem;font-weight:700;'>"
            "Internal error: Input/UTXO count mismatch — please report this bug."
            "</div>"
        )

    # Generate PSBT
    psbt_b64, _ = create_psbt(tx, utxos_for_psbt)

    qr_html, qr_warning = _generate_qr(psbt_b64)
    payjoin_note = _render_payjoin_note(params.dest_override or params.scan_source)

    # Warning if we excluded any inputs
    extra_note = ""
    if legacy_excluded:
        extra_note = (
            "<div style='"
            "color:#ffaa00;background:#332200;padding:16px;border-radius:12px;"
            "margin:40px 0 30px 0;text-align:center;font-size:1.15rem;"
            "border:2px solid #ff8800;box-shadow:0 0 30px rgba(255,136,0,0.4);"
            "'>"
            "⚠️ Some inputs were excluded from this PSBT<br>"
            "<small>Only Native SegWit and Taproot inputs are supported. "
            "Legacy/Nested inputs were automatically skipped.</small>"
            "</div>"
        )

    return _compose_psbt_html(
        fingerprint=params.fingerprint_short,
        qr_html=qr_html,
        qr_warning=qr_warning,
        psbt_b64=psbt_b64,
        payjoin_note=payjoin_note,
        extra_note=extra_note,
    )

def analyze_and_show_summary(
    addr_input,
    strategy,
    dust_threshold,
    fee_rate_slider,
    thank_you_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
    locked,
    dest_value,
):
    print(">>> analyze_and_show_summary STARTED")

    # Run core analysis
    df_update, enriched_new, warning_banner, gen_row_vis, import_vis, scan_source_new, _ = analyze(
        addr_input,
        strategy,
        dust_threshold,
        fee_rate_slider,
        thank_you_slider,
        future_fee_slider,
        offline_mode,
        manual_utxo_input,
    )

    # Extract fresh rows from the update payload
    if isinstance(df_update, dict):
        df_rows = df_update.get("value", [])
    elif hasattr(df_update, "value"):
        df_rows = df_update.value
    else:
        df_rows = []

    print(f">>> rows built: {len(df_rows)} UTXOs")

    # Call generate_summary_safe with fresh data
    status_box_html, generate_row_visibility = generate_summary_safe(
        df_rows,
        enriched_new,
        fee_rate_slider,
        future_fee_slider,
        thank_you_slider,
        locked,
        strategy,
        dest_value,
    )

    print(">>> status_box_html generated")
    print(f">>> generate_row should be visible: {generate_row_visibility.visible if hasattr(generate_row_visibility, 'visible') else 'unknown'}")

    # Return in correct order matching your .click() outputs
    return (
        df_update,              # 0: df (table)
        enriched_new,           # 1: enriched_state
        warning_banner,         # 2: warning_banner HTML
        generate_row_visibility,# 3: generate_row visibility (critical!)
        import_file,          # 4: import_file visibility
        scan_source_new,        # 5: scan_source state
        status_box_html,        # 6: status_output (the big glowing box)
    )
# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks(
    title="Ωmega Pruner v11 — Forged Anew"
) as demo:
    # Social / OpenGraph Preview
    gr.HTML("""
    <meta property="og:title" content="Ωmega Pruner v11 — Forged Anew">
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
        z-index:0;
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
                text-shadow:
                  0 0 80px rgba(247,147,26,0.55),
                  0 0 140px rgba(247,147,26,0.35);
                animation: omega-breath 28s infinite ease-in-out;
                user-select: none;
                line-height: 1;
                opacity: 0.96;
            ">Ω</span>
        </span>
    </div>

    <div style="text-align:center;margin:100px 0 30px 0;padding:60px 40px;
                background:rgba(0,0,0,0.42);
                backdrop-filter: blur(10px);
                border:8px solid #f7931a;
                border-radius:32px;
                box-shadow:0 0 100px rgba(247,147,26,0.5), inset 0 0 80px rgba(247,147,26,0.1);
                max-width:900px;margin-left:auto;margin-right:auto;
                position:relative;z-index:1;">

    <div style="color:#f7931a;font-size:4.8rem;font-weight:900;letter-spacing:12px;
                text-shadow:0 0 40px #f7931a, 0 0 80px #ffaa00, 0 0 120px rgba(247,147,26,0.9);
                margin-bottom:20px;">
        ΩMEGA PRUNER
    </div>
    
    <div style="color:#0f0;font-size:2.6rem;font-weight:900;letter-spacing:6px;
                text-shadow:0 0 35px #0f0, 0 0 70px #0f0;margin:30px 0;">
        NUCLEAR COIN CONTROL
    </div>

    <div style="color:#00ffaa;font-size:1.2rem;letter-spacing:3px;margin:20px 0 20px 0;
                text-shadow:0 0 15px #00ffaa;">
        FORGED ANEW — v11
    </div>
    
    <div style="color:#ddd;font-size:1.5rem;line-height:1.8;max-width:760px;margin:40px auto 50px auto;">
        Pruning isn't just about saving sats today — it's about <strong style="color:#0f0;">taking control</strong> of your coins for the long term.<br><br>
        
        By consolidating inefficient UTXOs, you:<br>
        • <strong style="color:#00ff9d;">Save significantly on fees</strong> during peak congestion<br>
        • <strong style="color:#00ff9d;">Gain true coin control</strong> — know exactly what you're spending<br>
        • <strong style="color:#00ff9d;">Improve privacy</strong> through deliberate structure<br>
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
        0%, 100% { opacity: 0.78; transform: scale(0.97); }
        50% { opacity: 1.0; transform: scale(1.03); }
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
        z-index: 1;
        background: transparent !important;
		
    }

    #omega-bg { 
        isolation: isolate;
    }
    .omega-wrap {
    will-change: transform;
    animation-timing-function: linear;
    transform-origin: 50% 52%;
    transform-origin: 49.5% 52.3%;
}
    .omega-symbol {
    will-change: opacity;
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

    /* Legacy row styling */
.health-legacy {
    color: #ff4444 !important;
    font-weight: bold;
}

tr:has(.health-legacy) {
    background-color: #330000 !important;
    opacity: 0.65;
}

tr:has(.health-legacy) td {
    color: #ffaaaa !important;
}

tr:has(.health-legacy) input[type="checkbox"],
tr:has(.health-nested) input[type="checkbox"] {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
    accent-color: #666 !important;
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
            gr.update(interactive=False),                # addr_input
            gr.update(interactive=False),                # dest (destination textbox)
            gr.update(interactive=False),                # strategy dropdown
            gr.update(interactive=False),                # dust slider
            gr.update(interactive=False),                # fee_rate_slider
            gr.update(interactive=False),                # future_fee_slider
            gr.update(interactive=False),                # thank_you_slider
            gr.update(interactive=False),                # offline_toggle
			gr.update(interactive=False),                # theme_toggle
            gr.update(interactive=False),                # manual_utxo_input
			gr.update(interactive=False),                # economy_btn
            gr.update(interactive=False),                # hour_btn
            gr.update(interactive=False),                # halfhour_btn
            gr.update(interactive=False),                # fastest_btn
        )

   
    # =================================================================
    # ========================= UI STARTS HERE ========================
    # =================================================================
    with gr.Column():
        theme_js = gr.HTML("")

         # ← Modern Bitcoin Optimization Note (after toggle, before inputs)
        gr.HTML(
            value="""
            <div style="
                margin: 0px auto 50px auto !important;
                padding: 28px !important;
                max-width: 900px !important;
                background: rgba(0, 20, 10, 0.6) !important;
                border: 3px solid #00ff9d !important;
                border-radius: 18px !important;
                text-align: center !important;
                font-size: 1.2rem !important;
                line-height: 1.8 !important;
                color: #ccffe6 !important;
                box-shadow: 0 0 60px rgba(0, 255, 157, 0.4) !important;
            ">
                <div style="
                    color:#00ffdd !important;
                    font-size:1.6rem !important;
                    font-weight:900 !important;
                    letter-spacing:2px !important;
                    margin-bottom:16px !important;
                    text-shadow:0 0 30px #00ffdd !important;
                ">
                    Optimized for Modern Bitcoin
                </div>

                Ωmega Pruner fully supports <strong style="color:#00ffff !important;font-weight:900 !important;text-shadow:0 0 20px #00ffff !important;">Native SegWit (bc1q...)</strong> and <strong style="color:#00ffff !important;font-weight:900 !important;text-shadow:0 0 20px #00ffff !important;">Taproot (bc1p...)</strong> inputs for maximum privacy and lowest fees.<br><br>
        
            Legacy addresses (starting with <strong style="color:#ffaa00 !important;font-weight:900 !important;">1...</strong>) and Nested SegWit (starting with <strong style="color:#ffaa00 !important;font-weight:900 !important;">3...</strong>) are displayed for transparency but 
            <strong style="color:#ff6666 !important;font-weight:900 !important;">cannot be included in the generated PSBT</strong>.<br><br>
        
            They will appear faded in the table and cannot be selected.<br>
            To fully take advantage of optimized fees and better privacy, we recommend spending or converting them separately.
        </div>
            """
        )
        mode_status = gr.HTML("")  # ← Empty placeholder — will be filled dynamically
        
        with gr.Row():
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
                label="Destination (optional • Payjoin enabled)",
                placeholder="Paste address or full invoice (PayJoin)",
                info=(
                    "Your pruned coins return here.<br>"
                    "• Blank = back to original scanned address<br>"
                    "• Paste a <strong>full invoice</strong> → PayJoin detection (zero new CIOH)<br>"
                    "• Required for xpub scans"
                ),
                scale=1,
            )

        # === ADVANCED PRIVACY HEADER (full width) ===
        gr.HTML(
        	value="""
        <div style="
            text-align:center;
            padding:20px !important;
        background:#001100 !important;
        border:3px solid #00ff88 !important;
        border-radius:16px !important;
        box-shadow:
            0 12px 40px rgba(0,0,0,0.6) !important,
            0 8px 32px rgba(0,255,136,0.4) !important,
            inset 0 0 30px rgba(0,255,136,0.2) !important;
    "> 
      <div style="
          color:#00ff88 !important;
          font-size:1.6rem !important;
          font-weight:900 !important;
          text-shadow:0 0 25px #00ff88 !important;
      ">
        🔒 Air-Gapped / Offline Mode
      </div>

      <div style="
          margin-top:12px !important;
          color:#aaffcc !important;
          font-size:1.1rem !important;
          line-height:1.6 !important;
      ">
        Fully offline operation — no API calls, perfect for cold wallets.<br>
        Paste raw UTXOs.
      </div>
    </div>
    """
    )
        
        # === OFFLINE MODE — FIRST ADVANCED TOOL ===
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                offline_toggle = gr.Checkbox(
                    label="🔒 Offline / Air-Gapped Mode",
                    value=False,
                    interactive=True,
                    info="No API calls • Paste raw UTXOs • True cold wallet prep",
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
            theme_icon = "🌙" if dark else "☀️"
            theme_text = "Dark" if dark else "Light"
            connection = "Offline 🔒 • No API calls • Fully air-gapped" if offline else "Online • API calls enabled"

            # Inline everything — overrides all inherited styles
            color = "#00ff88" if dark else "#006644"
            text_shadow = "0 0 20px #00ff88" if dark else "none"
            bg = "rgba(0, 30, 0, 0.5)" if dark else "rgba(200, 255, 220, 0.35)"

            return f"""
            <div style="
                text-align: center;
                padding: 16px;
                margin: 8px 0;
                font-size: 1.4rem;
                font-weight: 900;
                color: {color};
                text-shadow: {text_shadow};
                background: {bg};
                border-radius: 16px;
                box-shadow: 0 12px 40px rgba(0,0,0,0.5),
                            0 8px 32px rgba(0,255,136,0.4),
                            inset 0 0 20px rgba(0,255,136,0.3);
                transition: all 0.4s ease;
            ">
                <span style="font-size: 1.6rem; margin-right: 8px;">{theme_icon}</span>
                {theme_text} • {connection}
            </div>
            """

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
        )

        offline_toggle.change(
            fn=update_status_and_ui,
            inputs=[offline_toggle, theme_toggle],
            outputs=mode_status
        )

        theme_toggle.change(
            fn=update_status_and_ui,
            inputs=[offline_toggle, theme_toggle],
            outputs=mode_status,
            js="""
            (offline, dark) => {
                if (dark) {
                    document.body.classList.add("dark-mode");
                } else {
                    document.body.classList.remove("dark-mode");
                }
            }
            """
        )
        
		# Transition banner after Silent Payments
        gr.HTML(
            value="""
     <div style="
        text-align:center;
        padding:20px !important;
        margin:0px 0 0px 0 !important;
        background: linear-gradient(90deg, rgba(0,50,0,0.55), rgba(0,30,0,0.65)) !important;
        border:3px solid #00ff88 !important;
        border-radius:16px !important;
        box-shadow:
            0 12px 40px rgba(0,0,0,0.6) !important,
            0 8px 32px rgba(0,255,136,0.3) !important,
            inset 0 0 30px rgba(0,255,136,0.15) !important;
    "> 
      <div style="
          color:#00ff88 !important;
          font-size:1.5rem !important;
          font-weight:900 !important;
          text-shadow:0 0 25px #00ff88 !important;
      ">
        Pruning Strategy & Economic Controls
      </div>

      <div style="
          margin-top:12px !important;
          color:#aaffcc !important;
          font-size:1.1rem !important;
          line-height:1.6 !important;
          text-shadow:0 2px 4px rgba(0,0,0,0.8) !important;
      ">
        Choose how aggressive your prune will be — and fine-tune fees & donations below
      </div>
    </div>
    """
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
                0, 100, value=5.0, step=0.1, label="Thank You / DAO Donation (%)", 
                info="Applied to amount remaining after miner fee. 100% = full donation (no change output). Below ~546 sats will be absorbed into fee.", 
                scale=2,
            )

        # Fee preset buttons
        with gr.Row():
            economy_btn = gr.Button("Economy", size="sm", elem_classes="fee-btn")
            hour_btn = gr.Button("1 hour", size="sm", elem_classes="fee-btn")
            halfhour_btn = gr.Button("30 min", size="sm", elem_classes="fee-btn")
            fastest_btn = gr.Button("Fastest", size="sm", elem_classes="fee-btn")

        analyze_btn = gr.Button("1. ANALYZE & LOAD UTXOs", variant="primary")
        # States (invisible)
        dest_value = gr.State("")
        scan_source = gr.State("")
        enriched_state = gr.State([])
        locked = gr.State(False)
        psbt_snapshot = gr.State(None)
        locked_badge = gr.HTML("")  # Starts hidden
        selection_snapshot_state = gr.State({})
        warning_banner = gr.HTML(label="Input Compatibility Notice", visible=True)
        selected_utxos_for_psbt = gr.State([])

        # Capture destination changes for downstream use
        dest.change(
            fn=lambda x: x.strip() if x else "",
            inputs=dest,
            outputs=dest_value
        )

        # Import file last
        import_file = gr.File(
            label="Restore Previous Ωmega Selection: Upload your saved .json file",
            file_types=[".json"],
            type="filepath",
            visible=False,
        )

        gr.HTML("""
		    <div style="width: 100%; margin-top: 25px;"></div>
<div class="check-to-prune-header">
    <div class="header-title">CHECK TO PRUNE</div>
    <div class="header-subtitle">Pre-checked = recommended • OPTIMAL = ideal • DUST/HEAVY = prune</div>
</div>

<style>
.check-to-prune-header {
    text-align: center;
    margin-bottom: 8px;
}

/* Dark mode */
.dark-mode .check-to-prune-header .header-title {
    color: #00ff88;
    font-size: 1.3rem;
    font-weight: 900;
    text-shadow: 0 0 20px #00ff88;
    letter-spacing: 1px;
}

.dark-mode .check-to-prune-header .header-subtitle {
    color: #aaffaa;
    font-size: 1.05rem;
    margin-top: 8px;
}

/* Light mode — softer, readable colors */
body:not(.dark-mode) .check-to-prune-header .header-title {
    color: #008844;
    font-size: 1.3rem;
    font-weight: 900;
    letter-spacing: 1px;
}

body:not(.dark-mode) .check-to-prune-header .header-subtitle {
    color: #006633;
    font-size: 1.05rem;
    margin-top: 8px;
}
</style>
""")

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
            label=" ",
            static_columns=[1, 2, 3, 4, 5, 6, 7],
            column_widths=["90px", "160px", "200px", "120px", "140px", "160px", "130px", "90px", "80px"]
        )

        status_output = gr.HTML("")
        # Generate row — hidden until analysis complete
        with gr.Row(visible=False) as generate_row:
            gen_btn = gr.Button(
                "2. GENERATE NUCLEAR PSBT",
                variant="primary",
                elem_id="generate-btn"
            )

        # PSBT output — placed right below the generate row
        psbt_output = gr.HTML("")

        # Export sections
        with gr.Row(visible=False) as export_title_row:
            gr.HTML("""
            <div style="text-align:center;padding:40px 0 30px 0;">

  <!-- Main Header — FROZEN = icy blue theme -->
  <div style="color:#00ddff;font-size:2.6rem;font-weight:900;
              letter-spacing:8px;
              text-shadow:0 0 40px #00ddff, 0 0 80px #00ddff,
                          0 4px 8px #000, 0 8px 20px #000000ee,
                          0 12px 32px #000000cc;
              margin-bottom:20px;">
    🔒 SELECTION FROZEN
  </div>
  
  <!-- Core message — signature green -->
  <div style="color:#aaffaa;font-size:1.4rem;font-weight:700;
              text-shadow:0 0 20px #0f0,
                          0 3px 6px #000, 0 6px 16px #000000dd,
                          0 10px 24px #000000bb;
              max-width:720px;margin:0 auto 16px auto;
              line-height:1.6;">
    Your pruning intent is now immutable • Permanent audit trail secured
  </div>
  
  <!-- Extra reassurance — bright cyan -->
  <div style="color:#00ffdd;font-size:1.1rem;opacity:0.9;font-weight:700;
              text-shadow:0 2px 4px #000, 0 4px 12px #000000cc, 0 8px 20px #000000aa;
              max-width:640px;margin:20px auto 10px auto;line-height:1.7;">
    The file below includes:<br>
    All selected UTXOs • Ω fingerprint • Transaction parameters
  </div>
  
  <div style="color:#aaffaa;font-size:1.1rem;opacity:0.9;font-weight:700;
              text-shadow:0 2px 4px #000, 0 4px 12px #000000cc, 0 8px 20px #000000aa;
              max-width:640px;margin:0 auto 40px auto;line-height:1.7;">
    Download for backup, offline verification, or future reference
  </div>

</div>
""")

        with gr.Row(visible=False) as export_file_row:
            export_file = gr.File(
                label="",
                interactive=False
            )

        with gr.Column():  # ← Extra indent level for consistency
            reset_btn = gr.Button("NUCLEAR RESET — START OVER — NO FUNDS AFFECTED", variant="secondary")
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
            outputs=[fee_rate_slider, gr.State()],  # current fee rate slider
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
        outputs=[df, warning_banner], 
    ).then(
        fn=generate_summary_safe,
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row]
    )

    # =============================
    # — ANALYZE BUTTON (pure data loading + affordances) —
    # =============================
    analyze_btn.click(
        fn=analyze_and_show_summary,
        inputs=[
            addr_input,
            strategy,
            dust,
            fee_rate_slider,
            thank_you_slider,
            future_fee_slider,
            offline_toggle,
            manual_utxo_input,
			locked,
			dest_value,
        ],
        outputs=[
            df,
            enriched_state,
            warning_banner,
            generate_row,
            import_file,
            scan_source,
			status_output,
        ],
    )

    # =============================
    # — GENERATE BUTTON (pure execution + PSBT render) —
    # =============================
    gen_btn.click(
        fn=on_generate,
        inputs=[
            dest_value,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            enriched_state,
            scan_source,
        ],
        outputs=[psbt_snapshot, selected_utxos_for_psbt, locked, export_file],
    ).then(
        fn=generate_psbt,
        inputs=[psbt_snapshot, selected_utxos_for_psbt],
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
            addr_input,
            dest,
            strategy,
            dust,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            offline_toggle,
			theme_toggle,
            manual_utxo_input,
            economy_btn,
            hour_btn,
            halfhour_btn,
            fastest_btn,
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
            gr.update(value=""),                                     # warning_banner
            gr.update(visible=True),                                 # analyze_btn — show
            gr.update(visible=False),                                # generate_row — hide
            None,                                                    # psbt_snapshot — wipe
            False,                                                   # locked — unlock
            "",                                                      # locked_badge — clear
            gr.update(value="", interactive=True),                   # addr_input
            gr.update(value="", interactive=True),                   # dest_value — ENABLE + clear
            gr.update(interactive=True),                             # strategy
            gr.update(interactive=True),                             # dust
            gr.update(interactive=True),                             # fee_rate_slider
            gr.update(interactive=True),                             # future_fee_slider
            gr.update(interactive=True),                             # thank_you_slider
            gr.update(value=False, interactive=True),                # offline_toggle
            gr.update(value="", visible=False, interactive=True),    # manual_utxo_input
            gr.update(interactive=True),                             # theme_toggle — RE-ENABLE DARK MODE
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
            warning_banner,
            analyze_btn,
            generate_row,
            psbt_snapshot,
            locked,
            locked_badge,
            addr_input,
            dest,
            strategy,
            dust,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            offline_toggle,
            manual_utxo_input,
            theme_toggle,
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
        inputs=[df, enriched_state, fee_rate_slider, future_fee_slider, thank_you_slider, locked, strategy, dest_value,],
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
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row],
    )

    fee_rate_slider.change(
        fn=generate_summary_safe,
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row],
    )

    future_fee_slider.change(
        fn=generate_summary_safe,
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row],
    )

    thank_you_slider.change(
        fn=generate_summary_safe,
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row],
    )

    demo.load(
        fn=generate_summary_safe,
        inputs=[
            df,
            enriched_state,
            fee_rate_slider,
            future_fee_slider,
            thank_you_slider,
            locked,
            strategy,
            dest_value,
        ],
        outputs=[status_output, generate_row]
    )

    demo.load(
        fn=lambda: update_status_and_ui(False, True),
        outputs=mode_status,
        js="""
        () => {
            document.body.classList.add("dark-mode");
        }
        """
    )
    
    # CRITICAL: DO NOT use .change() on fee_rate/future_fee for anything else
    # 5. FOOTER
    gr.HTML(
    """
    <div style="width: 100%; margin-top:70px;"></div>

    <div style="
        width: 100%;
        max-width: 720px;
        margin: 0 auto 30px auto;
        text-align: center;
        line-height: 1.8;
    ">
        <!-- VERSION -->
        <div style="
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #f7931a;
            text-shadow: 0 0 12px rgba(247,147,26,0.65);
        ">
            Ωmega Pruner v11 — Forged Anew
        </div>

        <!-- GITHUB LINK -->
        <a href="https://github.com/babyblueviper1/Viper-Stack-Omega"
           target="_blank"
           rel="noopener"
           style="
               font-size: 0.94rem;
               font-weight: 600;
               text-decoration: none;
               color: #f7931a;
               text-shadow: 0 0 10px rgba(247,147,26,0.55);
           ">
            GitHub • Open Source • Apache 2.0
        </a>

        <br><br>

        <!-- CUSTOM BUILDS SECTION -->
        <div style="margin: 20px auto; max-width: 680px;">
            <a href="https://www.babyblueviper.com/p/omega-pruner-custom-builds"
               target="_blank"
               style="color: inherit; text-decoration: none;">
                   <div style="
            display: inline-block;
            padding: 6px 16px;
            margin: 8px 0;
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            border-radius: 12px;
            transition: all 0.4s ease;
            color: #00ff9d;
            background: rgba(0, 40, 20, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 157, 0.3);
        ">
            This build is engineered for speed and clarity.
        </div>
        <br>
        <div style="
            display: inline-block;
            padding: 6px 16px;
            margin: 8px 0;
            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            border-radius: 12px;
            transition: all 0.4s ease;
            color: #00ff88;
            background: rgba(0, 35, 15, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
        ">
            For extended capabilities or tailored integrations, custom versions can be commissioned.
        </div>
    </a>
</div>

<style>
/* Light mode — switch to forest green with light background */
body:not(.dark-mode) div[style*="00ff9d"], 
body:not(.dark-mode) div[style*="00ff88"] {
    color: #004d33 !important;
    background: rgba(220, 255, 235, 0.15) !important;  /* almost invisible */
    box-shadow: 0 2px 8px rgba(0, 80, 50, 0.1) !important;
}

/* Hover effect */
a:hover div[style*="padding: 6px 16px"] {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 255, 136, 0.3);
}
</style>

        <br><br>

        <!-- TAGLINE -->
        <span style="
            color: #0f0;
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.6px;
            text-shadow:
                0 0 12px #0f0,
                0 0 24px #0f0,
                0 0 6px #000,
                0 4px 10px #000,
                0 8px 20px #000000e6;
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
