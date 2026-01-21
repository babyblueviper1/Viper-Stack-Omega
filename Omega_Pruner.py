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
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "<div class='empty-state-msg' style='"
    "text-align:center !important;"
    "padding: clamp(40px, 10vw, 80px) !important;"
    "max-width:90% !important;"
    "margin:0 auto !important;"
    "background: rgba(0, 20, 10, 0.7) !important;"  # Dark green translucent background
    "border: 2px solid #00ff88 !important;"
    "border-radius: 18px !important;"
    "box-shadow: 0 0 50px rgba(0,255,136,0.4) !important;"
    "'>"
    "<div style='"
    "color:#00ffdd !important;"
    "font-size: clamp(1.3rem, 5.5vw, 1.8rem) !important;"
    "font-weight:900 !important;"
    "text-shadow: 0 0 30px #00ffdd !important;"
    "margin-bottom: clamp(16px, 4vw, 24px) !important;"
    "'>"
    "No UTXOs found yet"
    "</div>"
    "<div style='"
    "color:#88ffcc !important;"
    "font-size: clamp(1rem, 3.5vw, 1.2rem) !important;"
    "line-height:1.7 !important;"
    "'>"
    "Paste one or more addresses/xpubs below,<br>"
    "lower the dust threshold, or use Offline Mode to paste raw UTXOs."
    "</div>"
    "</div>"
)

select_msg = (
    "<div class='empty-state-msg' style='"
    "text-align:center !important;"
    "padding: clamp(40px, 10vw, 80px) !important;"
    "max-width:90% !important;"
    "margin:0 auto !important;"
    "background: rgba(0, 20, 10, 0.7) !important;"
    "border: 2px solid #00ff88 !important;"
    "border-radius: 18px !important;"
    "box-shadow: 0 0 50px rgba(0,255,136,0.4) !important;"
    "'>"
    "<div style='"
    "color:#00ffdd !important;"
    "font-size: clamp(1.3rem, 5.5vw, 1.8rem) !important;"
    "font-weight:900 !important;"
    "text-shadow: 0 0 30px #00ffdd !important;"
    "margin-bottom: clamp(16px, 4vw, 24px) !important;"
    "'>"
    "Select UTXOs to begin pruning"
    "</div>"
    "<div style='"
    "color:#88ffcc !important;"
    "font-size: clamp(1rem, 3.5vw, 1.2rem) !important;"
    "line-height:1.7 !important;"
    "'>"
    "Check the boxes — summary and privacy score update instantly."
    "</div>"
    "</div>"
)

DERIVATION_PROFILES = {
    "p2pkh":       "m/44'/0'/0'",
    "p2sh-p2wpkh": "m/49'/0'/0'",
    "p2wpkh":      "m/84'/0'/0'",
    "p2tr":        "m/86'/0'/0'",
}


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

def safe_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """Robust GET with retries, backoff, and proper timeout support."""
    for attempt in range(3):
        try:
            print(f">>> safe_get attempt {attempt+1}: {url} (timeout={timeout}s)")
            r = session.get(url, timeout=timeout)
            print(f">>> Response: {r.status_code}")
            if r.status_code == 200:
                return r
            elif r.status_code == 429:  # Rate limit
                sleep_time = 1.5 ** attempt
                print(f">>> Rate limited (429) — sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f">>> Bad status {r.status_code}")
        except Exception as e:
            print(f">>> Request failed (attempt {attempt+1}): {type(e).__name__}: {e}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))  # Short backoff
    print(f">>> safe_get failed after 3 attempts: {url}")
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

def get_prune_score():
    """Dynamically calculate Pruning Conditions using mempool.space 24h, 1w, and 1m block fee-rates."""
    urls = {
        '24h': "https://mempool.space/api/v1/mining/blocks/fee-rates/24h",
        '1w': "https://mempool.space/api/v1/mining/blocks/fee-rates/1w",
        '1m': "https://mempool.space/api/v1/mining/blocks/fee-rates/1m"
    }
    avgs = {}
    try:
        for period, url in urls.items():
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if len(data) > 5:
                    p50_fees = [block.get('avgFee_50', 1) for block in data if 'avgFee_50' in block]
                    if p50_fees:
                        avgs[period] = statistics.mean(p50_fees)
                    else:
                        avgs[period] = None
                else:
                    avgs[period] = None
            else:
                avgs[period] = None
    except Exception as e:
        log.warning(f"Failed to fetch fee-rates: {e}")
        avgs = {'24h': None, '1w': None, '1m': None}

    # Current economy fee
    current_fees = get_live_fees() or {'economy': 1}
    current = current_fees['economy']

    # Primary avg: 1w preferred
    primary_avg = avgs['1w'] or avgs['1m'] or avgs['24h'] or 5.0
    day_avg = avgs['24h'] or primary_avg
    month_avg = avgs['1m'] or primary_avg

    # Score calculation
    ratio = current / primary_avg if primary_avg > 0 else 1.0
    if ratio < 0.5:
        level = "Excellent"
        color = "#00ff88"
        score = 10
    elif ratio < 0.8:
        level = "Good"
        color = "#00ffdd"
        score = 8
    elif ratio < 1.2:
        level = "Fair"
        color = "#ff9900"
        score = 5
    else:
        level = "Poor"
        color = "#ff3366"
        score = 3
    score = max(1, min(10, round(score + (1 - ratio) * 2)))

    # Plural handling for current fee
    fee_unit = "sat/vB" if current == 1 else "sats/vB"

    # Fetch price & block height
    price = "—"
    height = "—"
    try:
        height_r = session.get("https://mempool.space/api/blocks/tip/height", timeout=8)
        height = height_r.text.strip() if height_r.status_code == 200 else "—"
        price_r = session.get("https://mempool.space/api/v1/prices", timeout=8)
        price_usd = price_r.json().get('USD', 0) if price_r.status_code == 200 else 0
        price = f"${price_usd:,}" if price_usd > 0 else "—"
    except Exception:
        pass

    # Hashrate (try 1w first, fallback to 3d for more reliability)
    hr_formatted = "—"
    try:
        for hr_period in ["1w", "3d"]:
            hr_r = session.get(f"https://mempool.space/api/v1/mining/hashrate/{hr_period}", timeout=10)
            if hr_r.status_code == 200:
                hr_data = hr_r.json()
                if isinstance(hr_data.get('currentHashrate'), dict):
                    current_hashrate = hr_data['currentHashrate'].get('avgHashrate')
                else:
                    current_hashrate = hr_data.get('currentHashrate')
                
                if isinstance(current_hashrate, (int, float)) and current_hashrate > 0:
                    if current_hashrate > 1e18:
                        hr_formatted = f"{current_hashrate / 1e18:.0f} EH/s"
                    elif current_hashrate > 1e15:
                        hr_formatted = f"{current_hashrate / 1e15:.0f} PH/s"
                    else:
                        hr_formatted = f"{current_hashrate / 1e12:.0f} TH/s"
                    break
    except Exception:
        pass

    # Next Difficulty Adjustment
    adjustment_text = ""
    try:
        da_r = session.get("https://mempool.space/api/v1/difficulty-adjustment", timeout=8)
        if da_r.status_code == 200:
            da_data = da_r.json()
            adjustment_pct = da_data.get('difficultyChange', 0)
            blocks_remaining = da_data.get('remainingBlocks', 0)
            days_remaining = round(blocks_remaining / 144) if blocks_remaining else 0
            adjustment_text = f"Next adj: {adjustment_pct:+.2f}% in ~{days_remaining} days"
    except Exception:
        adjustment_text = ""

    # Halving countdown
    halving_text = ""
    try:
        if height != "—":
            current_height = int(height)
            next_halving = ((current_height // 210000) + 1) * 210000
            blocks_to_halving = next_halving - current_height
            days_to_halving = round(blocks_to_halving / 144)
            halving_text = f"Halving: ~{days_to_halving:,} days"
    except Exception:
        halving_text = ""

    # Build context line with responsive breaks
    context_line = f"BTC: {price} • Block: {height}<br>Hashrate: {hr_formatted}"
    if adjustment_text:
        context_line += f" • {adjustment_text}"
    if halving_text:
        context_line += f"<br>{halving_text}"

    return f"""
    <div style="
        text-align:center !important;
        padding: clamp(20px, 5vw, 30px) !important;
        margin: clamp(20px, 5vw, 40px) auto !important;
        background: rgba(0, 20, 10, 0.7) !important;
        border: 3px solid {color} !important;
        border-radius: 20px !important;
        box-shadow: 0 0 60px {color} !important;
        max-width: 600px !important;
    ">
        <div style="
            color: {color} !important;
            font-size: clamp(1.6rem, 6vw, 2rem) !important;
            font-weight: 900 !important;
            text-shadow: 0 0 35px {color} !important;
            margin-bottom: 8px !important;
        ">
            Pruning Conditions: {level} ({score}/10)
        </div>

        <!-- Current Economy Fee -->
        <div style="
            color: #00ffff !important;
            font-size: clamp(2.2rem, 8vw, 3rem) !important;
            font-weight: 900 !important;
            text-shadow: 0 0 40px #00ffff, 0 0 80px #00ffff !important;
            margin: 16px 0 !important;
        ">
            {current} {fee_unit}
        </div>
        <div style="
            color: #88ffcc !important;
            font-size: clamp(1rem, 3.8vw, 1.3rem) !important;
            margin-bottom: 20px !important;
        ">
            Current economy fee
        </div>

        <!-- VS -->
        <div style="
            color: #f7931a !important;
            font-size: clamp(1.4rem, 5vw, 1.8rem) !important;
            font-weight: 900 !important;
            text-shadow: 0 0 30px #f7931a !important;
            margin: 20px 0 !important;
            letter-spacing: 2px !important;
        ">
            VS
        </div>

        <!-- Medians Row -->
        <div style="
            display: flex !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
            gap: clamp(16px, 4vw, 32px) !important;
            margin: 16px 0 !important;
            color: #aaffcc !important;
            font-size: clamp(1rem, 3.8vw, 1.3rem) !important;
            font-weight: 700 !important;
        ">
            <!-- 1-day -->
            <div style="text-align: center !important;">
                <div style="color: #00ddff !important; font-size: clamp(1.3rem, 5vw, 1.7rem) !important; font-weight: 900 !important;">
                    {day_avg:.1f} sats/vB
                </div>
                <div style="color: #88ccff !important; font-size: clamp(0.9rem, 3.2vw, 1.1rem) !important;">
                    1-day median
                </div>
            </div>

            <!-- 1-week -->
            <div style="text-align: center !important;">
                <div style="color: #00ff88 !important; font-size: clamp(1.4rem, 5.5vw, 1.8rem) !important; font-weight: 900 !important; text-shadow: 0 0 20px #00ff88 !important;">
                    {primary_avg:.1f} sats/vB
                </div>
                <div style="color: #88ffaa !important; font-size: clamp(0.95rem, 3.4vw, 1.15rem) !important;">
                    1-week median
                </div>
            </div>

            <!-- 1-month -->
            <div style="text-align: center !important;">
                <div style="color: #aaff88 !important; font-size: clamp(1.3rem, 5vw, 1.7rem) !important; font-weight: 900 !important;">
                    {month_avg:.1f} sats/vB
                </div>
                <div style="color: #99ffbb !important; font-size: clamp(0.9rem, 3.2vw, 1.1rem) !important;">
                    1-month median
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div style="
            color: #88ffaa !important;
            font-size: clamp(0.9rem, 3vw, 1rem) !important;
            margin-top: 20px !important;
            font-weight: 700 !important;
            line-height: 1.6 !important;
        ">
            Lower than medians = prime time to prune dust & consolidate! 🔥<br>
            <small style="color: #66cc99 !important; font-weight: normal !important;">
                Data from mempool.space mining stats<br>
                {context_line}
            </small>
        </div>
    </div>
    """

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
        
from typing import Tuple, Any

def load_selection(parsed_snapshot: dict, current_enriched: Any) -> Tuple[Any, str]:
    print(">>> load_selection CALLED - snapshot type:", type(parsed_snapshot))

    if not parsed_snapshot or not isinstance(parsed_snapshot, dict):
        return current_enriched, "No valid parsed JSON loaded"

    print(">>> Using pre-parsed snapshot - inputs count:", len(parsed_snapshot.get("inputs", [])))

    try:
        # No need for json.loads anymore — it's already parsed!
        snapshot = parsed_snapshot

        if "inputs" not in snapshot:
            print(">>> Invalid snapshot format")
            return current_enriched, "Invalid Ωmega Pruner selection file"

        # Case-insensitive + safe vout conversion
        selected_keys = set()
        for u in snapshot.get("inputs", []):
            if not isinstance(u, dict) or "txid" not in u or "vout" not in u:
                continue
            try:
                txid = str(u["txid"]).lower().strip()
                vout = int(u["vout"])
                selected_keys.add((txid, vout))
            except (ValueError, TypeError):
                print(">>> Skipping invalid input in JSON:", u)
                continue

        print(">>> Selected keys from JSON (case-insensitive):", len(selected_keys))

        # Normalize current_enriched to utxos list — critical fix!
        if isinstance(current_enriched, tuple) and len(current_enriched) == 2:
            meta, utxos = current_enriched
            utxos = list(utxos)  # make mutable copy for safety
            print(">>> Using frozen state: meta present, utxos len =", len(utxos))
        else:
            meta = {}
            utxos = list(current_enriched or [])
            print(">>> Using raw list state: utxos len =", len(utxos))

        # Early exit if no UTXOs to restore into
        if not utxos:
            print(">>> current_enriched is empty - cannot restore yet")
            return (), (
                "<div style='color:#aaffcc !important;padding:30px;background:#001100 !important;border:2px solid #00ff88 !important;border-radius:16px !important;text-align:center !important;'>"
                "<span style='color:#00ffdd !important;font-size:1.6rem;font-weight:900 !important;'>Selection file loaded!</span><br><br>"
                "<strong>Table is empty — restore will happen after:</strong><br>"
                "1. Paste the same addresses/xpubs<br>"
                "2. Click ANALYZE (table must load first)<br>"
                "3. Upload JSON again — checkboxes will restore<br><br>"
                "If table stays empty, check your address input."
                "</div>"
            )

        # Restore loop — now safe and case-insensitive
        updated = []
        matched_count = 0
        for u in utxos:
            new_u = dict(u)
            txid_lower = str(u.get("txid", "")).lower().strip()
            vout = u.get("vout")
            try:
                vout = int(vout) if vout is not None else None
            except (ValueError, TypeError):
                vout = None

            is_selected = (txid_lower, vout) in selected_keys if vout is not None else False
            new_u["selected"] = is_selected
            if is_selected:
                matched_count += 1
            updated.append(new_u)

        print(">>> Restore complete - matched:", matched_count, "out of", len(selected_keys))

        # Build return tuple consistently
        return_tuple = (meta, tuple(updated)) if meta else tuple(updated)

        if matched_count == 0:
            message = (
                "<div style='color:#ffddaa !important;padding:30px;background:#332200 !important;border:2px solid #ff9900 !important;border-radius:16px !important;text-align:center !important;'>"
                "<span style='color:#ffff66 !important;font-size:1.6rem;font-weight:900 !important;'>Selection loaded — no matching UTXOs found</span><br><br>"
                f"File contains {len(selected_keys)} UTXOs.<br>"
                "They don't match current analysis (different addresses? UTXOs spent? txid case?).<br>"
                "Checkboxes not restored."
                "</div>"
            )
        else:
            message = (
                "<div style='color:#aaffff !important;padding:30px;background:#001122 !important;border:2px solid #00ffff !important;border-radius:16px !important;text-align:center !important;'>"
                f"<span style='color:#00ffff !important;font-size:1.6rem;font-weight:900 !important;'>Selection loaded — {matched_count}/{len(selected_keys)} UTXOs restored "
                f"({len(utxos)} total in current table)</span>"
                "</div>"
            )

        return return_tuple, message

    except Exception as e:
        print(">>> Processing ERROR:", str(e))
        return current_enriched, f"Failed to process selection: {str(e)}"

def rebuild_df_rows(enriched_state) -> tuple[List[List], bool]:
    """
    Rebuild dataframe rows from current enriched_state.
    Used when loading a saved selection JSON.
    Safely handles invalid input, enforces unsupported type rules,
    and uses 'is_legacy' flag to disable legacy UTXOs.
    TXID shows full value (copyable on click).
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

        script_type = u.get("script_type", "").strip()
        selected = bool(u.get("selected", False))  # Preserve from saved state
        inferred = bool(u.get("script_type_inferred", False))
        is_legacy = bool(u.get("is_legacy", False))  # NEW: from _enrich_utxos

        # Force legacy to be unselected (safety + clear UX)
        if is_legacy:
            selected = False
            has_unsupported = True

        # Supported only: Native SegWit and Taproot
        supported_in_psbt = script_type in ("P2WPKH", "Taproot", "P2TR")

        # ── Health badge HTML ─────────────────────────────────────────────────
        if not supported_in_psbt or is_legacy:
            has_unsupported = True

            if is_legacy or script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;background:rgba(255,68,68,0.12);padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ LEGACY</span><br>'
                    f'<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported for PSBT – migrate first</small>'
                    '</div>'
                )
            elif script_type in ("P2SH-P2WPKH", "Nested"):
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;background:rgba(255,153,0,0.12);padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ NESTED</span><br>'
                    f'<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported yet</small>'
                    '</div>'
                )
            else:
                health = u.get("health", "UNKNOWN")
                health_html = (
                    f'<div class="health health-{health.lower()}" style="padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{health}</span><br>'
                    f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">Cannot prune</small>'
                    '</div>'
                )
        else:
            # Modern & supported
            health = u.get("health", "OPTIMAL")
            recommend = u.get("recommend", "")
            health_html = (
                f'<div class="health health-{health.lower()}" style="padding:6px;border-radius:6px;">'
                f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{health}</span><br>'
                f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">{recommend}</small>'
                '</div>'
            )

        # ── Friendly display name ─────────────────────────────────────────────
        display_type_map = {
            "P2WPKH": "Native SegWit",
            "Taproot": "Taproot",
            "P2TR": "Taproot",
            "P2SH-P2WPKH": "Nested SegWit",
            "P2PKH": "Legacy",
            "Legacy": "Legacy",
        }
        display_type = display_type_map.get(script_type, script_type)

        if inferred:
            display_type += ' <span style="color:#00cc66;font-weight:bold;">[inferred]</span>'

        if is_legacy:
            display_type += ' <span style="color:#ff6666;font-weight:bold;">[legacy – disabled]</span>'

        txid_full = u.get("txid", "unknown")

        # Build row
        rows.append([
            selected,                           # PRUNE checkbox (forced off for legacy)
            u.get("source", "Single"),          # Includes derivation group if from xpub
            txid_full,
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

    min_mixes, max_mixes = estimate_coinjoin_mixes_needed(input_count, distinct_addrs, privacy_score)

    recovery_note = ""
    if privacy_score <= 70:
        recovery_note = f"""
        <div style="
            margin-top:20px !important;
            padding:16px !important;
            background:#001100 !important;
            border:2px solid #00ff88 !important;
            border-radius:12px !important;
            color:#aaffcc !important;
            font-size:clamp(0.95rem, 3.2vw, 1.05rem) !important;
            line-height:1.6 !important;
            box-shadow:0 0 40px rgba(0,255,136,0.4) !important;
        ">
            💧 <span style="
                color:#00ffdd !important;
                font-size:clamp(1rem, 3.5vw, 1.2rem)!important;
                font-weight:900 !important;
                text-shadow:0 0 20px #00ffdd !important;
            ">Recovery Plan</span>:<br>
            Break address linkage using transactions that involve other participants<br>
            <small style="color:#88ffcc !important;">
                (~{min_mixes}–{max_mixes} coordinated rounds typically needed)
            </small><br>
            <small style="color:#66ffaa !important;">
                Examples: CoinJoin (Whirlpool), PayJoin, Silent Payments
            </small>
        </div>
        """

    # EXTREME CIOH
    if privacy_score <= 30:
        return f"""
        <div style="
            margin-top:16px !important;
            padding:16px !important;
            background:#440000 !important;
            border:3px solid #ff3366 !important;
            border-radius:14px !important;
            box-shadow:0 0 50px rgba(255,51,102,0.9) !important;
            font-size:clamp(1rem, 3.5vw, 1.2rem) !important;
            line-height:1.7 !important;
            color:#ffcccc !important;
        ">
            <div style="color:#ff3366 !important; font-size:clamp(1.3rem, 5vw, 1.6rem) !important; font-weight:900 !important; text-shadow:0 0 30px #ff3366 !important;">
                EXTREME CIOH LINKAGE
            </div><br>
            <div style="color:#ff6688 !important; font-size:clamp(1rem, 3.5vw, 1.2rem)!important;">
                Common Input Ownership Heuristic (CIOH)
            </div><br>
            This consolidation strongly proves common ownership of many inputs and addresses.<br><br>
            <div style="color:#ffaaaa !important;">Privacy state: Severely compromised</div><br>
            Maximum fee efficiency — but analysts will confidently cluster these addresses as yours.<br><br>
            <div style="color:#ffbbbb !important;">
                <span style="font-weight:900 !important;">Best practices after this point:</span><br>
                • Do not consolidate these addresses again<br>
                • Avoid direct spending to KYC or identity-linked services<br>
                • Restore privacy only via transactions involving other participants
            </div>
            <div style="margin-top:10px;color:#ff9999 !important;font-size:0.95em;">
                Examples include CoinJoin, PayJoin, or Silent Payments —
                which require wallet support or coordination and cannot be added after the fact.
            </div>
            {recovery_note}
        </div>
        """

    # HIGH CIOH
    elif privacy_score <= 50:
        return f"""
        <div style="
            margin-top:14px !important;
            padding:14px !important;
            background:#331100 !important;
            border:2px solid #ff8800 !important;
            border-radius:12px !important;
            font-size:clamp(1rem, 3.5vw, 1.2rem)!important;
            line-height:1.6 !important;
            color:#ffddaa !important;
        ">
            <div style="color:#ff9900 !important; font-size:clamp(1.3rem, 5vw, 1.6rem) !important; font-weight:900 !important; text-shadow:0 0 30px #ff9900 !important;">
                HIGH CIOH RISK
            </div><br>
            <div style="color:#ffaa44 !important; font-size:clamp(1rem, 3.5vw, 1.2rem)!important;">
                Common Input Ownership Heuristic (CIOH)
            </div><br>
            Merging {input_count} inputs from {distinct_addrs} address(es) → analysts will cluster them as belonging to the same entity.<br><br>
            <div style="color:#ffcc88 !important;">Privacy state: Significantly reduced</div><br>
            Good fee savings, but a real privacy trade-off.<br>
            Further consolidation will worsen linkage — consider restoring privacy before your next spend.
            {recovery_note}
        </div>
        """

    # MODERATE CIOH
    elif privacy_score <= 70:
        return f"""
        <div style="
            margin-top:12px !important;
            padding:12px !important;
            background:#113300 !important;
            border:1px solid #00ff9d !important;
            border-radius:10px !important;
            color:#aaffaa !important;
            font-size:clamp(1rem, 3.5vw, 1.2rem) !important;
            line-height:1.6 !important;
        ">
            <div style="color:#00ff9d !important; font-size:clamp(1.3rem, 5vw, 1.6rem)!important; font-weight:900 !important; text-shadow:0 0 30px #00ff9d !important;">
                MODERATE CIOH
            </div><br>
            <div style="color:#66ffaa !important; font-size:clamp(0.95rem, 3.2vw, 1.1rem)!important;">
                Common Input Ownership Heuristic (CIOH)
            </div><br>
            Spending multiple inputs together creates some on-chain linkage.<br>
            Analysts may assume common ownership — but it is not definitive.<br><br>
            Privacy impact is moderate.<br>
            Avoid repeating this pattern if long-term privacy is a priority.
            {recovery_note}
        </div>
        """

    # LOW CIOH — now with light green box for consistency
    else:
        return f"""
        <div style="
            margin-top:10px !important;
            padding:12px !important;
            background:#001a00 !important;           /* very dark green, subtle */
            border:1px solid #00ff88 !important;     /* light green border */
            border-radius:10px !important;
            color:#aaffcc !important;
            font-size:clamp(0.95rem, 3.2vw, 1.05rem) !important;
            line-height:1.6 !important;
            box-shadow:0 0 20px rgba(0,255,136,0.2) !important;  /* soft green glow */
        ">
            <div style="
                color:#00ffdd !important;
                font-size:clamp(1.3rem, 4vw, 1.6rem) !important;
                font-weight:900 !important;
                text-shadow:0 0 20px #00ffdd !important;
            ">
                LOW CIOH IMPACT
            </div><br>
            <span style="color:#00ffdd !important; font-size:clamp(0.95rem, 3.2vw, 1.1rem)!important;">
                (Common Input Ownership Heuristic)
            </span><br><br>
            Few inputs spent together — minimal new linkage created.<br>
            Address separation remains strong.<br>
            Privacy preserved.
            {recovery_note}  <!-- now included if score barely >70 but still has note -->
        </div>
        """
		
def estimate_coinjoin_mixes_needed(input_count: int, distinct_addrs: int, privacy_score: int) -> tuple[int, int]:
    if privacy_score > 80:
        return 0, 1    # Truly minimal linkage
    if privacy_score > 70:
        return 1, 2    # Light optional hardening

    # Start with inverse of score, but smoother
    base = max(2, round(100 / (privacy_score + 10)))  # Avoids huge spikes at low scores

    # Stronger input count scaling
    if input_count >= 50:
        base += 5
    elif input_count >= 30:
        base += 4
    elif input_count >= 15:
        base += 3
    elif input_count >= 8:
        base += 2
    elif input_count >= 5:
        base += 1

    # Heavier distinct address penalty — revealing more unique addrs is worse
    if distinct_addrs >= 15:
        base += 5
    elif distinct_addrs >= 10:
        base += 4
    elif distinct_addrs >= 6:
        base += 3
    elif distinct_addrs >= 3:
        base += 2
    elif distinct_addrs > 1:
        base += 1

    min_mixes = max(1, base - 2)
    max_mixes = base + 2

    return min(12, min_mixes), min(18, max_mixes)  # Cap at realistic Whirlpool use

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

def bech32_polymod(values: List[int]) -> int:
    """Checksum calculation for Bech32 addresses."""
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
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
    """Convert Bitcoin address to script pubkey + metadata."""
    addr = (addr or "").strip().lower()
    if not addr:
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'unknown'}

    # Legacy P2PKH (1...)
    if addr.startswith('1'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x00:
                hash160 = dec[1:21]
                return b'\x76\xa9\x14' + hash160 + b'\x88\xac', {
                    'input_vb': 148,
                    'output_vb': 34,
                    'type': 'P2PKH'
                }
        except Exception:
            pass
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 148, 'output_vb': 34, 'type': 'invalid'}

    # P2SH (3...)
    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:
                hash160 = dec[1:21]
                return b'\xa9\x14' + hash160 + b'\x87', {
                    'input_vb': 91,
                    'output_vb': 32,
                    'type': 'P2SH'
                }
        except Exception:
            pass
        return b'\x00\x14' + b'\x00'*20, {'input_vb': 91, 'output_vb': 32, 'type': 'invalid'}

    # Bech32 / Bech32m (bc1...)
    if addr.startswith('bc1'):
        data_part = addr[4:]

        if any(c not in CHARSET for c in data_part):
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        data = [CHARSET.find(c) for c in data_part]
        if len(data) < 1:
            return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

        witness_version = data[0]

        # P2WPKH / P2WSH (v0, bc1q...)
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

        # Taproot (v1, bc1p...)
        if addr.startswith('bc1p') and witness_version == 1 and bech32m_verify_checksum('bc', data):
            prog = convertbits(data[1:], 5, 8, False)
            if prog and len(prog) == 32:
                return b'\x51\x20' + bytes(prog), {
                    'input_vb': 57,
                    'output_vb': 43,
                    'type': 'Taproot'
                }

        return b'\x00\x14' + b'\x00'*20, {'input_vb': 68, 'output_vb': 31, 'type': 'invalid'}

    # Fallback
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
    Build a hardware-signable PSBT (P2WPKH + Taproot).
    Works with minimal UTXO data (txid/vout/value/scriptPubKey).
    BIP32 derivation is optional — added only when available.
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

    def write_kv(psbt: bytearray, key_type: int, key_data: bytes, value: bytes):
        key = bytes([key_type]) + key_data
        psbt += encode_varint(len(key))
        psbt += key
        psbt += encode_varint(len(value))
        psbt += value

    def write_bip32_derivation(psbt: bytearray, pubkey: bytes, fingerprint: bytes, path_bytes: bytes):
        write_kv(psbt, 0x06, pubkey, fingerprint + path_bytes)

    # Validation — minimal requirements
    if not tx.tx_ins:
        raise ValueError("Transaction has no inputs")
    if not tx.tx_outs:
        raise ValueError("Transaction has no outputs")
    if len(tx.tx_ins) != len(utxos):
        raise ValueError(f"Input count mismatch: {len(tx.tx_ins)} vs {len(utxos)}")

    raw_tx = tx.serialize_unsigned()
    psbt = bytearray(b"psbt\xff")

    # Global: unsigned tx
    write_kv(psbt, 0x00, b"", raw_tx)
    psbt += b"\x00"  # end global

    # Input maps
    for u in utxos:
        stype = u.get("script_type", "").lower()
        if stype not in ("p2wpkh", "p2tr", "taproot"):
            raise ValueError(f"Unsupported script type: {stype}")

        # Required: Witness UTXO
        spk = u["scriptPubKey"]
        if isinstance(spk, str):
            spk = bytes.fromhex(spk)

        witness_utxo = (
            u["value"].to_bytes(8, "little") +
            encode_varint(len(spk)) +
            spk
        )
        write_kv(psbt, 0x01, b"", witness_utxo)

        # Optional BIP32 derivation (only if all fields present)
        if all(k in u for k in ["pubkey", "fingerprint", "full_derivation_path"]):
            try:
                pubkey = bytes.fromhex(u["pubkey"])
                fingerprint = bytes.fromhex(u["fingerprint"])
                path_str = u["full_derivation_path"].replace("m/", "")
                path_bytes = b""
                for p in path_str.split("/"):
                    hardened = p.endswith("'")
                    n = int(p.rstrip("'"))
                    if hardened:
                        n |= 0x80000000
                    path_bytes += n.to_bytes(4, "little")

                write_bip32_derivation(psbt, pubkey, fingerprint, path_bytes)
            except Exception as e:
                print(f"Warning: Failed to add BIP32 derivation: {e}")

        # Explicit SIGHASH_ALL
        write_kv(psbt, 0x03, b"", (1).to_bytes(4, "little"))

        psbt += b"\x00"  # end input map

    # Empty output maps
    for _ in tx.tx_outs:
        psbt += b"\x00"

    psbt += b"\x00"  # final separator

    return base64.b64encode(psbt).decode("ascii"), ""

# =========================
# UTXO Fetching Functions
# =========================

def get_utxos(addr: str, dust: int = 546) -> List[dict]:
    """Fetch UTXOs from multiple APIs with retries, timeouts, and rate limit backoff."""
    addr = addr.strip()
    if not addr:
        return []

    apis = [
        f"{MEMPOOL_API}/address/{addr}/utxo",
        BLOCKSTREAM_API.format(addr=addr),
        BITCOINER_API.format(addr=addr),
    ]

    for attempt in range(3):
        for url in apis:
            try:
                r = safe_get(url, timeout=10)  # ← Add timeout to prevent hangs
                if r and r.status_code == 200:
                    try:
                        data = r.json()
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
                        logger.warning(f"JSON parse error from {url}: {str(e)}")
                elif r and r.status_code == 429:
                    logger.warning(f"Rate limited on {url}, attempt {attempt+1}")
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.warning(f"HTTP {r.status_code} from {url}")
            except Exception as e:
                logger.warning(f"Request error on {url}: {str(e)}")
            time.sleep(1.0)  # Delay per request

        time.sleep(5)  # Wait between full API cycles

    logger.warning(f"No UTXOs after retries for {addr}")
    return []


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
    hw_support_toggle: bool = False
    xpub: str = ""
    base_path: str = "m/84'/0'/0'"

def _sanitize_analyze_inputs(
    addr_input,
    strategy,
    dust_threshold,
    fee_rate_slider,
    thank_you_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
    hw_support_toggle,      
    xpub_field,
    base_path_field,
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
        hw_support_toggle=bool(hw_support_toggle),          
        xpub=(xpub_field or "").strip() if hw_support_toggle else "",  # Only keep if enabled
        base_path=(base_path_field or "m/84'/0'/0'").strip() if hw_support_toggle else "m/84'/0'/0'",
    )
def _collect_manual_utxos(params: AnalyzeParams) -> List[Dict]:
    """
    Parse user-supplied offline UTXO lines into normalized dicts.
    Format: txid:vout:value_in_sats[:change_address]
    """
    if not params.manual_utxo_input:
        return []

    utxos = []
    skipped = 0

    for line_num, line in enumerate(params.manual_utxo_input.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(":")
        if len(parts) < 3:
            skipped += 1
            continue

        try:
            txid = parts[0].strip()
            vout = int(parts[1].strip())
            value = int(parts[2].strip())

            # Optional address for change output or labeling
            addr = parts[3].strip() if len(parts) >= 4 else "unknown (manual)"

            if value <= params.dust_threshold:
                continue

            # Get scriptPubKey + metadata from address (fallback if no addr)
            _, meta = address_to_script_pubkey(addr)
            input_weight = meta.get("input_vb", 68) * 4  # Safe fallback to P2WPKH

            utxos.append({
                "txid": txid,
                "vout": vout,
                "value": value,
                "address": addr,
                "input_weight": input_weight,
                "health": "MANUAL",
                "recommend": "REVIEW",
                "script_type": meta.get("type", "Manual"),
                "script_type_inferred": True,
                "source": "Manual Offline",
                "selected": False,
            })

        except (ValueError, Exception) as e:
            skipped += 1
            # Optional: log skipped line for debugging
            # print(f"Skipped line {line_num}: {line} → {e}")

    if skipped:
        print(f">>> Manual UTXO parse: skipped {skipped} invalid/malformed lines")

    return utxos


def _collect_online_utxos(params: AnalyzeParams) -> Tuple[List[Dict], str, List[Any]]:
    if not params.addr_input:
        return [], "No addresses provided", []

    utxos = []
    debug_lines = []

    entries = [e.strip() for e in params.addr_input.splitlines() if e.strip()]

    for entry in entries:
        if not entry.startswith(("bc1q", "bc1p", "1", "3")):
            debug_lines.append(f"Skipped invalid entry: {entry[:20]}...")
            continue

        entry_utxos = get_utxos(entry, params.dust_threshold)
        count = len(entry_utxos)

        if count > 0:
            # Use FULL entry as source label (no truncation)
            source_label = entry  # ← full address
            debug_lines.append(f"Address {entry}: {count} UTXOs found")  # full for debug

            for u in entry_utxos:
                u["source"] = source_label          # ← full value here
                u["address"] = entry

            utxos.extend(entry_utxos)
        else:
            debug_lines.append(f"Address {entry}: no UTXOs")

    debug = "\n".join(debug_lines) if debug_lines else "No valid addresses or UTXOs found"
    return utxos, debug, []
    
def _classify_utxo(value: int, input_weight: int, script_type_from_meta: str = "") -> Tuple[str, str, str, bool]:
    """
    Classify a UTXO based on weight + value.
    Returns: (script_type, health, recommendation, is_legacy)
    """
    # Step 1: Determine base script type from weight (fallback)
    if input_weight <= 228:
        script_type = "Taproot"
        default_health = "OPTIMAL"
        default_rec = "KEEP"
        is_legacy = False
    elif input_weight <= 272:
        script_type = "P2WPKH"
        default_health = "OPTIMAL"
        default_rec = "KEEP"
        is_legacy = False
    elif input_weight <= 364:
        script_type = "P2SH-P2WPKH"
        default_health = "MEDIUM"
        default_rec = "CAUTION"
        is_legacy = False
    else:
        script_type = "Legacy"
        default_health = "HEAVY"
        default_rec = "PRUNE"
        is_legacy = True

    # Step 2: Prefer script_type from metadata when available (more accurate)
    if script_type_from_meta and script_type_from_meta in ("Taproot", "P2WPKH", "P2SH-P2WPKH", "P2PKH", "Legacy"):
        script_type = script_type_from_meta
        is_legacy = script_type in ("P2PKH", "Legacy")

        # Re-assign defaults based on corrected type
        if script_type == "Taproot":
            default_health = "OPTIMAL"
            default_rec = "KEEP"
        elif script_type == "P2WPKH":
            default_health = "OPTIMAL"
            default_rec = "KEEP"
        elif script_type == "P2SH-P2WPKH":
            default_health = "MEDIUM"
            default_rec = "CAUTION"
        else:  # Legacy / P2PKH
            default_health = "HEAVY"
            default_rec = "PRUNE"

    # Step 3: Value-based overrides (highest priority)
    if value < 10_000:
        return script_type, "DUST", "PRUNE", is_legacy

    # Large legacy/nested UTXOs deserve caution
    if value > 100_000_000 and is_legacy:
        return script_type, "CAREFUL", "OPTIONAL", is_legacy

    # Fall back to base classification
    return script_type, default_health, default_rec, is_legacy

def _enrich_utxos(raw_utxos: list[dict], params: AnalyzeParams) -> list[dict]:
    """
    Enrich raw UTXOs with script metadata, weight, health, recommendation,
    scriptPubKey, Taproot internal key, and legacy flag.
    """
    enriched: list[dict] = []

    for u in raw_utxos:
        addr = u.get("address", "")

        # Get scriptPubKey + metadata from address
        script_pubkey, meta = address_to_script_pubkey(addr)
        input_weight = meta.get("input_vb", 68) * 4  # fallback to P2WPKH size

        # Prefer meta["type"] when available (more reliable than weight)
        script_type_from_meta = meta.get("type", "")

        # Classify with both weight and meta type
        script_type, health, recommend, is_legacy = _classify_utxo(
            u["value"], input_weight, script_type_from_meta
        )

        # Normalize script_type for consistency
        if script_type.upper() in ("TAPROOT", "P2TR"):
            script_type = "Taproot"

        # Taproot internal key (only if Taproot)
        tap_internal_key = None
        if script_type == "Taproot":
            candidate = meta.get("tap_internal_key")
            if candidate and len(candidate) == 32:
                tap_internal_key = candidate

        enriched.append({
            **u,
            "input_weight": input_weight,
            "health": health,
            "recommend": recommend,
            "script_type": script_type,
            "script_type_inferred": u.get("script_type_inferred", False),
            "scriptPubKey": script_pubkey,          # bytes, needed for PSBT
            "tap_internal_key": tap_internal_key,   # optional, only Taproot
            "is_legacy": is_legacy,                 # NEW: for table styling / auto-unselect
            # "selected" still handled later by strategy / user
        })

    return enriched

def _apply_pruning_strategy(enriched: List[Dict], strategy: str) -> List[Dict]:
    """
    Apply deterministic pruning strategy.
    - Excludes legacy/unsupported from pruning calculation
    - Prunes only supported UTXOs in priority order (dust → heavy → optimal)
    - Returns list with supported first (pruned), unsupported last (always unselected)
    """
    ratio = PRUNING_RATIOS.get(strategy, 0.40)

    # Step 1: Split supported vs unsupported
    supported = []
    unsupported = []
    for u in enriched:
        new_u = dict(u)
        script_type = new_u.get("script_type", "")
        is_unsupported = new_u.get("is_legacy", False) or script_type not in ("P2WPKH", "Taproot", "P2TR")
        if is_unsupported:
            new_u["selected"] = False
            unsupported.append(new_u)
        else:
            supported.append(new_u)

    # If no supported UTXOs, return unsupported only
    if not supported:
        return unsupported

    # Step 2: Sort & prune ONLY supported
    # Primary sort: value descending (stable base)
    sorted_supported = sorted(
        supported,
        key=lambda u: (u["value"], u["txid"], u["vout"]),
        reverse=True,
    )

    total_supported = len(sorted_supported)
    keep_count = max(MIN_KEEP_UTXOS, int(total_supported * (1 - ratio)))
    prune_count = total_supported - keep_count

    # Secondary sort: prune priority (lowest HEALTH_PRIORITY first = dust → heavy → optimal)
    by_health = sorted(
        sorted_supported,
        key=lambda u: HEALTH_PRIORITY.get(u["health"], 999),  # DUST=0 first, OPTIMAL=4 last
    )

    # Apply selection
    result_supported = []
    for i, u in enumerate(by_health):
        new_u = dict(u)
        new_u["selected"] = i < prune_count
        result_supported.append(new_u)

    # Step 3: Combine — supported (pruned) first, unsupported last
    result = result_supported + unsupported

    # Debug
    legacy_skipped = len(unsupported)
    if legacy_skipped > 0:
        print(f">>> Excluded {legacy_skipped} unsupported/legacy from pruning. "
              f"Pruned {prune_count} of {total_supported} supported.")

    return result

def _build_df_rows(enriched: List[Dict]) -> tuple[List[List], bool]:
    """
    Convert enriched UTXOs into dataframe rows.
    - Uses 'is_legacy' flag to disable selection and show strong warnings
    - Handles unsupported script types by disabling selection
    - Fully responsive health badges for mobile
    - TXID shows full value (copyable on click)
    """
    rows: List[List] = []
    has_unsupported = False

    for u in enriched:
        script_type = u.get("script_type", "").strip()
        selected = bool(u.get("selected", False))           # From strategy/offline
        inferred = bool(u.get("script_type_inferred", False))
        is_legacy = bool(u.get("is_legacy", False))         # NEW: from _enrich_utxos

        # Force legacy to be unselected (safety + UX)
        if is_legacy:
            selected = False
            has_unsupported = True

        supported_in_psbt = script_type in ("P2WPKH", "Taproot", "P2TR")

        # ── Health badge HTML ─────────────────────────────────────────────────
        if not supported_in_psbt or is_legacy:
            has_unsupported = True

            if is_legacy or script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;background:rgba(255,68,68,0.12);padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ LEGACY</span><br>'
                    f'<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported in PSBT — migrate first</small>'
                    '</div>'
                )
            elif script_type in ("P2SH-P2WPKH", "Nested"):
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;background:rgba(255,153,0,0.12);padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ NESTED</span><br>'
                    f'<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported yet</small>'
                    '</div>'
                )
            else:
                health_html = (
                    f'<div class="health health-{u.get("health", "unknown").lower()}" style="padding:6px;border-radius:6px;">'
                    f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{u.get("health", "UNKNOWN")}</span><br>'
                    f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">Cannot prune</small>'
                    '</div>'
                )
        else:
            # Modern & supported
            health = u.get("health", "OPTIMAL")
            recommend = u.get("recommend", "")
            health_html = (
                f'<div class="health health-{health.lower()}" style="padding:6px;border-radius:6px;">'
                f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{health}</span><br>'
                f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">{recommend}</small>'
                '</div>'
            )

        # ── Friendly display name ─────────────────────────────────────────────
        display_type_map = {
            "P2WPKH": "Native SegWit",
            "Taproot": "Taproot",
            "P2TR": "Taproot",
            "P2SH-P2WPKH": "Nested SegWit",
            "P2PKH": "Legacy",
            "Legacy": "Legacy",
        }
        display_type = display_type_map.get(script_type, script_type)

        if inferred:
            display_type += ' <span style="color:#00cc66;font-weight:bold;">[inferred]</span>'

        if is_legacy:
            display_type += ' <span style="color:#ff6666;font-weight:bold;">[legacy – disabled]</span>'

        txid_full = u.get("txid", "unknown")

        # Build row
        rows.append([
            selected,                           # PRUNE checkbox (forced off for legacy)
            u.get("source", "Single"),          # Now includes derivation group e.g. "xpub(...) — BIP84"
            txid_full,
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
    hw_support_toggle,        
    xpub_field,              
    base_path_field,          
):
    """Main entrypoint: sanitize, collect, enrich, prune, build UI outputs."""

    print(">>> ANALYZE STARTED")
    print(f">>> addr_input: {addr_input[:50] if addr_input else 'empty'}...")
    print(f">>> offline_mode: {offline_mode}")
    print(f">>> hw_support_toggle: {hw_support_toggle}")

    # 1. Sanitize inputs
    params = _sanitize_analyze_inputs(
        addr_input=addr_input,
        strategy=strategy,
        dust_threshold=dust_threshold,
        fee_rate_slider=fee_rate_slider,
        thank_you_slider=thank_you_slider,
        future_fee_slider=future_fee_slider,
        offline_mode=offline_mode,
        manual_utxo_input=manual_utxo_input,
        hw_support_toggle=hw_support_toggle,
        xpub_field=xpub_field,
        base_path_field=base_path_field,
    )

    # 2. Collect UTXOs
    if params.offline_mode:
        raw_utxos = _collect_manual_utxos(params)
        scan_debug = "Offline mode — manual UTXOs only"
        print(f">>> Offline mode: collected {len(raw_utxos)} manual UTXOs")
    else:
        raw_utxos, scan_debug, _ = _collect_online_utxos(params)  # Ignore third item (no derivations)
        print(f">>> Online mode: collected {len(raw_utxos)} raw UTXOs")
        if raw_utxos:
            print(f">>> First UTXO address: {raw_utxos[0].get('address', 'unknown')}")

    # Early exit if nothing found
    if not raw_utxos:
        print(">>> No UTXOs found — returning empty state")
        return _analyze_empty(params.scan_source)

    # 3. Enrich UTXOs with metadata, health, etc.
    enriched = _enrich_utxos(raw_utxos, params)
    print(f">>> Enriched {len(enriched)} UTXOs")

    # Safety: ensure every UTXO has script_type
    if any("script_type" not in u for u in enriched):
        raise RuntimeError("Missing 'script_type' in enriched UTXOs — invariant violation")

    # 4. Apply pruning strategy (sets 'selected' flags)
    enriched_pruned = _apply_pruning_strategy(enriched, params.strategy)
    print(f">>> After pruning strategy: {len(enriched_pruned)} UTXOs, "
          f"{sum(1 for u in enriched_pruned if u['selected'])} selected")

    # Safety assertions
    assert len(enriched_pruned) >= MIN_KEEP_UTXOS
    assert any(not u["selected"] for u in enriched_pruned)
    assert params.strategy in PRUNING_RATIOS

    # 5. Build dataframe rows for display
    df_rows, has_unsupported = _build_df_rows(enriched_pruned)
    print(f">>> Built {len(df_rows)} table rows, has_unsupported={has_unsupported}")

    # 6. Taproot hardware wallet compatibility check (still useful for manual Taproot UTXOs)
    has_taproot = any(u["script_type"] in ("p2tr", "Taproot", "P2TR") for u in enriched_pruned)
    taproot_inferred = any(
        u.get("full_derivation_path") and u["script_type"] in ("p2tr", "Taproot", "P2TR")
        for u in enriched_pruned
    )
    taproot_hw_needed = (
        params.hw_support_toggle
        and has_taproot
        and not taproot_inferred
        and not (params.base_path_field and params.base_path_field.strip())
    )
    # 7. Build warning banner (HTML) — only legacy/nested + Taproot HW
    warning_banner = ""

    # Legacy/Nested unsupported warning (still needed)
    if has_unsupported:
        warning_banner += (
            "<div style='"
            "color:#ffddaa !important;"
            "background:#332200 !important;"
            "padding:clamp(20px,6vw,32px) !important;"
            "margin:clamp(20px,5vw,40px) auto !important;"  # centered
            "border:3px solid #ff9900 !important;"
            "border-radius:18px !important;"
            "text-align:center !important;"
            "font-size:clamp(1.1rem,4.5vw,1.4rem) !important;"
            "font-weight:700 !important;"
            "line-height:1.7 !important;"
            "box-shadow:0 0 60px rgba(255,153,0,0.5) !important;"
            "max-width:95% !important;"
            "width:100% !important;"
            "'>"
            "⚠️ Some UTXOs are unsupported for pruning<br><br>"
            "<span style='font-size:clamp(0.95rem,3.5vw,1.15rem);color:#ffcc88;'>"
            "Legacy (1...) and Nested SegWit (3...) are displayed for transparency "
            "but cannot be selected or included in the generated PSBT.<br><br>"
            "To prune, migrate funds to Native SegWit (bc1q...) or Taproot (bc1p...)."
            "</span>"
            "</div>"
        )

       # Taproot hardware signing warning (accurate for Omega Pruner)
        if taproot_hw_needed:
            warning_banner += (
                "<div style='"
                "color:#fff4cc !important;"
                "background:#2a1a00 !important;"
                "padding:clamp(20px,6vw,32px) !important;"
                "margin:clamp(20px,5vw,40px) auto !important;"
                "border:3px solid #ffcc00 !important;"
                "border-radius:18px !important;"
                "text-align:center !important;"
                "font-size:clamp(1.05rem,4.2vw,1.35rem) !important;"
                "font-weight:700 !important;"
                "line-height:1.7 !important;"
                "box-shadow:0 0 60px rgba(255,200,0,0.45) !important;"
                "max-width:95% !important;"
                "width:100% !important;"
                "'>"
                "⚠️ Taproot inputs detected — hardware signing notice<br><br>"
                "<span style='font-size:clamp(0.95rem,3.5vw,1.15rem);'>"
                "This PSBT includes Taproot (BIP86) inputs.<br><br>"
                "Some hardware wallets require full key origin information "
                "(derivation path + fingerprint) to sign Taproot inputs.<br><br>"
                "<strong>If signing fails:</strong><br>"
                "• Import the PSBT into a wallet that already knows the account (e.g. Sparrow)<br>"
                "• Or re-create the transaction in the originating wallet where the derivation path is known<br><br>"
                "Ωmega Pruner does not infer or reconstruct derivation paths."
                "</span>"
                "</div>"
            )
    # 8. Freeze the enriched state (immutable tuple)
    frozen_state = _freeze_enriched(
        enriched_pruned,
        strategy=params.strategy,
        scan_source=params.scan_source,
    )

    # 9. Return unified success state
    return _analyze_success(
        df_rows=df_rows,
        frozen_state=frozen_state,
        scan_source=params.scan_source,
        warning_banner=warning_banner,
    )

def _analyze_success(df_rows, frozen_state, scan_source, warning_banner=""):
    """Unified success return for analyze() — exactly 9 outputs to match Gradio click handler"""
    return (
        gr.update(value=df_rows),              # 0: DataFrame rows for the UTXO table
        frozen_state,                          # 1: enriched_state (frozen tuple: meta + utxos)
        gr.update(value=warning_banner),       # 2: Combined warning banner HTML
        gr.update(visible=True),               # 3: Show generate_row (PSBT button area)
        gr.update(visible=True),               # 4: Show import_file (JSON load area)
        scan_source,                           # 5: scan_source state (for later use)
        "",                                    # 6: Reserved/placeholder — was debug/status, keep empty for now
        gr.update(visible=True),               # 7: Show load_json_btn
        gr.update(visible=False),              # 8: Hide analyze_btn after success
    )

def _analyze_empty(scan_source: str = ""):
    """Empty/failure state return — exactly 9 outputs to match Gradio click handler"""
    return (
        gr.update(value=[]),                   # 0: Empty DataFrame
        (),                                    # 1: Empty enriched_state
        gr.update(value=""),                   # 2: No warning banner
        gr.update(visible=False),              # 3: Hide generate_row
        gr.update(visible=False),              # 4: Hide import_file
        scan_source,                           # 5: Preserve scan_source
        "",                                    # 6: Reserved/placeholder
        gr.update(visible=False),              # 7: Hide load_json_btn on failure
        gr.update(visible=True),               # 8: Keep analyze_btn visible (retry)
    )


# ====================
# generate_summary_safe() — Refactored for Clarity
# ====================

def _render_locked_state() -> Tuple[str, gr.update]:
    """Early return when selection is locked — responsive on mobile."""
    return (
        "<div style='"
        "text-align:center !important;"
        "padding: clamp(40px, 10vw, 80px) !important;"   # Scales padding
        "color:#aaaaaa !important;"                     # Slightly brighter gray
        "font-size: clamp(1.2rem, 5vw, 1.8rem) !important;"  # Responsive title
        "font-weight:700 !important;"
        "line-height:1.7 !important;"
        "max-width:90% !important;"
        "margin:0 auto !important;"
        "'>"
        "<span style='color:#00ffdd !important; text-shadow:0 0 30px #00ffdd !important;'>"
        "SELECTION LOCKED"
        "</span>"
        " — Ready to sign PSBT"
        "</div>",
        gr.update(visible=False)
    )

def _validate_utxos_and_selection(
    df_rows: List[list],
    utxos: List[dict],
    *,
    offline_mode: bool = False,
) -> Tuple[Optional[List[dict]], int, Optional[str]]:
    """
    Resolve selected UTXOs from df_rows checkboxes.
    
    NEW: Prioritizes UI state (df_rows) over enriched_state flags.
    - If df_rows present and matches utxos len → ALWAYS trust checkboxes, even if all unchecked
    - Only fallback to enriched_state if df_rows is missing/mismatched (safety net)
    - In offline_mode: even stricter, never fallback to avoid surprises
    """
    if not utxos:
        return None, 0, "NO_UTXOS"

    # Filter to valid dicts (safety)
    utxos = [u for u in utxos if isinstance(u, dict)]
    if not utxos:
        return None, 0, "NO_UTXOS"

    # Collect checked indices from df_rows
    selected_indices = []
    if isinstance(df_rows, list) and df_rows:
        for i, row in enumerate(df_rows):
            if not row or len(row) <= CHECKBOX_COL:
                continue
                
            checkbox_val = row[CHECKBOX_COL]
            is_checked = (
                checkbox_val in (True, 1, "true", "True", "1") 
                or bool(checkbox_val)
            )
            if is_checked and i < len(utxos):
                selected_indices.append(i)

    # NEW LOGIC: trust df_rows if it matches utxos
    has_valid_df = df_rows and len(df_rows) == len(utxos)
    
    if has_valid_df:
        # We have fresh UI state → use it directly
        # (even if selected_indices == [] → that's intentional "none selected")
        selected_utxos = [utxos[i] for i in selected_indices]
    else:
        # Fallback only when df_rows missing/outdated
        # (should be rare — mostly initial load)
        selected_utxos = [u for u in utxos if u.get("selected", False)]

    # Offline: override fallback if needed
    # (extra safety — users expect exact manual control)
    if offline_mode and not selected_indices:
        selected_utxos = []  # Force empty if nothing checked

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
            f" • <span style='"
            "color:#00ff88 !important;"
            "font-weight:800 !important;"
            "font-size:clamp(1.1rem, 4vw, 1.3rem) !important;"
            "text-shadow:0 0 25px #00ff88 !important;"
            "'>"
            f"DAO: {sats_to_btc_str(econ.dao_amt)}"
            "</span><br>"
            f"<span style='"
            "color:#00ffaa !important;"
            "font-size:clamp(0.9rem, 3.2vw, 1rem) !important;"
            "font-style:italic !important;"
            "'>"
            f"Thank you. Your support keeps Ωmega Pruner free, sovereign, and evolving. • Ω"
            "</span>"
        )
    elif dao_raw > 0:
        return (
            f" • <span style='"
            "color:#ff3366 !important;"
            "font-weight:800 !important;"
            "font-size:clamp(1.1rem, 4vw, 1.3rem) !important;"
            "text-shadow:0 0 25px #ff3366 !important;"
            "'>"
            f"DAO: {sats_to_btc_str(dao_raw)} → absorbed into fee"
            "</span><br>"
            f"<span style='"
            "color:#ff6688 !important;"
            "font-size:clamp(0.85rem, 3vw, 0.95rem) !important;"
            "font-style:italic !important;"
            "'>"
            "(below 546 sat dust threshold)"
            "</span>"
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
        margin: clamp(20px, 6vw, 40px) 0 !important;
        padding: clamp(20px, 6vw, 32px) !important;
        background:{bg} !important;
        border:4px solid {border} !important;
        border-radius:18px !important;
        box-shadow:0 0 60px rgba(255,100,100,0.8) !important;
        font-size: clamp(1.1rem, 4vw, 1.35rem) !important;
        line-height:1.8 !important;
        color:#ffeeee !important;
        max-width:95% !important;
        margin-left:auto !important;
        margin-right:auto !important;
    ">
      <div style="
          color:{color} !important;
          font-size: clamp(1.3rem, 5.5vw, 1.7rem) !important;
          font-weight:900 !important;
          text-shadow:0 0 20px {color} !important;
          margin-bottom: clamp(16px, 4vw, 24px) !important;
      ">
        {title}
      </div>
      Post-fee remainder (~{remainder_after_fee:,} sats) is small.<br>
      The pruned value will likely be fully or partially absorbed into miner fees.<br><br>
      <div style="
          color:#ffff88 !important;
          font-size: clamp(1.15rem, 4.5vw, 1.45rem) !important;
          font-weight:800 !important;
          text-shadow:0 0 15px #ffff99 !important;
          line-height:1.8 !important;
      ">
        Only proceed if your goal is wallet cleanup
      </div>
      <div style="color:#ffdd88 !important; font-size: clamp(1rem, 3.5vw, 1.15rem) !important; margin-top:8px !important;">
        — not expecting significant change back.
      </div><br>
      <div style="color:#ffaaaa !important; font-size: clamp(0.95rem, 3.2vw, 1.1rem) !important; line-height:1.7 !important;">
        💡 For a reliable change output, aim for:<br>
        • Value Pruned > ~5× Current Fee (good change back)<br>
        • Value Pruned > ~10× Current Fee (very comfortable)<br><br>
        This prune: <span style="color:#ffffff !important; font-weight:800 !important;">{sats_to_btc_str(econ.total_in)}</span> value and <span style="color:#ffffff !important; font-weight:800 !important;">{current_fee:,} sats</span> fee<br>
        Ratio: <span style="color:#ffffff !important; font-weight:800 !important;">{ratio}×</span> current fee
      </div><br>
      <small style="color:#88ffcc !important; font-size: clamp(0.9rem, 3vw, 1rem) !important;">
        💡 Pro tip: The bigger the prune (relative to fee), the more you get back as change. Small prunes = cleanup only.
      </small>
    </div>
    """

def _render_pruning_explanation(pruned_count: int, remaining_utxos: int) -> str:
    return f"""
<div style="
    margin: clamp(24px, 8vw, 48px) 0 !important;
    padding: clamp(20px, 6vw, 36px) !important;
    background:#001a00 !important;
    border:3px solid #00ff9d !important;
    border-radius:18px !important;
    box-shadow:
        0 0 50px rgba(0,255,157,0.6) !important,
        inset 0 0 30px rgba(0,255,157,0.08) !important;
    font-size: clamp(1.05rem, 3.8vw, 1.25rem) !important;
    line-height:1.85 !important;
    color:#ccffe6 !important;
    max-width:95% !important;
    margin-left:auto !important;
    margin-right:auto !important;
">
  <div style="
      color:#00ff9d !important;
      font-size: clamp(1.35rem, 5vw, 1.7rem) !important;
      font-weight:900 !important;
      text-shadow:0 0 20px #00ff9d !important;
      margin-bottom: clamp(12px, 4vw, 18px) !important;
  ">
    🧹 WHAT PRUNING ACTUALLY DOES
  </div>

  Pruning removes <span style="color:#aaffff !important;font-weight:700 !important;">inefficient UTXOs</span>
  (dust, legacy, or heavy) from your address.<br><br>

  • You pay a fee now to delete
    <span style="color:#00ffff !important;font-weight:800 !important;">{pruned_count}</span>
    inefficient inputs<br>

  • The remaining
    <span style="color:#00ffff !important;font-weight:800 !important;">{remaining_utxos}</span>
    UTXOs become cheaper and easier to spend later<br>

  • If no change output is created, the pruned value is absorbed into fees —
    but your wallet structure is
    <span style="color:#aaffff !important;">permanently cleaner</span><br><br>

  <span style="color:#00ffaa !important;font-weight:800 !important;">Goal:</span>
  a healthier address and lower future fees.<br>
  Pruning is often worth doing during low-fee periods.

  <small style="
      display:block !important;
      margin-top:18px !important;
      color:#88ffcc !important;
      font-style:italic !important;
      font-size: clamp(0.9rem, 3vw, 1rem) !important;
      opacity:0.85 !important;
  ">
    💡 Tip (optional): If your goal is to receive change, prune only when total value pruned exceeds
    ~10–20× the expected fee.
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
    dest_value,
    offline_mode,
) -> tuple:
    print(">>> generate_summary_safe CALLED")

    if locked:
        return _render_locked_state()

    # Extract UTXOs
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        meta, utxos = enriched_state
    else:
        utxos = enriched_state or []

    total_utxos = len(utxos)

    if total_utxos == 0:
        return no_utxos_msg, gr.update(visible=False)

    selected_utxos, pruned_count, error = _validate_utxos_and_selection(
        df, utxos, offline_mode=offline_mode
    )

    # DEBUG PRINTS
    print(f"pruned_count from validate: {pruned_count}, error: {error}, "
          f"selected_utxos_len: {len(selected_utxos) if selected_utxos else 0}")
    print(f"df_rows first row type: {type(df[0]) if df and df else 'empty'}")
    if df and df:
        print(f"df_rows first row preview: {df[0][:5]}")  # Optional: first 5 columns
    print(f"total_utxos: {total_utxos}")

    # Offline mode address check + warning banner
    offline_address_warning = ""
    if offline_mode:
        has_address = any(
            u.get("address")
            and isinstance(u["address"], str)
            and u["address"].strip().startswith(('bc1q', 'bc1p'))
            for u in utxos
        )
        if not has_address:
            offline_address_warning = """
<div style="
    color: #ffdd88 !important;
    background: rgba(51, 34, 0, 0.75) !important;
    border: 3px solid #ff9900 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin: 20px auto !important;
    font-weight: 700 !important;
    text-align: center !important;
    font-size: 1.1rem !important;
    line-height: 1.5 !important;
    box-shadow: 0 0 25px rgba(255,153,0,0.5) !important;
    text-shadow: 0 0 6px #000000 !important;
    max-width: 90% !important;
">
  ⚠️ Offline Mode: No valid address detected<br>
  To receive change back to your wallet, include 
  <strong style="color:#ffffff !important; text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important; font-weight:900 !important;">
    at least one bc1q... or bc1p... address
  </strong> 
  in your pasted UTXOs.<br><br>
  Format: 
    <code style="
        background: #000000 !important;
        color: #ffffff !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-family: monospace !important;
        text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.8) !important;
    ">txid:vout:value_in_sats:bc1qyouraddresshere</code><br><br>
  Right now, 
  <strong style="color:#ffffff !important; text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important; font-weight:900 !important;">
    no change output
  </strong> 
  will be created — all remaining value absorbed into fees (full wallet cleanup).<br>
  Edit your input and re-analyze to enable change.
</div>
"""

    # Hard errors
    if error == "NO_UTXOS":
        return no_utxos_msg, gr.update(visible=False)
    if error == "NO_SELECTION":
        return select_msg, gr.update(visible=False)

    # Final guard: only count supported inputs (offline-safe)
    supported_selected = [
        u for u in selected_utxos
        if (
            not u.get("script_type")  # None/unknown → allow
            or u.get("script_type") in ("P2WPKH", "Taproot", "P2TR")
        )
    ]

    offline_inferred_count = sum(
        1 for u in supported_selected if u.get("script_type_inferred")
    )

    # Offline badge — only show if there are actually inferred types (no empty box ever)
    offline_badge = ""
    if offline_mode:
        offline_inferred_count = sum(
            1 for u in supported_selected if u.get("script_type_inferred")
        )
        
        if offline_inferred_count > 0:
            offline_badge = f"""
            <div style="
                margin: 20px 0 !important;
                padding: 16px 20px !important;
                background: rgba(0, 50, 20, 0.45) !important;
                border: 3px solid #00ff88 !important;
                border-radius: 12px !important;
                box-shadow: 0 0 25px rgba(0,255,136,0.5) !important;
                font-size: clamp(1rem, 3.5vw, 1.15rem) !important;
                color: #ffffff !important;
                line-height: 1.5 !important;
            ">
                <div style="
                    color: #00ffdd !important;
                    font-weight: 900 !important;
                    font-size: clamp(1.15rem, 4vw, 1.3rem) !important;
                    margin-bottom: 8px !important;
                    text-shadow: 0 0 12px #00ffdd !important;
                ">
                    🛰️ OFFLINE MODE — IMPORTANT
                </div>
                
                {offline_inferred_count} input(s) accepted<br>
                <strong style="color: #ffff88 !important; font-weight: 800 !important;">
                    Script types are INFERRED from addresses only
                </strong><br>
                <span style="color: #ffdd88 !important; opacity: 0.95 !important;">
                    → Please double-check and verify before signing
                </span>
            </div>
            """

    pruned_count = len(supported_selected)
    if pruned_count == 0:
        return select_msg, gr.update(visible=False)

    remaining_utxos = total_utxos - pruned_count

    privacy_score, score_color = _compute_privacy_metrics(selected_utxos, total_utxos)
    econ = _compute_economics_safe(selected_utxos, fee_rate, dao_percent)

    if econ is None or econ.remaining <= 0:
        return (
            "<div style='"
            "text-align:center !important;"
            "padding:clamp(30px, 8vw, 60px) !important;"
            "background:rgba(30,0,0,0.7) !important;"
            "backdrop-filter:blur(10px) !important;"
            "border:2px solid #ff3366 !important;"
            "border-radius:16px !important;"
            "box-shadow:0 0 40px rgba(255,51,102,0.5) !important;"
            "font-size:clamp(1.2rem, 4.5vw, 1.5rem) !important;"
            "color:#ffaa88 !important;"
            "max-width:95% !important;"
            "margin:0 auto !important;"
            "'>"
            "<strong style='"
            "color:#ff3366 !important;"
            "font-size:clamp(1.4rem, 5.5vw, 1.8rem) !important;"
            "text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;"
            "'>Transaction Invalid</strong><br><br>"
            f"Current fee ({econ.fee:,} sats @ {fee_rate} s/vB) exceeds available balance.<br><br>"
            "<strong style='"
            "color:#ff3366 !important;"
            "text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;"
            "font-weight:900 !important;"
            "'>Lower the fee rate</strong> or select more UTXOs."
            "</div>",
            gr.update(visible=False)
        )

    all_input_weight = sum(u["input_weight"] for u in utxos)
    pre_vsize = max(
        (all_input_weight + 172 + total_utxos) // 4 + 10,
        (all_input_weight + 150 + total_utxos * 60) // 4 + 10,
    )
    savings_pct = round(100 * (1 - econ.vsize / pre_vsize), 1) if pre_vsize > econ.vsize else 0
    savings_label = (
        "NUCLEAR" if savings_pct >= 70 else
        "EXCELLENT" if savings_pct >= 50 else
        "GOOD" if savings_pct >= 30 else
        "WEAK"
    )

    # Future savings estimate (using selected vsize only)
    sats_saved = max(0, econ.vsize * (future_fee_rate - fee_rate))

    # Component rendering
    dao_line = _render_dao_feedback(econ, dao_percent)
    small_warning = _render_small_prune_warning(econ, fee_rate)

    cioh_warning = get_cioh_warning(
        pruned_count,
        len({u["address"] for u in selected_utxos}),
        privacy_score
    )

    pruning_explanation = _render_pruning_explanation(pruned_count, remaining_utxos)

    strategy_label = strategy.split(" — ")[0] if " — " in strategy else "Recommended"
    # Update the "change sent to" line to reflect reality
    change_line = (
        f"💧 Expected output: "
        f"<span style='color:#0f0 !important;font-weight:800 !important;'>{econ.change_amt:,} sats</span> "
        "change sent to standard address"
    )
    if offline_mode and not any(u.get("address") for u in utxos):
        change_line = (
            f"💧 <span style='color:#ff9900 !important;font-weight:800 !important;'>No change output</span> "
            "(all remaining value absorbed into fees - full cleanup)"
        )

    status_box_html = offline_address_warning + f"""
    <div style="
        text-align:center !important;
        margin:clamp(30px, 8vw, 60px) auto 20px auto !important;
        padding:clamp(24px, 6vw, 40px) !important;
        background: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) saturate(160%) !important;
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
            font-size:clamp(2.2rem, 8vw, 2.8rem) !important;
            font-weight:900 !important;
            letter-spacing:3px !important;
            text-shadow:0 0 35px #0f0 !important, 0 0 70px #0f0 !important;
            margin-bottom:clamp(16px, 4vw, 24px) !important;
        ">
            SELECTION READY
        </div>
        {offline_badge}
        <div style="
            color:#f7931a !important;
            font-size:clamp(1.5rem, 5vw, 1.9rem) !important;
            font-weight:800 !important;
            margin:clamp(12px, 3vw, 20px) 0 !important;
        ">
            {total_utxos:,} UTXOs • <span style="color:#00ff9d !important;">{strategy_label}</span> Strategy Active
        </div>
        <div style="
            color:#fff !important;
            font-size:clamp(1.3rem, 4.5vw, 1.7rem) !important;
            font-weight:700 !important;
            margin:clamp(12px, 3vw, 20px) 0 !important;
        ">
            Pruning <span style="color:#ff6600 !important;font-weight:900 !important;">{pruned_count:,}</span> inputs
        </div>
        <div style="
        color:#88ffcc !important;
        font-size:clamp(0.95rem, 3.2vw, 1.05rem) !important;
        line-height:1.6 !important;
        margin-bottom:16px !important;
    ">
        One-time structural cleanup — the numbers below show why this matters now.
    </div>
        <div style="
            color:#fff !important;
            font-size:clamp(1.4rem, 5vw, 1.8rem) !important;
            font-weight:800 !important;
            margin:clamp(16px, 4vw, 28px) 0 !important;
        ">
            Privacy Score (after this transaction): 
            <span style="
                color:{score_color} !important;
                font-size:clamp(1.8rem, 7vw, 2.5rem) !important;
                margin-left:12px !important;
                text-shadow:0 0 30px {score_color} !important;
            ">
              {privacy_score}/100
            </span>
        </div>
        <hr style="border:none !important;border-top:1px solid rgba(247,147,26,0.3) !important;margin:clamp(24px, 6vw, 40px) 0 !important;">
        <div style="font-size:clamp(1rem, 3.5vw, 1.15rem) !important;line-height:2.1 !important;">
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;">
              <b style="color:#fff !important;">Full wallet spend size today (before pruning):</b> 
              <span style="color:#ff9900 !important;font-weight:800 !important;">~{pre_vsize:,} vB</span>
            </div>
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;">
              <b style="color:#fff !important;">Size of this one-time pruning cleanup transaction:</b> 
              <span style="color:#0f0 !important;font-weight:800 !important;">~{econ.vsize:,} vB</span>
            </div>
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;color:#88ffcc !important;font-size:clamp(0.95rem, 3.2vw, 1.1rem) !important;line-height:1.6 !important;">
              💡 After pruning: your full wallet spend size drops to roughly <span style="color:#aaffcc !important;font-weight:700 !important;">~{pre_vsize - econ.vsize + 200:,} vB</span>
            </div>
            <div style="
                margin:clamp(16px, 4vw, 28px) 0 !important;
                color:#0f0 !important;
                font-size:clamp(1.2rem, 4.5vw, 1.5rem) !important;
                font-weight:900 !important;
                text-shadow:0 0 30px #0f0 !important;
                line-height:1.6 !important;
            ">
              {savings_label.upper()} WALLET CLEANUP!
            </div>
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;">
              <b style="color:#fff !important;">Current fee (paid now):</b> 
              <span style="color:#0f0 !important;font-weight:800 !important;">{econ.fee:,} sats @ {fee_rate} s/vB</span>{dao_line}
            </div>
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;color:#88ffcc !important;font-size:clamp(0.95rem, 3.2vw, 1.1rem) !important;line-height:1.6 !important;">
              {change_line}
            </div>
            <div style="margin:clamp(12px, 3vw, 16px) 0 !important;color:#88ffcc !important;font-size:clamp(0.95rem, 3.2vw, 1.1rem) !important;line-height:1.7 !important;">
              💡 Pruning now saves you <span style="color:#0f0 !important;font-weight:800 !important;">+{sats_saved:,} sats</span> versus pruning later if fees reach
        <span style="
              color:#ff3366 !important;
              font-weight:900 !important;
              text-shadow: 0 0 12px #ff3366, 0 0 24px #ff3366 !important;
          ">{future_fee_rate} s/vB</span>
            </div>
          </div>
          <hr style="border:none !important;border-top:1px solid rgba(247,147,26,0.3) !important;margin:clamp(24px, 6vw, 40px) 0 !important;">
          <div style="margin:clamp(24px, 6vw, 40px) 0 30px 0 !important;line-height:1.7 !important;">
            {cioh_warning}
          </div>
          <div style="
              margin:0 20px clamp(20px, 5vw, 40px) 20px !important;
              padding:clamp(20px, 5vw, 36px) !important;
              background: rgba(0,20,0,0.6) !important;
              backdrop-filter: blur(8px) !important;
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
        "<div style='"
        "color:#ff6666 !important;"
        "text-align:center !important;"
        "padding: clamp(30px, 8vw, 60px) !important;"
        "font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;"
        "font-weight:700 !important;"
        "line-height:1.7 !important;"
        "max-width:90% !important;"
        "margin:0 auto !important;"
        "'>"
        "No snapshot — run <strong style='color:#ffaa66 !important;'>Generate</strong> first."
        "</div>"
    )
def _render_no_inputs() -> str:
    return (
        "<div style='"
        "color:#ff6666 !important;"
        "text-align:center !important;"
        "padding: clamp(30px, 8vw, 60px) !important;"
        "font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;"
        "font-weight:700 !important;"
        "line-height:1.7 !important;"
        "max-width:90% !important;"
        "margin:0 auto !important;"
        "'>"
        "No UTXOs selected for pruning!<br><br>"
        "<span style='font-size: clamp(1rem, 3.5vw, 1.2rem) !important; color:#ffaa88 !important;'>"
        "Check the boxes next to inputs you want to prune."
        "</span>"
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
    full_spend_no_change: bool = False


def _extract_psbt_params(snapshot: dict) -> PsbtParams:
    return PsbtParams(
        inputs=snapshot["inputs"],
        scan_source=snapshot["scan_source"],
        dest_override=snapshot.get("dest_addr_override"),
        fee_rate=snapshot["fee_rate"],
        dao_percent=snapshot["dao_percent"],
        fingerprint_short=snapshot["fingerprint_short"],
        full_spend_no_change=snapshot.get("full_spend_no_change", False),
    )
def _resolve_destination(dest_override: Optional[str], scan_source: str) -> Union[bytes, str]:
    """
    Resolve final destination to scriptPubKey (bytes) or return error HTML.
    
    - Prefers explicit dest_override
    - Falls back to scan_source only if dest_override is empty/falsy
    - If both empty → returns b'' (no change output, absorb to fees/DAO)
    - Error message focused on modern addresses (bc1q... / bc1p...)
    """
    # Clean inputs
    override_clean = (dest_override or "").strip()
    source_clean = scan_source.strip()

    # Use override if provided; fallback to source only if override is empty
    final_dest = override_clean if override_clean else source_clean

    if not final_dest:
        # No destination at all → allow no-change PSBT (absorb remainder)
        return b''

    try:
        spk, _ = address_to_script_pubkey(final_dest)
        return spk
    except Exception:
        return (
            "<div style='color:#ff6666 !important; text-align:center !important; padding:30px; background:#440000 !important; border-radius:18px; box-shadow:0 0 50px rgba(255,51,102,0.5) !important;'>"
            "<div style='color:#ff3366 !important; font-size:1.8rem; font-weight:900;'>Invalid destination address</div><br><br>"
            "Please enter a <span style='font-weight:900; color:#ffffff !important;'>valid modern Bitcoin address</span>:<br>"
            "• <span style='font-weight:900; color:#00ffff !important;'>Native SegWit</span>: starts with bc1q...<br>"
            "• <span style='font-weight:900; color:#00ffff !important;'>Taproot</span>: starts with bc1p...<br><br>"
            "Legacy (1...) and Nested (3...) are supported for display but <span style='font-weight:900; color:#ffdd88 !important;'>not recommended</span> for new change outputs."
            "</div>"
        )
    
def _build_unsigned_tx(
    inputs: list[dict],
    econ: TxEconomics,
    dest_spk: bytes,
    params: PsbtParams,
) -> tuple[Tx, list[dict], str, bool]:
    """
    Construct unsigned transaction and return prepared UTXO info for PSBT.

    Returns:
        tx: Unsigned Tx object.
        utxos_for_psbt: Prepared UTXO info for PSBT.
        no_change_warning: HTML warning string (empty if none).
        has_change_output: True if a real change output was added.
    """
    tx = Tx()
    utxos_for_psbt: list[dict[str, any]] = []

    for u in inputs:
        tx.tx_ins.append(TxIn(bytes.fromhex(u["txid"]), int(u["vout"])))
        utxos_for_psbt.append({
            "value": u["value"],
            "scriptPubKey": address_to_script_pubkey(u["address"])[0],
            "script_type": u.get("script_type", "unknown"),
        })

    # DAO output
    if econ.dao_amt > 0:
        tx.tx_outs.append(TxOut(econ.dao_amt, DEFAULT_DAO_SCRIPT_PUBKEY))

    # Change / remainder
    change_amt = econ.change_amt
    if getattr(params, "full_spend_no_change", False):
        change_amt = econ.total_in - econ.fee - econ.dao_amt

    no_change_warning = ""
    has_change_output = False

    if change_amt <= 0:
        # No remainder exists → no warning, no change output
        pass

    elif not dest_spk:
        # Remainder exists but nowhere to send it → absorbed into fees
        no_change_warning = (
            "<div style='"
            "color:#ff9900 !important;"
            "background:rgba(51,34,0,0.6) !important;"
            "border:2px solid #ff9900 !important;"
            "border-radius:12px !important;"
            "padding:16px !important;"
            "margin:16px 0 !important;"
            "text-align:center !important;"
            "font-weight:600 !important;"
            "'>"
            "⚠️ No change address provided<br>"
            "All remaining value absorbed into fees (full wallet cleanup).<br><br>"
            "To receive change, include at least one bc1q... or bc1p... address in your pasted UTXOs."
            "</div>"
        )

    else:
        # Normal change output
        tx.tx_outs.append(TxOut(change_amt, dest_spk))
        has_change_output = True

    return tx, utxos_for_psbt, no_change_warning, has_change_output

def _generate_qr(psbt_b64: str) -> Tuple[str, str]:
    """Generate QR code for PSBT, with graceful fallback for large PSBTs."""
    # Safe threshold — QR version 40 max ~2953 chars
    if len(psbt_b64) > 2900:
        qr_html = ""
        qr_warning = (
            "<div style='"
            "margin: clamp(30px, 8vw, 60px) 0 !important;"
            "padding: clamp(24px, 6vw, 40px) !important;"
            "background:#221100 !important;"
            "border:4px solid #ff9900 !important;"
            "border-radius:18px !important;"
            "text-align:center !important;"
            "font-size: clamp(1.15rem, 4.5vw, 1.4rem) !important;"
            "color:#ffeecc !important;"
            "box-shadow:0 0 70px rgba(255,153,0,0.6) !important;"
            "max-width:95% !important;"
            "margin-left:auto !important;"
            "margin-right:auto !important;"
            "'>"
            "<span style='"
            "color:#ffff66 !important;"
            "font-size: clamp(1.4rem, 6vw, 1.8rem) !important;"
            "font-weight:900 !important;"
            "text-shadow:0 0 35px #ffff00 !important;"
            "'>"
            "PSBT Too Large for QR Code"
            "</span><br><br>"
            f"<span style='color:#ffddaa !important; font-size: clamp(1rem, 3.8vw, 1.2rem) !important;'>"
            f"Size: {len(psbt_b64):,} characters"
            "</span><br><br>"
            "Use the <span style='"
            "color:#00ffff !important;"
            "font-size: clamp(1.2rem, 5vw, 1.5rem) !important;"
            "font-weight:900 !important;"
            "text-shadow:0 0 30px #00ffff !important;"
            "'>"
            "COPY PSBT"
            "</span> button below and paste directly into your wallet.<br><br>"
            "<span style='color:#aaffff !important; font-size: clamp(0.95rem, 3.5vw, 1.1rem) !important;'>"
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
            "<div style='"
            "margin-top: clamp(12px, 3vw, 20px) !important;"
            "padding: clamp(10px, 3vw, 16px) !important;"
            "background:#331a00 !important;"
            "border:1px solid #ff9900 !important;"
            "border-radius:12px !important;"
            "color:#ffb347 !important;"
            "font-size: clamp(0.9rem, 3.2vw, 1rem) !important;"
            "line-height:1.5 !important;"
            "text-align:center !important;"
            "max-width:95% !important;"
            "margin-left:auto !important;"
            "margin-right:auto !important;"
            "'>"
            "Large PSBT — if QR scan fails, use <span style='color:#00ffff !important;font-weight:900 !important;'>COPY PSBT</span> and paste manually."
            "</div>"
        )

    return qr_html, qr_warning
    
def _compose_psbt_html(
    fingerprint: str,
    qr_html: str,
    qr_warning: str,
    psbt_b64: str,
    extra_note: str = "",
    has_change_output: bool = False,
    no_change_warning: str = "",  # NEW: to avoid duplicating the big orange warning
) -> str:
    """Compose full PSBT HTML output, with conditional QR display."""

    # Change message — success, skip if big warning exists, or subtle note
    change_message = ""
    if has_change_output:
        # Green success: change was actually sent
        change_message = """
        <div style="
            color:#aaffaa !important;
            font-size: clamp(1rem, 3.5vw, 1.15rem) !important;
            font-weight: 700 !important;
            margin: 24px 0 !important;
            padding: 12px !important;
            background: rgba(0, 50, 20, 0.4) !important;
            border: 1px solid #00aa66 !important;
            border-radius: 10px !important;
            text-align:center !important;
        ">
            💧 Change output sent to standard address
        </div>
        """

    elif no_change_warning:
        # Big orange warning already present (no destination) → skip subtle note
        change_message = ""

    else:
        # Dust / small remainder / absorbed (no user error, no big warning)
        change_message = """
        <div style="
            color:#ffcc88 !important;
            font-size: clamp(0.95rem, 3.2vw, 1.1rem) !important;
            margin: 16px 0 !important;
            text-align:center !important;
            opacity: 0.9 !important;
        ">
            No change output — remaining value absorbed into fee (wallet cleanup)
        </div>
        """

    # QR section — only rendered if we actually have a QR image
    qr_section = ""
    if qr_html:
        qr_section = f"""
        <!-- QR -->
        <div style="
            margin: clamp(30px, 8vw, 50px) auto !important;
            width: clamp(300px, 80vw, 520px) !important;
            max-width:96vw !important;
            padding: clamp(16px, 4vw, 24px) !important;
            background: rgba(0,0,0,0.7) !important;
            backdrop-filter: blur(10px) !important;
            border: clamp(6px, 2vw, 10px) solid #00cc77 !important;
            border-radius: clamp(16px, 4vw, 24px) !important;
            box-shadow: 0 0 45px rgba(0, 200, 120, 0.6) !important;
        ">
            {qr_html}
        </div>
        """

    # Warning from _generate_qr() — shows when too large or other QR issues
    qr_feedback = qr_warning if qr_warning else ""

    # Main HTML structure
    return f"""
    <div style="height: clamp(60px, 15vw, 100px) !important;"></div>

    <div style="
        text-align:center !important;
        margin: clamp(40px, 10vw, 80px) auto 0 !important;
        max-width:960px !important;
        position: relative;
        z-index: 1;
    ">
    <div style="
        display:inline-block !important;
        padding: clamp(40px, 10vw, 70px) !important;
        background: rgba(0, 0, 0, 0.42) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(16px) saturate(140%) !important;
        border: clamp(10px, 3vw, 14px) solid #f7931a !important;
        border-radius: clamp(24px, 6vw, 36px) !important;
        box-shadow: 
            0 0 140px rgba(247,147,26,0.95) !important,
            inset 0 0 60px rgba(247,147,26,0.15) !important;
        position: relative;
        max-width:95% !important;
    ">

        <!-- Selection Fingerprint -->
        <div style="
            margin: clamp(30px, 8vw, 50px) 0 !important;
            padding: clamp(20px, 5vw, 32px) !important;
            background: rgba(0, 30, 0, 0.6) !important;
            backdrop-filter: blur(8px) !important;
            border:4px solid #0f0 !important;
            border-radius: clamp(12px, 3vw, 18px) !important;
            box-shadow:0 0 80px rgba(0,255,0,0.8) !important;
            font-family:monospace !important;
        ">
            <div style="
                color:#0f0 !important;
                font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;
                font-weight:900 !important;
                letter-spacing:3px !important;
                text-shadow:0 0 20px #0f0 !important;
                margin-bottom: clamp(12px, 3vw, 20px) !important;
            ">
                Ω FINGERPRINT
            </div>
            <div style="
                color:#00ff9d !important;
                font-size: clamp(1.8rem, 6vw, 2.4rem) !important;
                font-weight:900 !important;
                letter-spacing: clamp(6px, 1.5vw, 10px) !important;
                text-shadow:0 0 30px #00ff9d !important, 0 0 60px #00ff9d !important;
            ">
                {fingerprint}
            </div>
            <div style="
                margin-top: clamp(16px, 4vw, 24px) !important;
                color:#00ffaa !important;
                font-size: clamp(1rem, 3.5vw, 1.2rem) !important;
                line-height:1.6 !important;
                font-weight:800 !important;
            ">
                Cryptographic proof of your pruning selection<br>
                Deterministic • Audit-proof • Never changes
            </div>
            <button onclick="navigator.clipboard.writeText('{fingerprint}').then(() => {{this.innerText='COPIED';setTimeout(()=>this.innerText='COPY FINGERPRINT',1500);}})"
                style="margin-top: clamp(12px, 3vw, 20px) !important;padding: clamp(8px, 2vw, 12px) clamp(16px, 4vw, 28px) !important;background:#000;color:#0f0;border:2px solid #0f0;border-radius:12px;font-size: clamp(1rem, 3.5vw, 1.2rem) !important;font-weight:800;cursor:pointer;box-shadow:0 0 20px #0f0;">
                COPY FINGERPRINT
            </button>
        </div>

        {extra_note}
        {change_message}
        {qr_section}
        {qr_feedback}

        <!-- PSBT Output -->
        <div style="margin: clamp(40px, 10vw, 80px) auto 20px !important;width:92% !important;max-width:880px !important;">
            <div style="
                position:relative !important;
                background: rgba(0,0,0,0.75) !important;
                backdrop-filter: blur(12px) !important;
                border: clamp(4px, 1.5vw, 6px) solid #f7931a !important;
                border-radius: clamp(12px, 3vw, 18px) !important;
                box-shadow:0 0 40px #0f0 !important;
                overflow:hidden !important;
            ">
                <textarea id="psbt-output" readonly 
                    style="
                        width:100% !important;
                        height: clamp(140px, 40vw, 200px) !important;
                        background:transparent !important;
                        color:#0f0 !important;
                        font-size: clamp(0.9rem, 3vw, 1rem) !important;
                        padding: clamp(16px, 4vw, 28px) !important;
                        padding-right: clamp(100px, 25vw, 160px) !important;
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
                        top: clamp(10px, 3vw, 16px) !important;
                        right: clamp(10px, 3vw, 16px) !important;
                        padding: clamp(10px, 3vw, 16px) clamp(24px, 6vw, 40px) !important;
                        background:#f7931a !important;
                        color:#000 !important;
                        border:none !important;
                        border-radius: clamp(10px, 3vw, 16px) !important;
                        font-weight:800 !important;
                        font-size: clamp(1rem, 3.5vw, 1.2rem) !important;
                        cursor:pointer !important;
                        box-shadow:0 0 30px #f7931a !important;
                    ">
                    COPY PSBT
                </button>
            </div>
            <div style="text-align:center !important;margin-top: clamp(10px, 3vw, 16px) !important;">
                <span style="color:#00f0ff !important;font-weight:700 !important;font-size: clamp(1rem, 3.5vw, 1.2rem) !important;">RBF enabled</span>
                <span style="color:#888 !important;font-size: clamp(0.9rem, 3vw, 1rem) !important;"> • Raw PSBT • </span>
                <span style="color:#666 !important;font-size: clamp(0.85rem, 2.8vw, 0.95rem) !important;">Inspect before signing</span>
            </div>
        </div>

        <!-- Wallet support -->
        <div style="
            color:#ff9900 !important;
            font-size: clamp(0.95rem, 3.2vw, 1.1rem) !important;
            text-align:center !important;
            margin: clamp(30px, 8vw, 60px) 0 clamp(16px, 4vw, 32px) 0 !important;
            padding: clamp(12px, 4vw, 20px) !important;
            background: rgba(30,0,0,0.6) !important;
            backdrop-filter: blur(8px) !important;
            border:2px solid #f7931a !important;
            border-radius: clamp(10px, 3vw, 16px) !important;
            box-shadow:0 0 40px rgba(247,147,26,0.4) !important;
            max-width:95% !important;
            margin-left:auto !important;
            margin-right:auto !important;
        ">
            <div style='color:#fff !important;font-weight:800 !important;font-size: clamp(1rem, 3.5vw, 1.2rem) !important;'>
                Important: Wallet must support <span style='color:#0f0 !important;'>PSBT</span>
            </div>
            <div style='color:#0f8 !important;margin-top: clamp(8px, 2vw, 12px) !important;font-size: clamp(0.95rem, 3.2vw, 1.1rem) !important;'>
                Sparrow • BlueWallet • Electrum • UniSat • Nunchuk • OKX
            </div>
        </div>
    </div>
    </div>
    """

def generate_psbt(psbt_snapshot: dict, full_selected_utxos: list[dict], df_rows) -> str:
    """Orchestrate PSBT generation using both snapshot (params) and full enriched UTXOs."""

    if not psbt_snapshot:
        return _render_no_snapshot()

    if not full_selected_utxos:
        return _render_no_inputs()

    # Safety check: nothing actually selected for pruning
    if not any(row[0] for row in df_rows if row):
        return (
            "<div style='"
            "    color:#ff9900 !important;"
            "    background:rgba(51,34,0,0.6) !important;"
            "    border:3px solid #ff9900 !important;"
            "    border-radius:14px !important;"
            "    padding: clamp(24px,6vw,40px) !important;"
            "    margin:20px 0 !important;"
            "    text-align:center !important;"
            "    font-size: clamp(1.1rem,4vw,1.3rem) !important;"
            "    font-weight:700 !important;"
            "    box-shadow:0 0 25px rgba(255,153,0,0.5) !important;"
            "'>"
            "⚠️ Nothing selected yet<br><br>"
            "Start over and check at least one UTXO in the table to generate the PSBT.<br>"
            "Your coins are waiting — just pick which ones to prune!"
            "</div>"
        )

    # Safe param extraction — critical guard
    try:
        params = _extract_psbt_params(psbt_snapshot)
    except Exception as e:
        log.error(f"Failed to extract PSBT params: {e}", exc_info=True)
        return (
            "<div style='"
            "    color:#ff6666 !important;"
            "    text-align:center !important;"
            "    padding: clamp(30px, 8vw, 60px) !important;"
            "    background:#440000 !important;"
            "    border-radius:18px !important;"
            "    box-shadow:0 0 50px rgba(255,51,102,0.5) !important;"
            "    font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;"
            "    line-height:1.7 !important;"
            "    max-width:90% !important;"
            "    margin:0 auto !important;"
            "'>"
            "    <span style='color:#ff3366 !important;font-size: clamp(1.4rem, 5.5vw, 1.8rem) !important;font-weight:900 !important;'>"
            "        Invalid or corrupted snapshot"
            "    </span><br><br>"
            "    Please click <strong style='color:#ffaa66 !important;'>GENERATE</strong> again."
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
            "<div style='"
            "color:#ffdd88 !important;"
            "text-align:center !important;"
            "padding: clamp(30px, 8vw, 60px) !important;"
            "background:#332200 !important;"
            "border:3px solid #ff9900 !important;"
            "border-radius:18px !important;"
            "box-shadow:0 0 60px rgba(255,153,0,0.5) !important;"
            "font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;"
            "line-height:1.8 !important;"
            "max-width:90% !important;"
            "margin:0 auto !important;"
            "'>"
            "<strong style='color:#ffff66 !important;font-size: clamp(1.4rem, 5.5vw, 1.8rem) !important;'>"
            "No supported inputs selected"
            "</strong><br><br>"
            "Only <strong style='color:#00ffff !important;'>Native SegWit (bc1q...)</strong> and <strong style='color:#00ffff !important;'>Taproot (bc1p...)</strong> inputs can be pruned.<br><br>"
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
            "<div style='color:#ff6666 !important; text-align:center !important; padding:30px; background:#440000 !important; border-radius:18px; box-shadow:0 0 50px rgba(255,51,102,0.5) !important;'>"
            "Invalid transaction economics — please re-analyze."
            "</div>"
        )

    # Build unsigned transaction
    tx, utxos_for_psbt, no_change_warning, has_change_output = _build_unsigned_tx(
        supported_inputs,
        econ,
        dest_spk,
        params,
    )

    if len(utxos_for_psbt) != len(tx.tx_ins):
        return (
            "<div style='color:#ff6666 !important; text-align:center !important; padding:30px; font-size:1.2rem; font-weight:700;'>"
            "Internal error: Input/UTXO count mismatch — please report this bug."
            "</div>"
        )

    # Generate PSBT
    psbt_b64, _ = create_psbt(tx, utxos_for_psbt)
    qr_html, qr_warning = _generate_qr(psbt_b64)

    # -------------------------
    # Warnings only (NO success state here)
    # -------------------------
    extra_note = ""

    if legacy_excluded:
        extra_note += (
            "<div style='color:#ffdd88 !important; background:#332200 !important; "
            "padding:16px; margin:30px 0; border:3px solid #ff9900 !important; "
            "border-radius:16px; text-align:center;'>"
            "⚠️ Some inputs were excluded from this PSBT<br><br>"
            "<small style='color:#ffcc88 !important;'>"
            "Only Native SegWit and Taproot inputs are supported.<br>"
            "Legacy/Nested inputs were automatically skipped."
            "</small>"
            "</div>"
        )

    # -------------------------
    # Final render
    # -------------------------
    return _compose_psbt_html(
        fingerprint=params.fingerprint_short,
        qr_html=qr_html,
        qr_warning=qr_warning,
        psbt_b64=psbt_b64,
        extra_note=extra_note,
		has_change_output=has_change_output,
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

    df_update, enriched_new, warning_banner, gen_row_vis, import_vis, scan_source_new, status_box_html, load_btn_vis, analyze_btn_vis = analyze(
        addr_input,
        strategy,
        dust_threshold,
        fee_rate_slider,
        thank_you_slider,
        future_fee_slider,
        offline_mode,
        manual_utxo_input,
        hw_support_toggle,       
        xpub_field,             
        base_path_field,  
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
		offline_mode,
    )

    # Return in correct order matching your .click() outputs
    return (
        df_update,
        enriched_new,
        warning_banner,
        gen_row_vis,
        import_vis,
        scan_source_new,
        status_box_html,
        load_btn_vis,   # ← This controls load_json_btn visibility
		analyze_btn_vis, 
    )

def fresh_empty_dataframe():
    return gr.DataFrame(
        value=[],                      # Truly empty — no dummy rows
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
        datatype=["bool", "str", "str", "html", "number", "str", "number", "html", "number"],
        type="array",
        interactive=True,              # Enables checkbox interaction
        wrap=True,
        row_count=(5, "dynamic"),
        max_height=500,
        max_chars=None,
        label=" ",
        static_columns=[1, 2, 3, 4, 5, 6, 7, 8],  # ← VERY IMPORTANT: keeps non-PRUNE columns fixed
        column_widths=["120px", "360px", "380px", "120px", "140px", "380px", "130px", "105px", "80px"]
    )


def process_uploaded_file(file):
    if not file:
        print(">>> process_uploaded_file: No file provided on button click")
        return {}
    try:
        print(">>> Opening file on button click:", file.name)
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
            print(">>> Raw content length on button:", len(content))
            if len(content) == 0:
                print(">>> File is empty on button click!")
                return {}
            parsed = json.loads(content)
            print(">>> Parsed OK on button - inputs count:", len(parsed.get("inputs", [])))
            return parsed
    except json.JSONDecodeError as e:
        print(">>> JSON decode error on button:", str(e))
        return {}
    except Exception as e:
        print(">>> File process error on button:", type(e).__name__, str(e))
        return {}

# =================================================================
# ========================= HANDLER FUNCTIONS (MUST BE TOP LEVEL) =
def update_status_and_ui(offline, dark):
    theme_icon = "🌙" if dark else "☀️"
    theme_text = "Dark" if dark else "Light"
    connection = "Offline 🔒 • No API calls • Fully air-gapped" if offline else "Online • API calls enabled"

    color = "#00ff88" if dark else "#006644"
    text_shadow = "0 0 20px #00ff88" if dark else "none"
    bg = "rgba(0, 30, 0, 0.5)" if dark else "rgba(200, 255, 220, 0.35)"

    return f"""
    <div style="
        text-align: center !important;
        padding: clamp(12px, 4vw, 20px) !important;
        margin: clamp(8px, 2vw, 16px) 0 !important;
        font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;
        font-weight: 900 !important;
        color: {color} !important;
        text-shadow: {text_shadow} !important;
        background: {bg} !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5),
                    0 8px 32px rgba(0,255,136,0.4),
                    inset 0 0 20px rgba(0,255,136,0.3);
        transition: all 0.4s ease;
        max-width: 95% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    ">
        <span style="font-size: clamp(1.4rem, 5.5vw, 1.8rem) !important; margin-right: 12px !important;">{theme_icon}</span>
        {theme_text} • {connection}
    </div>
    """

def offline_toggle_handler(offline, dark):
    manual_box = gr.update(visible=offline)
    warning_visible = gr.update(visible=offline)

    addr_clear = gr.update(value="") if offline else gr.update()
    addr_ui = gr.update(
        interactive=not offline,
        placeholder=(
            "Offline mode active — paste raw UTXOs below (txid:vout:value[:address])\n"
			"Include at least one bc1q... or bc1p... address for change output."
            if offline
            else "Paste a single Bitcoin address (bc1q..., bc1p..., 1..., or 3...)\n"
			    "Only the first valid address is used."
        )
    )

    status_html = update_status_and_ui(offline, dark)

    return manual_box, addr_ui, status_html

# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks(
    title="Ωmega Pruner v11 — Forged Anew"
) as demo:
    
    # Full-screen animated Ωmega background + Hero Banner
    gr.HTML("""
    <div id="omega-bg" style="
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
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
                font-size: 100vh !important;
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

<div style="
    text-align: center !important;
    margin: clamp(40px, 12vw, 100px) auto 30px auto !important;
    padding: clamp(30px, 6vw, 50px) clamp(15px, 4vw, 30px) !important;
    background: rgba(0,0,0,0.42) !important;
    backdrop-filter: blur(10px) !important;
    border: clamp(4px, 2vw, 8px) solid #f7931a !important;
    border-radius: clamp(16px, 5vw, 24px) !important;
    box-shadow: 0 0 80px rgba(247,147,26,0.4), inset 0 0 60px rgba(247,147,26,0.08) !important;
    max-width: 95vw !important;
    width: 100% !important;
    position: relative !important;
    z-index: 1 !important;
    overflow: hidden !important;
">

  <!-- Reclaim (line 1) -->
  <div style="
      color: #ffcc00 !important;
      font-size: clamp(2.8rem, 11vw, 5.0rem) !important;
      font-weight: 900 !important;
      letter-spacing: clamp(3px, 2vw, 12px) !important;
      text-shadow: 
          0 0 50px #ffcc00,
          0 0 100px #ffaa00,
          -2px -2px 0 #ffffff, 2px -2px 0 #ffffff,
          -2px  2px 0 #ffffff, 2px  2px 0 #ffffff !important;
      margin-bottom: clamp(8px, 2vw, 15px) !important;
      text-align: center !important;
      line-height: 1.0 !important;
  ">
    Reclaim
  </div>

  <!-- Sovereignty (line 2) -->
  <div style="
      color: #ffcc00 !important;
      font-size: clamp(2.6rem, 10vw, 4.6rem) !important;
      font-weight: 900 !important;
      letter-spacing: clamp(3px, 2vw, 12px) !important;
      text-shadow: 
          0 0 50px #ffcc00,
          0 0 100px #ffaa00,
          -2px -2px 0 #ffffff, 2px -2px 0 #ffffff,
          -2px  2px 0 #ffffff, 2px  2px 0 #ffffff !important;
      margin-bottom: clamp(20px, 5vw, 40px) !important;
      text-align: center !important;
      line-height: 1.0 !important;
  ">
    Sovereignty
  </div>

  <!-- ΩMEGA PRUNER -->
  <div style="
      color: #e65c00 !important;
      font-size: clamp(2.4rem, 9vw, 4.2rem) !important;
      font-weight: 900 !important;
      letter-spacing: clamp(2px, 1.5vw, 9px) !important;
      text-shadow: 
          0 0 25px #e65c00,
          0 0 50px #c94a00,
          -2px -2px 0 #000000, 2px -2px 0 #000000,
          -2px  2px 0 #000000, 2px  2px 0 #000000 !important;
      margin-bottom: clamp(30px, 6vw, 50px) !important;
      text-align: center !important;
  ">
    ΩMEGA PRUNER
  </div>

  <!-- NUCLEAR COIN CONTROL -->
  <div style="
      color: #0f0 !important;
      font-size: clamp(1.8rem, 7vw, 2.8rem) !important;
      font-weight: 900 !important;
      letter-spacing: clamp(3px, 1.2vw, 6px) !important;
      text-shadow: 0 0 30px #0f0, 0 0 60px #0f0;
      margin: clamp(20px, 5vw, 35px) 0 !important;
      text-align: center !important;
  ">
    NUCLEAR COIN CONTROL
  </div>

  <!-- Version -->
  <div style="
      color: #00ffaa !important;
      font-size: clamp(1rem, 3.5vw, 1.2rem) !important;
      letter-spacing: clamp(1px, 0.8vw, 3px) !important;
      text-shadow: 0 0 12px #00ffaa;
      margin: clamp(15px, 4vw, 25px) 0 !important;
      text-align: center !important;
  ">
    FORGED ANEW — v11
  </div>

  <!-- Body text -->
  <div style="
      color:#ddd !important;
      font-size: clamp(1.1rem, 3.8vw, 1.4rem) !important;
      line-height: 1.6 !important;
      max-width: 90vw !important;
      margin: clamp(30px, 6vw, 45px) auto !important;
      padding: 0 clamp(10px, 3vw, 20px) !important;
      text-align: center !important;
      word-break: break-word !important;
  ">
    Pruning isn’t just about saving sats today — it’s a deliberate step toward taking
    <strong style="color:#0f0 !important;">full strategic control</strong> of your Bitcoin.<br><br>
    
    By pruning inefficient UTXOs, you:<br>
    • <strong style="color:#00ff9d !important;">Slash fees</strong> during high-congestion periods<br>
    • <strong style="color:#00ff9d !important;">Reduce future costs</strong> with a cleaner UTXO set<br>
    • <strong style="color:#00ff9d !important;">Optimize your stack</strong> for speed, savings and privacy<br><br>

    <strong style="color:#f7931a !important;font-size: clamp(1.3rem, 4.5vw, 1.7rem) !important;font-weight:900 !important;letter-spacing:1px !important;">
      Prune smarter. Win forever.
    </strong><br><br>
    
    Paste your address below → click <strong style="color:#f7931a;">ANALYZE</strong>.
  </div>

  <!-- Down arrow -->
  <div style="
      font-size: clamp(2.5rem, 7vw, 4rem) !important;
      color:#f7931a !important;
      opacity:0.9;
      animation:pulse 2s infinite;
      text-align:center !important;
      margin-top: clamp(20px, 5vw, 40px) !important;
  ">
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
    /* Force dark mode on body and all Gradio containers */
    .dark-mode,
    .dark-mode .gradio-container,
    .dark-mode .gr-panel,
    .dark-mode .gr-form,
    .dark-mode .gr-box,
    .dark-mode .gr-group,
    .dark-mode textarea,
    .dark-mode input,
    .dark-mode .gr-button,
    .dark-mode .gr-textbox,
    .dark-mode .gr-dropdown {
        background: #000 !important;
        color: #0f0 !important;
        border-color: #f7931a !important;
    }

    /* Buttons in dark mode */
    .dark-mode .gr-button {
        background: #000 !important;
        color: #0f0 !important;
        border: 2px solid #f7931a !important;
    }
    .dark-mode .gr-button:hover {
        background: #f7931a !important;
        color: #000 !important;
    }

    /* Nuclear checkbox — bigger and more visible */
    input[type="checkbox"] {
        width: clamp(28px, 6vw, 36px) !important;
        height: clamp(28px, 6vw, 36px) !important;
        accent-color: #0f0 !important;
        background: #000 !important;
        border: clamp(2px, 0.5vw, 3px) solid #f7931a !important;
        border-radius: 8px !important;
        cursor: pointer;
        box-shadow: 0 0 20px rgba(247,147,26,0.6) !important;
        appearance: none;
        position: relative;
    }

    input[type="checkbox"]:checked {
        background: #0f0 !important;
        box-shadow: 0 0 30px #0f0 !important;
    }

    input[type="checkbox"]:checked::after {
        content: '✓';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #000;
        font-size: clamp(18px, 4vw, 24px) !important;
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

    /* Empty state messages */
    .empty-state-msg {
        color: #aaffcc !important;
        background: rgba(0, 30, 20, 0.6) !important;
        padding: clamp(30px, 8vw, 50px) !important;
        border-radius: 18px !important;
        border: 2px solid #00ff88 !important;
        box-shadow: 0 0 50px rgba(0, 255, 136, 0.4) !important;
    }

    body:not(.dark-mode) .empty-state-msg {
        color: #003322 !important;
        background: rgba(200, 255, 220, 0.25) !important;
        border: 2px solid #006644 !important;
        box-shadow: 0 0 30px rgba(0, 100, 68, 0.3) !important;
    }

    body:not(.dark-mode) .empty-state-msg > div:first-child {
        color: #004d33 !important;
    }
.gr-dataframe th:nth-child(1) {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow: visible !important;
    line-height: 1.2 !important;
    padding: 8px 4px !important;
    font-size: clamp(0.9rem, 3vw, 1rem) !important;
}
/* Make sure these columns can wrap + break long strings */
.gr-dataframe td:nth-child(2),  /* Source */
.gr-dataframe td:nth-child(3),  /* TXID */
.gr-dataframe td:nth-child(6) { /* Address */
    white-space: normal !important;
    word-break: break-all !important;     /* critical for hex & bech32 */
    overflow-wrap: break-word !important; /* fallback + better looking */
    hyphens: auto;                        /* optional: nicer breaks on some browsers */
    font-family: monospace !important;
    font-size: 0.95rem !important;
    line-height: 1.35 !important;
    padding: 8px 6px !important;          /* give a bit more breathing room */
}

/* Make sure the parent doesn't fight us */
.gr-dataframe td:nth-child(2),
.gr-dataframe td:nth-child(3),
.gr-dataframe td:nth-child(6) {
    max-width: none !important;           /* prevent artificial width caps */
    overflow: visible !important;         /* let it grow vertically */
}

/* Hover feedback for copy-ability */
.gr-dataframe td:nth-child(2):hover,
.gr-dataframe td:nth-child(3):hover,
.gr-dataframe td:nth-child(6):hover {
    background: rgba(0, 255, 136, 0.12) !important;
    cursor: pointer !important;
}

/* Optional: slightly reduce ellipsis aggression globally (safety net) */
.gr-dataframe td {
    white-space: normal !important;       /* try to weaken global nowrap */
    overflow: visible !important;
}
</style>
""")

    prune_badge = gr.HTML("")
   
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
        """
        Lock the UI completely after generating PSBT.
        Disables all inputs, toggles, sliders, and buttons.
        Hides unnecessary rows/elements.
        """
        return (
            gr.update(visible=False),                    # 0: gen_btn
            gr.update(visible=False),                    # 1: generate_row
            gr.update(visible=True),                     # 2: export_title_row
            gr.update(visible=True),                     # 3: export_file_row
            gr.update(visible=False, interactive=False), # 4: import_file
            "<div class='locked-badge'>LOCKED</div>",    # 5: locked_badge
            gr.update(interactive=False),                # 6: addr_input
            gr.update(interactive=False),                # 7: dest (destination textbox)
            gr.update(interactive=False),                # 8: strategy dropdown
            gr.update(interactive=False),                # 9: dust slider
            gr.update(interactive=False),                # 10: fee_rate_slider
            gr.update(interactive=False),                # 11: future_fee_slider
            gr.update(interactive=False),                # 12: thank_you_slider
            gr.update(interactive=False),                # 13: offline_toggle
            gr.update(interactive=False),                # 14: theme_toggle
            gr.update(interactive=False),                # 15: manual_utxo_input
            gr.update(interactive=False),                # 16: economy_btn
            gr.update(interactive=False),                # 17: hour_btn
            gr.update(interactive=False),                # 18: halfhour_btn
            gr.update(interactive=False),                # 19: fastest_btn
            gr.update(visible=False),                    # 20: load_json_btn (hide when locked)
            gr.update(visible=False),                    # 21: file uploader (optional extra hide)
        )

    with gr.Column():
        theme_js = gr.HTML("")

        # ← Modern Bitcoin Optimization Note
        gr.HTML(
            value="""
            <div style="
                margin: clamp(20px, 5vw, 60px) auto !important;
                padding: clamp(16px, 4vw, 28px) !important;
                max-width: 95% !important;
                width: 100% !important;
                background: rgba(0, 20, 10, 0.6) !important;
                border: 3px solid #00ff9d !important;
                border-radius: 18px !important;
                text-align: center !important;
                font-size: clamp(1rem, 3.5vw, 1.15rem) !important;
                line-height: 1.7 !important;
                color: #ccffe6 !important;
                box-shadow: 0 0 clamp(30px, 8vw, 60px) rgba(0, 255, 157, 0.4) !important;
                overflow-wrap: break-word !important;
                word-break: break-word !important;
            ">
                <div style="
                    color: #00ffdd !important;
                    font-size: clamp(1.3rem, 5vw, 1.8rem) !important;
                    font-weight: 900 !important;
                    letter-spacing: clamp(1px, 0.5vw, 2px) !important;
                    margin-bottom: clamp(12px, 3vw, 16px) !important;
                    text-shadow: 0 0 25px #00ffdd !important;
                ">
                    Optimized for Modern Bitcoin
                </div>

                Ωmega Pruner is built for <strong style="color:#00ffff !important;font-weight:900 !important;">modern single-signature wallets</strong>,
                prioritizing <strong style="color:#00ffff !important;font-weight:900 !important;">privacy</strong>,
                <strong style="color:#00ffff !important;font-weight:900 !important;">fee efficiency</strong>,
                and <strong style="color:#00ffff !important;font-weight:900 !important;">hardware-wallet compatibility</strong>.
                <br><br>
                
                This tool is intentionally <strong style="color:#00ffff !important; font-weight:900 !important;">non-interactive at the transaction level</strong> —  
                it optimizes wallet structure and future spend efficiency,  
                not counterparty-dependent transaction negotiation.
                <br><br>

                ✅ Fully supported for PSBT creation and hardware signing:<br>
                <strong style="color:#00ffff !important;font-weight:900 !important;">Native SegWit (bc1q…)</strong>  •  
                <strong style="color:#00ffff !important;font-weight:900 !important;">Taproot / BIP86 (bc1p…)</strong>
                <br><br>

                PSBTs generated here include all required metadata
                (<span style="font-weight:900 !important; color:#00ffcc !important; text-shadow:0 0 10px rgba(0,255,204,0.5) !important;">
				UTXO data, derivation paths, fingerprints
				</span>)
                and can be signed either online or fully offline / air-gapped
                using <strong style="color:#00ffcc !important;">Sparrow</strong>,
               	<strong style="color:#00ffcc !important;">Coldcard</strong>,
                <strong style="color:#00ffcc !important;">Ledger</strong>, 
                <strong style="color:#00ffcc !important;">Trezor</strong>, 
                <strong style="color:#00ffcc !important;">Jade</strong>, 
                and similar wallets.
                <br><br>

                ⚠️ Legacy inputs
                (<strong style="color:#ffaa00 !important;font-weight:900 !important;">1…</strong>)
                and Nested SegWit inputs
                (<strong style="color:#ffaa00 !important;font-weight:900 !important;">3…</strong>)
                are shown for transparency only and
				<strong style="color:#ff6666 !important;font-weight:900 !important;">
				cannot be included in the generated PSBT
				</strong>.
				<br>
                To spend or consolidate these inputs, use a compatible wallet or migrate them separately.
            </div>
            """
        )
        mode_status = gr.HTML("")  # ← Empty placeholder — will be filled dynamically

        # ── Theme Toggle ──
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                theme_toggle = gr.Checkbox(
                    label="🌙 Dark Mode (pure black)",
                    value=True,
                    interactive=True,
                    info="Retinal protection • Nuclear glow preserved • Recommended",
                )

           # ── Main Input Fields ──
        with gr.Row():
            addr_input = gr.Textbox(
                label="Enter Bitcoin Address",
                placeholder=(
                    "Paste a single modern Bitcoin address (bc1q... or bc1p...)\n"
                    "Multiple lines are ignored — only the first valid address is used.\n"
                    "100% non-custodial."
                ),
                lines=4,
                scale=2,
            )

            dest = gr.Textbox(
                label="Destination (optional)",
                placeholder="Paste Bitcoin address",
                info=(
                    "Change output returns here.<br>"
                    "• Leave blank → returns to original scanned address<br>"
                ),
                scale=1,
            )
        # === AIR-GAPPED / OFFLINE MODE HEADER ===
        gr.HTML(
            value="""
            <div style="
                text-align:center !important;
                padding: clamp(20px, 6vw, 32px) !important;
                margin: clamp(30px, 8vw, 50px) 0 !important;
                background:#001100 !important;
                border:3px solid #00ff88 !important;
                border-radius:18px !important;
                box-shadow:
                    0 12px 40px rgba(0,0,0,0.6) !important,
                    0 8px 32px rgba(0,255,136,0.4) !important,
                    inset 0 0 30px rgba(0,255,136,0.2) !important;
                max-width:95% !important;
                margin-left:auto !important;
                margin-right:auto !important;
            "> 
              <div style="
                  color:#00ff88 !important;
                  font-size: clamp(1.3rem, 5.5vw, 1.7rem) !important;
                  font-weight:900 !important;
                  text-shadow:0 0 30px #00ff88 !important;
                  margin-bottom: clamp(10px, 3vw, 16px) !important;
              ">
                🔒 Air-Gapped / Offline Mode
              </div>

              <div style="
                  color:#aaffcc !important;
                  font-size: clamp(1rem, 3.8vw, 1.2rem) !important;
                  line-height:1.7 !important;
              ">
                Fully offline operation — no API calls, perfect for cold wallets.<br><br>
                Paste raw UTXOs manually below.
              </div>
            </div>
            """
        )

        # === OFFLINE MODE ===
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                offline_toggle = gr.Checkbox(
                    label="🔒 Offline / Air-Gapped Mode",
                    value=False,
                    interactive=True,
                    info="No API calls • Paste raw UTXOs • True cold wallet prep",
                )

        with gr.Row(visible=False) as manual_box_row:
            with gr.Column():
                gr.HTML("""
                <div style="
                    color: #ffdd88 !important;
                    background: rgba(51, 34, 0, 0.75) !important;
                    border: 3px solid #ff9900 !important;
                    border-radius: 14px !important;
                    padding: clamp(16px, 4vw, 20px) !important;
                    margin: 16px 0 20px 0 !important;
                    font-weight: 700 !important;
                    text-align: center !important;
                    font-size: clamp(1rem, 3.8vw, 1.1rem) !important;
                    line-height: 1.5 !important;
                    box-shadow: 0 0 25px rgba(255,153,0,0.5) !important, inset 0 0 12px rgba(0,0,0,0.6) !important;
                    max-width: 100% !important;
                    overflow-x: hidden !important;
                ">
                  ⚠️ Important: Offline Mode Address Requirement<br>
                  <span style="
                      font-size: clamp(1.05rem, 4vw, 1.15rem) !important;
                      color: #ffffff !important;
                      text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;
                      font-weight: 900 !important;
                  ">
                    To receive change back to your wallet, include 
                    <strong style="color: #ffffff !important;">at least one valid address</strong> 
                    (bc1q... or bc1p...) in your pasted UTXOs.<br><br>

                    Format example:<br>
                    <div style="overflow-x: auto; max-width: 100%; margin: 12px 0;">
                      <code style="
                          display: block !important;
                          background: #000000 !important;
                          color: #ffffff !important;
                          padding: 12px !important;
                          border-radius: 8px !important;
                          font-family: monospace !important;
                          font-size: clamp(0.85rem, 3.2vw, 0.95rem) !important;
                          text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;
                          box-shadow: inset 0 0 6px rgba(0,0,0,0.8) !important;
                          white-space: pre-wrap !important;
                          word-break: break-all !important;
                          overflow-wrap: anywhere !important;
                          line-height: 1.4 !important;
                          width: fit-content !important;
                          min-width: 100% !important;
                      ">txid:vout:value_in_sats:bc1qyouraddresshere</code>
                    </div><br>

                    If no address is provided, 
                    <strong style="color:#ffffff !important; text-shadow: 0 0 6px #000000, 0 0 12px #000000 !important;">
                    no change output
                    </strong> 
                    will be created — all remaining value absorbed into fees (full wallet cleanup only).<br>
                    Add an address and re-analyze if you want change.
                  </span>
                </div>
                """)
                manual_utxo_input = gr.Textbox(
                    label="🔒 OFFLINE MODE • ACTIVE INPUT • Paste raw UTXOs (one per line)",
                    placeholder="""Paste raw UTXOs — one per line

Format: txid:vout:value_in_sats[:address]

Examples:
abc123...000:0:125000:bc1qexample...          ← include address
def456...789:1:5000000:bc1p...               ← REQUIRED for change output
txidhere:2:999999                            ← OK if another line has address

No API calls • Fully air-gapped safe""",
                    lines=10,
                )

        # === Pruning Strategy & Economic Controls Header ===
        gr.HTML(
            value="""
            <div style="
                text-align:center !important;
                padding: clamp(20px, 6vw, 32px) !important;
                margin: clamp(30px, 8vw, 60px) 0 0 0 !important;
                background: linear-gradient(90deg, rgba(0,50,0,0.55), rgba(0,30,0,0.65)) !important;
                border:3px solid #00ff88 !important;
                border-radius:18px !important;
                box-shadow:
                    0 12px 40px rgba(0,0,0,0.6) !important,
                    0 8px 32px rgba(0,255,136,0.3) !important,
                    inset 0 0 30px rgba(0,255,136,0.15) !important;
                max-width:95% !important;
                margin-left:auto !important;
                margin-right:auto !important;
            "> 
              <div style="
                  color:#00ff88 !important;
                  font-size: clamp(1.3rem, 5.5vw, 1.6rem) !important;
                  font-weight:900 !important;
                  text-shadow:0 0 30px #00ff88 !important;
                  margin-bottom: clamp(10px, 3vw, 16px) !important;
              ">
                Pruning Strategy & Economic Controls
              </div>

              <div style="
                  color:#aaffcc !important;
                  font-size: clamp(1rem, 3.8vw, 1.2rem) !important;
                  line-height:1.7 !important;
                  text-shadow:0 2px 4px rgba(0,0,0,0.8) !important;
              ">
                Choose how aggressive your prune will be — and fine-tune fees & donations below
              </div>
            </div>
            """
        )

        # Strategy dropdown + Dust threshold
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

        load_json_btn = gr.Button("Load Selection from JSON", variant="primary", visible=False)
        json_parsed_state = gr.State({})  # ← dict instead of str

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
                font-size: clamp(1.2rem, 5vw, 1.4rem);
                font-weight: 900;
                text-shadow: 0 0 20px #00ff88;
                letter-spacing: 1px;
            }

            .dark-mode .check-to-prune-header .header-subtitle {
                color: #aaffaa;
                font-size: clamp(0.95rem, 3.5vw, 1.1rem);
                margin-top: 8px;
            }

            /* Light mode — softer, readable colors */
            body:not(.dark-mode) .check-to-prune-header .header-title {
                color: #008844;
                font-size: clamp(1.2rem, 5vw, 1.4rem);
                font-weight: 900;
                letter-spacing: 1px;
            }

            body:not(.dark-mode) .check-to-prune-header .header-subtitle {
                color: #006633;
                font-size: clamp(0.95rem, 3.5vw, 1.1rem);
                margin-top: 8px;
            }
            </style>
        """)

        df = gr.Dataframe(
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
            datatype=["bool", "str", "str", "html", "number", "str", "number", "html", "number"],
            type="array",
            interactive=True,
            wrap=True,
            row_count=(5, "dynamic"),
            max_height=500,
            max_chars=None,
            label=" ",
            static_columns=[1, 2, 3, 4, 5, 6, 7, 8],  # 0-based index — PRUNE is editable
            column_widths=["120px", "380", "380px", "120px", "140px", "380px", "130px", "105px", "80px"]
        )

        gr.HTML("""
            <script>
                // Simple click-to-expand for TXID cells (very long TXIDs)
                document.addEventListener('DOMContentLoaded', function() {
                    const txidCells = document.querySelectorAll('.gr-dataframe td:nth-child(3)');
                    txidCells.forEach(cell => {
                        cell.addEventListener('click', function(e) {
                            e.stopPropagation();  // Prevent any row-level events
                            this.classList.toggle('expanded');
                        });
                    });
                });
            </script>
        """)

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
                <div style="text-align:center;padding:clamp(30px, 8vw, 60px) 0 clamp(20px, 5vw, 40px) 0 !important;">

                  <!-- Main Header — FROZEN = icy blue theme -->
                  <div style="
                      color:#00ddff !important;
                      font-size:clamp(2.2rem, 9vw, 3rem) !important;
                      font-weight:900 !important;
                      letter-spacing:clamp(6px, 2vw, 10px) !important;
                      text-shadow:0 0 40px #00ddff, 0 0 80px #00ddff,
                                  0 4px 8px #000, 0 8px 20px #000000ee,
                                  0 12px 32px #000000cc;
                      margin-bottom:clamp(16px, 4vw, 24px) !important;
                  ">
                    🔒 SELECTION FROZEN
                  </div>
                  
                  <!-- Core message — signature green -->
                  <div style="
                      color:#aaffaa !important;
                      font-size:clamp(1.2rem, 4.5vw, 1.6rem) !important;
                      font-weight:700 !important;
                      text-shadow:0 0 20px #0f0,
                                  0 3px 6px #000, 0 6px 16px #000000dd,
                                  0 10px 24px #000000bb;
                      max-width:760px !important;
                      margin:0 auto clamp(12px, 3vw, 20px) auto !important;
                      line-height:1.7 !important;
                  ">
                    Your pruning intent is now immutable • Permanent audit trail secured
                  </div>
                  
                  <!-- Extra reassurance — bright cyan -->
                  <div style="
                      color:#00ffdd !important;
                      font-size:clamp(1rem, 3.8vw, 1.2rem) !important;
                      opacity:0.9;
                      font-weight:700 !important;
                      text-shadow:0 2px 4px #000, 0 4px 12px #000000cc, 0 8px 20px #000000aa;
                      max-width:680px !important;
                      margin:clamp(16px, 4vw, 24px) auto clamp(8px, 2vw, 12px) auto !important;
                      line-height:1.7 !important;
                  ">
                    The file below includes:<br>
                    All selected UTXOs • Ω fingerprint • Transaction parameters
                  </div>
                  
                  <div style="
                      color:#aaffaa !important;
                      font-size:clamp(1rem, 3.8vw, 1.2rem) !important;
                      opacity:0.9;
                      font-weight:700 !important;
                      text-shadow:0 2px 4px #000, 0 4px 12px #000000cc, 0 8px 20px #000000aa;
                      max-width:680px !important;
                      margin:0 auto clamp(30px, 8vw, 50px) auto !important;
                      line-height:1.7 !important;
                  ">
                    Download for backup, offline verification, or future reference
                  </div>

                </div>
            """)

        with gr.Row(visible=False) as export_file_row:
            export_file = gr.File(
                label="",
                interactive=False
            )

        # Reset button (final control)
        with gr.Column():
            reset_btn = gr.Button("NUCLEAR RESET — START OVER — NO FUNDS AFFECTED", variant="secondary")

    # =============================
    # — Handlers —
    # =============================

        offline_toggle.change(
            fn=offline_toggle_handler,
            inputs=[offline_toggle, theme_toggle],
            outputs=[
                manual_box_row,
                addr_input,
                mode_status,
            ],
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
    # — Import File (pure state mutation) — Now with Button for stability
    # =============================
    load_json_btn.click(
        fn=process_uploaded_file,  # ← This one reads + parses
        inputs=[import_file],      # ← Directly from the file component
        outputs=[json_parsed_state],
    ).then(
        fn=load_selection,
        inputs=[json_parsed_state, enriched_state],
        outputs=[enriched_state, warning_banner],
        js="() => { console.log('LOAD_JSON_BUTTON_CLICKED'); return true; }"
    ).then(
        fn=rebuild_df_rows,
        inputs=[enriched_state],
        outputs=[df, gr.State()]
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
            offline_toggle,
        ],
        outputs=[status_output, generate_row]
    ).then(
        fn=lambda x: print(">>> RESTORE_CHAIN_COMPLETED - enriched len:", len(x) if x else 0),
        inputs=[enriched_state],
        outputs=[gr.State()]
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
			dest_value
        ],
        outputs=[
            df,
            enriched_state,
            warning_banner,
            generate_row,
            import_file,
            scan_source,
			status_output,
			load_json_btn,
			analyze_btn,
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
        inputs=[psbt_snapshot, selected_utxos_for_psbt,df],
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
            load_json_btn,          # ← NEW
            import_file,            # ← Optional: hide uploader too (if you want double-hide)
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
            fresh_empty_dataframe(),
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
            gr.update(value="", interactive=True),                    # manual_utxo_input
            gr.update(visible=False),                                # manual_box_row 
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
			gr.update(visible=False),								 # ← load_json_btn hidden on reset
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
            manual_box_row,             
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
			load_json_btn,

        ],
    ).then(
        fn=lambda: (
            "", 
            gr.update(visible=False),    # generate_row hidden
            gr.update(visible=True)      # analyze_btn re-shown
        ),
        outputs=[status_output, generate_row, analyze_btn]
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
            offline_toggle,
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
            offline_toggle,
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
            offline_toggle,
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
            offline_toggle,
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
            offline_toggle,
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
    demo.load(
        fn=get_prune_score,
        outputs=prune_badge
    )
    
    # 5. FOOTER
    gr.HTML(
        """
        <div style="width: 100%; margin-top: clamp(50px, 12vw, 100px) !important;"></div>

        <div style="
            width: 100%;
            max-width: 760px;
            margin: 0 auto 30px auto;
            text-align: center;
            line-height: 1.8;
        ">
            <!-- VERSION -->
            <div style="
                font-size: clamp(1rem, 4vw, 1.2rem) !important;
                font-weight: 700;
                letter-spacing: 0.5px;
                color: #f7931a;
                text-shadow: 0 0 15px rgba(247,147,26,0.7);
            ">
                Ωmega Pruner v11 — Forged Anew
            </div>

            <!-- GITHUB LINK -->
            <a href="https://github.com/babyblueviper1/Viper-Stack-Omega"
               target="_blank"
               rel="noopener"
               style="
                   font-size: clamp(0.9rem, 3.5vw, 1.05rem) !important;
                   font-weight: 600;
                   text-decoration: none;
                   color: #f7931a;
                   text-shadow: 0 0 12px rgba(247,147,26,0.6);
               ">
                GitHub • Open Source • Apache 2.0
            </a>

            <br><br>

            <!-- CUSTOM BUILDS SECTION -->
            <div style="margin: clamp(20px, 5vw, 30px) auto; max-width: 720px;">
                <a href="https://www.babyblueviper.com/p/omega-pruner-custom-builds"
                   target="_blank"
                   style="color: inherit; text-decoration: none;">
                    <div style="
                        display: inline-block;
                        padding: clamp(8px, 2.5vw, 12px) clamp(16px, 4vw, 24px) !important;
                        margin: clamp(8px, 2vw, 12px) 0 !important;
                        font-size: clamp(0.9rem, 3.5vw, 1.05rem) !important;
                        font-weight: 700;
                        letter-spacing: 0.3px;
                        border-radius: 14px;
                        transition: all 0.4s ease;
                        color: #00ff9d;
                        background: rgba(0, 40, 20, 0.4);
                        box-shadow: 0 0 20px rgba(0, 255, 157, 0.4);
                    ">
                        This build is engineered for speed and clarity.
                    </div>
                    <br>
                    <div style="
                        display: inline-block;
                        padding: clamp(8px, 2.5vw, 12px) clamp(16px, 4vw, 24px) !important;
                        margin: clamp(8px, 2vw, 12px) 0 !important;
                        font-size: clamp(0.9rem, 3.5vw, 1.05rem) !important;
                        font-weight: 700;
                        letter-spacing: 0.3px;
                        border-radius: 14px;
                        transition: all 0.4s ease;
                        color: #00ff88;
                        background: rgba(0, 35, 15, 0.4);
                        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
                    ">
                        For extended capabilities or tailored integrations, custom versions can be commissioned.
                    </div>
                </a>
            </div>

            <!-- Donation section -->
            <div style="
                text-align: center !important;
                margin: clamp(40px, 10vw, 80px) auto 60px auto !important;
                padding: clamp(20px, 5vw, 40px) !important;
                background: rgba(0,0,0,0.5) !important;
                border-top: 2px solid #f7931a !important;
                max-width: 95vw !important;
                color: #aaa !important;
                font-size: clamp(0.9rem, 3vw, 1.1rem) !important;
            ">
                <div style="margin-bottom: 12px !important;">
                    <strong style="color:#f7931a !important;">Support Ωmega Pruner</strong><br>
                    <small>If this tool saved you sats or helped your stack — show your love.</small>
                </div>

                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: clamp(20px, 5vw, 40px);
                    flex-wrap: wrap;
                ">
                    <!-- On-chain QR -->
                    <div style="
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        min-width: 180px;
                    ">
                        <div style="
                            font-weight: 700;
                            color: #f7931a;
                            margin-bottom: 10px;
                            font-size: clamp(0.95rem, 3.2vw, 1.1rem);
                        ">
                            On-chain Bitcoin
                        </div>
                        <img src="https://api.qrserver.com/v1/create-qr-code/?data=bitcoin:bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj&size=300x300&color=247-147-26&bgcolor=0-0-0"
                             alt="Donate Bitcoin On-chain"
                             style="
                                 width: 180px;
                                 height: 180px;
                                 border: 2px solid #f7931a;
                                 border-radius: 12px;
                                 box-shadow: 0 0 20px rgba(247,147,26,0.5);
                                 max-width: 45vw;
                             " />
                        <br>
                        <small style="display: block; margin-top: 8px; word-break: break-all;">
                            bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj
                            <button onclick="navigator.clipboard.writeText('bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj').then(() => {
                                this.innerText='COPIED';
                                this.style.color='#00ff88';
                                this.style.fontWeight='900';
                                this.style.textShadow='0 0 10px #00ff88';
                                setTimeout(() => {
                                    this.innerText='Copy';
                                    this.style.color='#f7931a';
                                    this.style.fontWeight='normal';
                                    this.style.textShadow='none';
                                }, 1500);
                            })"
                                    style="
                                        background:none !important;
                                        border:none !important;
                                        color:#f7931a !important;
                                        cursor:pointer !important;
                                        font-size:0.9rem !important;
                                        margin-left:8px !important;
                                        padding:0 !important;
                                        transition: all 0.2s ease !important;
                                    ">
                                Copy
                            </button>
                        </small>
                    </div>

                    <!-- Lightning (Bolt 12) -->
                    <div style="
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        min-width: 180px;
                    ">
                        <div style="
                            font-weight: 700;
                            color: #00ff88;
                            margin-bottom: 10px;
                            font-size: clamp(0.95rem, 3.2vw, 1.1rem);
                        ">
                            Lightning (Bolt 12)
                        </div>
                        <img src="https://api.qrserver.com/v1/create-qr-code/?data=lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c&size=300x300&color=0-255-136&bgcolor=0-0-0"
                             alt="Donate Lightning (Bolt 12)"
                             style="
                                 width: 180px;
                                 height: 180px;
                                 border: 2px solid #00ff88;
                                 border-radius: 12px;
                                 box-shadow: 0 0 20px rgba(0,255,136,0.5);
                                 max-width: 45vw;
                             " />
                        <br>
                            <small style="
                                display: block;
                                margin-top: 8px;
                                max-width: 220px;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                                font-family: monospace;
                            ">
                                lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c
                                <button onclick="navigator.clipboard.writeText('lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c').then(() => {
                                    this.innerText='COPIED';
                                    this.style.color='#00ff88';
                                    this.style.fontWeight='900';
                                    this.style.textShadow='0 0 10px #00ff88';
                                    setTimeout(() => {
                                        this.innerText='Copy';
                                        this.style.color='#00ff88';
                                        this.style.fontWeight='normal';
                                        this.style.textShadow='none';
                                    }, 1500);
                                })"
                                        style="
                                            background:none !important;
                                            border:none !important;
                                            color:#00ff88 !important;
                                            cursor:pointer !important;
                                            font-size:0.9rem !important;
                                            margin-left:8px !important;
                                            padding:0 !important;
                                            transition: all 0.2s ease !important;
                                        ">
                                    Copy
                                </button>
                            </small>
                    </div>
                </div>

                <div style="margin-top: 20px; font-size: 0.9rem; opacity: 0.8;">
                    Thank you for supporting open-source Bitcoin tools. • Ω
                </div>
            </div>

            <!-- Light mode overrides -->
            <style>
            body:not(.dark-mode) div[style*="00ff9d"],
            body:not(.dark-mode) div[style*="00ff88"] {
                color: #004d33 !important;
                background: rgba(220, 255, 235, 0.15) !important;
                box-shadow: 0 2px 8px rgba(0, 80, 50, 0.1) !important;
            }

            a:hover div[style*="padding"] {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 255, 136, 0.4);
            }
            </style>

            <br><br>

            <!-- TAGLINE -->
            <span style="
                color: #0f0;
                font-size: clamp(0.9rem, 3.8vw, 1.1rem) !important;
                font-weight: 800;
                letter-spacing: 0.6px;
                text-shadow:
                    0 0 15px #0f0,
                    0 0 30px #0f0,
                    0 0 6px #000,
                    0 4px 10px #000,
                    0 8px 20px #000000e6;
            ">
                Prune smarter. Win forever. • Ω
            </span>
        </div>
        """,
        elem_id="omega_footer",
    )
	
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False, debug=False)
