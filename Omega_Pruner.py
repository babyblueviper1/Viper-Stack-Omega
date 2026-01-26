"""
CANONICAL STATE MODEL (AUTHORITATIVE — DO NOT VIOLATE)

User Inputs (mutable via UI):
- addr_input
- strategy (pruning strategy dropdown)
- dust_threshold
- dest_addr
- fee_rate_slider
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

import gradio as gr
import requests
import time
import base64
import io
import qrcode
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from functools import partial
import threading
import hashlib
import tempfile
from datetime import datetime
import copy
import pandas as pd
import statistics
import concurrent.futures

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────
MEMPOOL_API = "https://mempool.space/api"
BLOCKSTREAM_API = "https://blockstream.info/api/address/{addr}/utxo"
BITCOINER_API = "https://bitcoiner.live/api/address/{addr}/utxo"

MIN_KEEP_UTXOS = 1

HEALTH_PRIORITY = {
    "DUST":    0,
    "HEAVY":   1,
    "CAREFUL": 2,
    "MEDIUM":  3,
    "MANUAL":  3,     # Neutral — no strong prune/keep bias
    "OPTIMAL": 4,
}

# DataFrame column indices (0-based)
CHECKBOX_COL = 0
SOURCE_COL   = 1
TXID_COL     = 2
HEALTH_COL   = 3
VALUE_COL    = 4
ADDRESS_COL  = 5
WEIGHT_COL   = 6
TYPE_COL     = 7
VOUT_COL     = 8

# Empty state messages
no_utxos_msg = (
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

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

PRUNING_RATIOS = {
    "Privacy First — ~30% pruned (lowest CIOH risk)": 0.30,
    "Recommended — ~40% pruned (balanced savings & privacy)": 0.40,
    "More Savings — ~50% pruned (stronger fee reduction)": 0.50,
    "NUCLEAR PRUNE — ~90% pruned (maximum savings, highest CIOH)": 0.90,
}

# ── Global HTTP session ─────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "User-Agent": "OmegaPruner-v11 (+https://github.com/babyblueviper1/Viper-Stack-Omega)",
    "Accept": "application/json",
    "Connection": "keep-alive",
})

_fee_cache_lock = threading.Lock()

# Globals for simple in-memory cache
_last_medians = {"24h": None, "1w": None, "1m": None}
_last_fetch_time = 0
_CACHE_TTL = 300  # 5 minutes


# ===========================
# Selection resolver
# ===========================
def _resolve_selected(df_rows: List[list], enriched_state: tuple) -> List[dict]:
    """
    Resolve selected UTXOs from Gradio DataFrame checkbox state.
    Uses row-order matching — safe for Gradio's mutable UI representation.
    """
    if not df_rows:
        return []

    # Extract frozen utxos (enforced by canonical model: tuple[dict])
    _, utxos = enriched_state   # safe — analyze() always returns (meta, tuple)

    if len(df_rows) != len(utxos):
        log.warning(f"Row mismatch: df_rows={len(df_rows)}, utxos={len(utxos)}")
        return []

    selected = []
    for idx, row in enumerate(df_rows):
        if not row or len(row) <= CHECKBOX_COL:
            continue

        checkbox_val = row[CHECKBOX_COL]
        # Gradio can send bool, int, str — normalize safely
        is_checked = checkbox_val in (True, 1, "true", "True", "1") or bool(checkbox_val)
        if is_checked:
            selected.append(utxos[idx])

    log.debug(f"_resolve_selected: {len(selected)} UTXOs selected via checkboxes")
    return selected


def _selection_snapshot(selected_utxos: List[dict]) -> dict:
    """
    Deterministic, audit-friendly snapshot of the current user selection.
    This is the canonical representation saved to JSON.
    """
    if not selected_utxos:
        return {
            "fingerprint": "none",
            "count": 0,
            "total_value": 0,
            "utxos": [],
        }

    sorted_utxos = sorted(selected_utxos, key=lambda u: (u["txid"], u["vout"]))

    return {
        "fingerprint": _selection_fingerprint(sorted_utxos),
        "count": len(sorted_utxos),
        "total_value": sum(u["value"] for u in sorted_utxos),
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
            for u in sorted_utxos
        ],
    }


def _selection_fingerprint(selected_utxos: List[dict]) -> str:
    """
    Deterministic short hash of selected inputs (sorted by txid:vout).
    Used for audit trail and duplicate detection.
    """
    if not selected_utxos:
        return "none"

    keys = sorted((u["txid"], u["vout"]) for u in selected_utxos)
    data = ":".join(f"{txid}:{vout}" for txid, vout in keys).encode()
    return hashlib.sha256(data).hexdigest()[:16]

# =========================
# Utility Functions
# =========================

def safe_get(url: str, timeout: int = 8) -> Optional[requests.Response]:
    """Robust GET with 3 retries, exponential backoff, and strict timeout."""
    for attempt in range(3):
        try:
            log.debug(f"safe_get attempt {attempt+1}: {url} (timeout={timeout}s)")
            r = session.get(url, timeout=timeout)

            if r.status_code == 200:
                return r

            if r.status_code == 429:  # Rate limit
                sleep_time = 1.5 ** attempt
                log.warning(f"Rate limited (429) on {url} — sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue

            log.debug(f"Bad status from {url}: {r.status_code}")

        except Exception as e:
            log.debug(f"Request failed (attempt {attempt+1}): {type(e).__name__}: {e}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))  # Short backoff

    log.warning(f"safe_get failed after 3 attempts: {url}")
    return None

def get_live_fees() -> Dict[str, int]:
    """Fetch recommended fees from mempool.space with 30-second caching."""
    # Safe cache check
    if hasattr(get_live_fees, "cache") and hasattr(get_live_fees, "cache_time"):
        if time.time() - get_live_fees.cache_time < 30:
            return get_live_fees.cache

    try:
        r = session.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        if r.status_code == 200:
            data = r.json()
            fees = {
                "fastest":   int(data["fastestFee"]),
                "half_hour": int(data["halfHourFee"]),
                "hour":      int(data["hourFee"]),
                "economy":   int(data["economyFee"]),
                "minimum":   int(data["minimumFee"]),
            }
            now = time.time()
            with _fee_cache_lock:
                get_live_fees.cache = fees
                get_live_fees.cache_time = now
            log.debug(f"Fetched live fees: economy={fees['economy']}, fastest={fees['fastest']}")
            return fees
        else:
            log.warning(f"Fees endpoint returned {r.status_code}: {r.text[:100]}")
    except requests.exceptions.RequestException as e:
        log.warning(f"Failed to fetch live fees: {e}")

    # Conservative fallback
    return {
        "fastest":   10,
        "half_hour": 6,
        "hour":      3,
        "economy":   1,
        "minimum":   1,
    }

def get_prune_score() -> str:
    """
    Display current fee vs recent medians with color-coded badge.
    No explicit score — colors reflect relative conditions.
    Uses real medians from mempool.space (avgFee_90 mean).
    Includes price, block height, hashrate, difficulty adjustment, and halving in context.
    """
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
                    p90_fees = [block.get('avgFee_90', 1) for block in data if 'avgFee_90' in block]
                    if p90_fees:
                        avgs[period] = statistics.mean(p90_fees)
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
    primary_avg = avgs['1w'] or avgs['1m'] or avgs['24h'] or None
    day_avg   = avgs['24h']
    month_avg = avgs['1m']

    # Ratio for color only (neutral if no primary avg)
    ratio = current / primary_avg if primary_avg is not None and primary_avg > 0 else 1.0

    # Badge color based on ratio
    if ratio < 0.5:
        color = "#00ff88"  # Excellent green
    elif ratio < 0.8:
        color = "#00ffdd"  # Good cyan
    elif ratio < 1.2:
        color = "#ff9900"  # Fair orange
    else:
        color = "#ff3366"  # Poor red

    fee_unit = "sat/vB" if current == 1 else "sats/vB"

    # Context: price, block height, hashrate, difficulty adjustment, halving
    price = "—"
    height = "—"
    hr_formatted = "—"
    adjustment_text = ""
    halving_text = ""

    try:
        # Block height
        height_r = session.get("https://mempool.space/api/blocks/tip/height", timeout=8)
        if height_r.status_code == 200:
            height = height_r.text.strip()

        # Price
        price_r = session.get("https://mempool.space/api/v1/prices", timeout=8)
        if price_r.status_code == 200:
            price_usd = price_r.json().get('USD', 0)
            price = f"${price_usd:,}" if price_usd > 0 else "—"

        # Hashrate (prefer 1w, fallback 3d)
        for hr_period in ["1w", "3d"]:
            hr_r = session.get(f"https://mempool.space/api/v1/mining/hashrate/{hr_period}", timeout=10)
            if hr_r.status_code == 200:
                hr_data = hr_r.json()
                current_hashrate = None
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

        # Next Difficulty Adjustment
        da_r = session.get("https://mempool.space/api/v1/difficulty-adjustment", timeout=8)
        if da_r.status_code == 200:
            da_data = da_r.json()
            adjustment_pct = da_data.get('difficultyChange', 0)
            blocks_remaining = da_data.get('remainingBlocks', 0)
            days_remaining = round(blocks_remaining / 144) if blocks_remaining else 0
            adjustment_text = f"Next adj: {adjustment_pct:+.2f}% in ~{days_remaining} days"

        # Halving countdown (computed from height)
        if height != "—":
            try:
                current_height = int(height)
                next_halving = ((current_height // 210000) + 1) * 210000
                blocks_to_halving = next_halving - current_height
                days_to_halving = round(blocks_to_halving / 144)
                halving_text = f"Halving: ~{days_to_halving:,} days"
            except ValueError:
                pass

    except Exception as e:
        log.warning(f"Context fetch partial failure: {e}")

    # Build context line with responsive breaks
    context_line = f"BTC: {price} • Block: {height}"
    if hr_formatted != "—":
        context_line += f" • Hashrate: {hr_formatted}"
    if adjustment_text:
        context_line += f" • {adjustment_text}"
    if halving_text:
        context_line += f"<br>{halving_text}"
    if context_line == f"BTC: {price} • Block: {height}":
        context_line = "—"

    # Final HTML
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
            Pruning Conditions
        </div>

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
            <div style="text-align: center !important;">
                <div style="color: #00ddff !important; font-size: clamp(1.3rem, 5vw, 1.7rem) !important; font-weight: 900 !important;">
                    {f"{day_avg:.1f}" if day_avg is not None else "—"} sats/vB
                </div>
                <div style="color: #88ccff !important; font-size: clamp(0.9rem, 3.2vw, 1.1rem) !important;">
                    1-day median
                </div>
            </div>

            <div style="text-align: center !important;">
                <div style="color: #00ff88 !important; font-size: clamp(1.4rem, 5.5vw, 1.8rem) !important; font-weight: 900 !important; text-shadow: 0 0 20px #00ff88 !important;">
                    {f"{primary_avg:.1f}" if primary_avg is not None else "—"} sats/vB
                </div>
                <div style="color: #88ffaa !important; font-size: clamp(0.95rem, 3.4vw, 1.15rem) !important;">
                    1-week median
                </div>
            </div>

            <div style="text-align: center !important;">
                <div style="color: #aaff88 !important; font-size: clamp(1.3rem, 5vw, 1.7rem) !important; font-weight: 900 !important;">
                    {f"{month_avg:.1f}" if month_avg is not None else "—"} sats/vB
                </div>
                <div style="color: #99ffbb !important; font-size: clamp(0.9rem, 3.2vw, 1.1rem) !important;">
                    1-month median
                </div>
            </div>
        </div>

        <div style="
            color: #88ffaa !important;
            font-size: clamp(1rem, 3.5vw, 1.2rem) !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            margin-top: 20px !important;
        ">
            <small style="color: #66cc99 !important; font-weight: normal !important;">
                {context_line}
            </small>
        </div>
    </div>
    """
	
# ── Update enriched state from UI checkbox changes ──────────────────────────────
def update_enriched_from_df(df_rows: List[list], enriched_state: tuple, locked: bool) -> tuple:
    """
    Apply current checkbox state from DataFrame to the frozen enriched_state.
    Enforces unbreakable rule: legacy/nested inputs ALWAYS unselected.
    Returns new immutable tuple (never mutates original).
    """
    if locked or not enriched_state:
        return enriched_state

    # Unpack frozen state (invariant: always (meta, tuple[dict]))
    meta, utxos = enriched_state if isinstance(enriched_state, tuple) and len(enriched_state) == 2 else ({}, enriched_state)

    if len(df_rows) != len(utxos):
        log.warning(f"df_rows length mismatch: {len(df_rows)} vs utxos {len(utxos)}")
        return enriched_state

    updated_utxos = []
    for row, u in zip(df_rows, utxos):
        new_u = dict(u)  # shallow copy — safe since we rebuild tuple

        script_type = u.get("script_type", "")
        if script_type not in ("P2WPKH", "Taproot", "P2TR"):
            new_u["selected"] = False  # Unbreakable defense: legacy/nested never selected
        else:
            # Gradio checkbox can be bool, int, str — normalize safely
            checkbox_val = row[CHECKBOX_COL] if len(row) > CHECKBOX_COL else False
            new_u["selected"] = checkbox_val in (True, 1, "true", "True", "1") or bool(checkbox_val)

        updated_utxos.append(new_u)

    return (meta, tuple(updated_utxos))


# ── Load saved selection from JSON snapshot ─────────────────────────────────────
def load_selection(parsed_snapshot: dict, current_enriched: Any) -> Tuple[Any, str]:
    """
    Restore checkbox selection from a saved JSON snapshot into the current enriched state.
    Returns updated enriched_state tuple + user-facing HTML message.
    """
    log.debug(f"load_selection called - snapshot type: {type(parsed_snapshot)}")

    if not parsed_snapshot or not isinstance(parsed_snapshot, dict):
        log.debug("No valid parsed snapshot provided")
        return current_enriched, "No valid parsed JSON loaded"

    try:
        if "inputs" not in parsed_snapshot:
            log.warning("Invalid snapshot format: missing 'inputs'")
            return current_enriched, "Invalid Ωmega Pruner selection file"

        # Build case-insensitive set of (txid.lower(), vout) keys
        selected_keys = set()
        for inp in parsed_snapshot.get("inputs", []):
            if not isinstance(inp, dict) or "txid" not in inp or "vout" not in inp:
                log.debug(f"Skipping invalid input in JSON: {inp}")
                continue
            try:
                txid = str(inp["txid"]).lower().strip()
                vout = int(inp["vout"])
                selected_keys.add((txid, vout))
            except (ValueError, TypeError):
                log.debug(f"Skipping invalid txid/vout: {inp}")
                continue

        log.debug(f"Loaded {len(selected_keys)} selected keys from snapshot")

        # Normalize current enriched state
        if isinstance(current_enriched, tuple) and len(current_enriched) == 2:
            meta, utxos = current_enriched
            utxos = list(utxos)
        else:
            meta = {}
            utxos = list(current_enriched or [])

        if not utxos:
            log.info("Current enriched state empty — restore deferred until analysis")
            return (), (
                "<div style='color:#aaffcc;padding:30px;background:#001100;border:2px solid #00ff88;border-radius:16px;text-align:center;'>"
                "<span style='color:#00ffdd;font-size:1.6rem;font-weight:900;'>Selection file loaded!</span><br><br>"
                "<strong>Table is empty — restore will happen after:</strong><br>"
                "1. Paste the same addresses/xpubs<br>"
                "2. Click ANALYZE (table must load first)<br>"
                "3. Upload JSON again — checkboxes will restore<br><br>"
                "If table stays empty, check your address input."
                "</div>"
            )

        # Restore selection flags
        updated = []
        matched = 0
        for u in utxos:
            new_u = dict(u)
            txid_lower = str(u.get("txid", "")).lower().strip()
            vout = None
            try:
                vout_raw = u.get("vout")
                vout = int(vout_raw) if vout_raw is not None else None
            except (ValueError, TypeError):
                pass

            is_selected = (txid_lower, vout) in selected_keys if vout is not None else False
            new_u["selected"] = is_selected
            if is_selected:
                matched += 1
            updated.append(new_u)

        log.debug(f"Restore complete - matched {matched}/{len(selected_keys)}")

        return_tuple = (meta, tuple(updated)) if meta else tuple(updated)

        if matched == 0:
            message = (
                "<div style='color:#ffddaa;padding:30px;background:#332200;border:2px solid #ff9900;border-radius:16px;text-align:center;'>"
                "<span style='color:#ffff66;font-size:1.6rem;font-weight:900;'>Selection loaded — no matching UTXOs found</span><br><br>"
                f"File contains {len(selected_keys)} UTXOs.<br>"
                "They don't match current analysis (different addresses? UTXOs spent? txid case?).<br>"
                "Checkboxes not restored."
                "</div>"
            )
        else:
            message = (
                "<div style='color:#aaffff;padding:30px;background:#001122;border:2px solid #00ffff;border-radius:16px;text-align:center;'>"
                f"<span style='color:#00ffff;font-size:1.6rem;font-weight:900;'>Selection loaded — {matched}/{len(selected_keys)} UTXOs restored "
                f"({len(utxos)} total in current table)</span>"
                "</div>"
            )

        return return_tuple, message

    except Exception as e:
        log.error(f"Error processing selection snapshot: {e}", exc_info=True)
        return current_enriched, f"Failed to process selection: {str(e)}"


# ── Rebuild DataFrame rows from enriched state (for restore / refresh) ───────────
def rebuild_df_rows(enriched_state: Any) -> tuple[List[List], bool]:
    """
    Rebuild DataFrame rows from enriched_state for display.
    Handles legacy/nested disabling, preserves saved 'selected' flags,
    and flags unsupported types for UX warnings.
    Returns (rows, has_unsupported)
    """
    if not enriched_state:
        return [], False

    # Normalize to list of dicts
    if isinstance(enriched_state, tuple) and len(enriched_state) == 2:
        _, utxos = enriched_state
        state_list = list(utxos) if isinstance(utxos, (list, tuple)) else []
    elif isinstance(enriched_state, (list, tuple)):
        state_list = list(enriched_state)
    else:
        log.warning(f"Invalid enriched_state type: {type(enriched_state)}")
        return [], False

    rows = []
    has_unsupported = False

    for u in state_list:
        if not isinstance(u, dict):
            continue

        script_type = u.get("script_type", "").strip()
        selected = bool(u.get("selected", False))
        inferred = bool(u.get("script_type_inferred", False))
        is_legacy = bool(u.get("is_legacy", False))

        # Force legacy unselected (safety + UX)
        if is_legacy:
            selected = False
            has_unsupported = True

        supported = script_type in ("P2WPKH", "Taproot", "P2TR")

        # Health badge
        if not supported or is_legacy:
            has_unsupported = True
            if is_legacy or script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;background:rgba(255,68,68,0.12);padding:6px;border-radius:6px;">'
                    '<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ LEGACY</span><br>'
                    '<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported for PSBT – migrate first</small>'
                    '</div>'
                )
            elif script_type in ("P2SH-P2WPKH", "Nested"):
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;background:rgba(255,153,0,0.12);padding:6px;border-radius:6px;">'
                    '<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ NESTED</span><br>'
                    '<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported yet</small>'
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
            health = u.get("health", "OPTIMAL")
            recommend = u.get("recommend", "")
            health_html = (
                f'<div class="health health-{health.lower()}" style="padding:6px;border-radius:6px;">'
                f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{health}</span><br>'
                f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">{recommend}</small>'
                '</div>'
            )

        # Friendly type display
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

        rows.append([
            selected,
            u.get("source", "Single"),
            u.get("txid", "unknown"),
            health_html,
            u.get("value", 0),
            u.get("address", "unknown"),
            u.get("input_weight", 0),
            display_type,
            u.get("vout", 0),
        ])

    return rows, has_unsupported


# ── Simple sat → BTC string formatter ───────────────────────────────────────────
def sats_to_btc_str(sats: int) -> str:
    """Convert satoshis to human-readable BTC string (removes trailing zeros)."""
    btc = sats / 100_000_000
    if btc >= 1:
        return f"{btc:,.8f}".rstrip("0").rstrip(".") + " BTC"
    return f"{int(sats):,} sats"

# ── Privacy scoring & CIOH warnings ─────────────────────────────────────────────
def calculate_privacy_score(selected_utxos: List[dict], total_utxos: int) -> int:
    """
    Realistic 2025–2026 privacy score.
    Harsh penalties for consolidation — reflects current chain analysis reality.
    Single input = near-perfect; 2+ inputs = significant linkage risk.
    """
    n = len(selected_utxos)
    if n == 0:
        return 100  # Nothing spent → perfect privacy
    if n == 1:
        return 92   # Single input → very low new linkage

    # Steep drop-off — even small consolidations hurt badly today
    if n == 2:
        return 65
    if n <= 4:
        return 45
    if n <= 8:
        return 28
    if n <= 15:
        return 15
    if n <= 30:
        return 8
    if n <= 50:
        return 5

    # Beyond 50 inputs → privacy is effectively gone
    return max(3, 100 - 9 * n)


def estimate_coinjoin_mixes_needed(
    input_count: int,
    distinct_addrs: int,
    privacy_score: int
) -> tuple[int, int]:
    """
    Rough estimate of CoinJoin rounds needed to meaningfully restore privacy.
    Caps at realistic Whirlpool / similar limits.
    """
    if privacy_score > 80:
        return 0, 1     # Minimal linkage — optional single mix
    if privacy_score > 70:
        return 1, 2     # Light hardening recommended

    # Base: inverse of score (smoother curve)
    base = max(2, round(100 / (privacy_score + 10)))

    # Scale by input count (more inputs = more work)
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

    # Scale by distinct addresses revealed (worse than raw count)
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

    return min(12, min_mixes), min(18, max_mixes)  # Realistic upper bounds


def get_cioh_warning(
    input_count: int,
    distinct_addrs: int,
    privacy_score: int
) -> str:
    """
    Generate CIOH (Common Input Ownership Heuristic) warning HTML.
    Color-coded severity, includes recovery suggestions when needed.
    """
    if input_count <= 1:
        return ""

    min_mixes, max_mixes = estimate_coinjoin_mixes_needed(
        input_count, distinct_addrs, privacy_score
    )

    # Recovery suggestion shown when score ≤ 70
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

    # ── Extreme linkage (≤30) ───────────────────────────────────────────────────
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

    # ── High risk (≤50) ─────────────────────────────────────────────────────────
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

    # ── Moderate risk (≤70) ─────────────────────────────────────────────────────
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

    # ── Low impact (>70) ────────────────────────────────────────────────────────
    else:
        return f"""
        <div style="
            margin-top:10px !important;
            padding:12px !important;
            background:#001a00 !important;
            border:1px solid #00ff88 !important;
            border-radius:10px !important;
            color:#aaffcc !important;
            font-size:clamp(0.95rem, 3.2vw, 1.05rem) !important;
            line-height:1.6 !important;
            box-shadow:0 0 20px rgba(0,255,136,0.2) !important;
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
            {recovery_note}
        </div>
        """
# =========================
# Helper Functions for Bitcoin Addresses
# =========================

def base58_decode(s: str) -> bytes:
    """Decode Base58Check-encoded string (with leading zeros preserved)."""
    n = 0
    for c in s:
        n = n * 58 + BASE58_ALPHABET.index(c)
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + n.to_bytes((n.bit_length() + 7) // 8, 'big')

def bech32_polymod(values: list[int]) -> int:
    """Polymod computation - same generator table for both Bech32 and Bech32m"""
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
    return chk


def bech32_hrp_expand(hrp: str) -> list[int]:
    """Expand HRP for checksum calculation (BIP-173 / BIP-350)"""
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def bech32_verify_checksum(hrp: str, data: list[int]) -> bool:
    """Verify classic Bech32 checksum (BIP-173)"""
    return bech32_polymod(bech32_hrp_expand(hrp) + data) == 1


def bech32m_verify_checksum(hrp: str, data: list[int]) -> bool:
    """Verify Bech32m checksum (BIP-350, used for Taproot)"""
    return bech32_polymod(bech32_hrp_expand(hrp) + data) == 0x2bc830a3


def convertbits(data: list[int], frombits: int, tobits: int, pad: bool = True) -> list[int] | None:
    """Convert bits between groups with strict validation when pad=False"""
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
    else:
        # BIP-173 strict rule: leftover bits must be zero
        if bits >= frombits or ((acc << (tobits - bits)) & maxv):
            return None

    return ret


def address_to_script_pubkey(addr: str) -> tuple[bytes, dict]:
    """
    Convert Bitcoin address → scriptPubKey + metadata.
    Supports: P2PKH, P2SH, P2WPKH, P2WSH, P2TR (Taproot).
    Returns fallback on invalid input.
    """
    addr = (addr or "").strip().lower()
    fallback_spk = b'\x00\x14' + b'\x00' * 20
    fallback_meta = {'input_vb': 68, 'output_vb': 31, 'type': 'unknown'}

    if not addr:
        return fallback_spk, fallback_meta

    # ── Legacy P2PKH (1...) ────────────────────────────────────────────────────
    if addr.startswith('1'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x00:
                hash160 = dec[1:21]
                return (
                    b'\x76\xa9\x14' + hash160 + b'\x88\xac',
                    {'input_vb': 148, 'output_vb': 34, 'type': 'P2PKH'}
                )
        except (ValueError, IndexError) as e:
            log.warning("P2PKH decode failed for %s: %s", addr, str(e))
        return fallback_spk, {'input_vb': 148, 'output_vb': 34, 'type': 'invalid'}

    # ── P2SH (3...) ────────────────────────────────────────────────────────────
    if addr.startswith('3'):
        try:
            dec = base58_decode(addr)
            if len(dec) == 25 and dec[0] == 0x05:
                hash160 = dec[1:21]
                return (
                    b'\xa9\x14' + hash160 + b'\x87',
                    {'input_vb': 91, 'output_vb': 32, 'type': 'P2SH'}
                )
        except (ValueError, IndexError) as e:
            log.warning("P2SH decode failed for %s: %s", addr, str(e))
        return fallback_spk, {'input_vb': 91, 'output_vb': 32, 'type': 'invalid'}

    # ── Bech32 / Bech32m (bc1...) ──────────────────────────────────────────────
    if addr.startswith('bc1'):
        hrp = 'bc'
        data_part = addr[3:]
        data = [CHARSET.find(c) for c in data_part]
        if -1 in data:
            log.warning("Invalid charset in Bech32 address: %s", addr)
            return fallback_spk, fallback_meta

        if bech32_verify_checksum(hrp, data):
            witness_version = 0
            log.debug("Valid classic Bech32 (v0) checksum for: %s", addr)
        elif bech32m_verify_checksum(hrp, data):
            witness_version = 1
            log.debug("Valid Bech32m (v1+) checksum for: %s", addr)
        else:
            polymod_val = bech32_polymod(bech32_hrp_expand(hrp) + data)
            log.warning("Checksum failed for both Bech32 and Bech32m: %s (polymod=0x%08x)", addr, polymod_val)
            return fallback_spk, fallback_meta

        if len(data) < 7:
            log.warning("Bech32 address too short: %s", addr)
            return fallback_spk, fallback_meta

        prog_data = data[1 : -6]  # drop version (first) + checksum (last 6)

        prog = convertbits(prog_data, 5, 8, pad=False)
        if prog is None:
            log.warning("convertbits failed after dropping checksum (leftover/invalid bits): %s", addr)
            return fallback_spk, fallback_meta

        # ── Valid cases ────────────────────────────────────────────────────────
        if witness_version == 0:
            if len(prog) == 20:
                log.debug("Valid P2WPKH: %s", addr)
                return b'\x00\x14' + bytes(prog), {'input_vb': 68, 'output_vb': 31, 'type': 'P2WPKH'}
            if len(prog) == 32:
                log.debug("Valid P2WSH: %s", addr)
                return b'\x00\x20' + bytes(prog), {'input_vb': 69, 'output_vb': 43, 'type': 'P2WSH'}

        if witness_version == 1 and len(prog) == 32:
            log.debug("Valid Taproot (P2TR): %s", addr)
            return b'\x51\x20' + bytes(prog), {'input_vb': 57, 'output_vb': 43, 'type': 'Taproot'}

        log.warning("Unsupported witness version %d or program length %d: %s",
                    witness_version, len(prog), addr)
        return fallback_spk, fallback_meta

    # ── Fallback for unrecognized format ───────────────────────────────────────
    log.warning("Unrecognized address format: %s", addr)
    return fallback_spk, fallback_meta
	
# =========================
# Transaction Economics (Single Source of Truth)
# =========================

@dataclass(frozen=True)
class TxEconomics:
    total_in: int          # Sum of selected UTXO values (satoshis)
    vsize: int             # Final transaction virtual size (vbytes)
    fee: int               # Final miner fee (includes any absorbed dust/change)
    change_amt: int        # Amount returned as change output (0 if dust or no destination)


def estimate_tx_economics(
    selected_utxos: List[Dict],
    fee_rate: int,
) -> TxEconomics:
    """
    Estimate transaction economics — prioritizes returning meaningful change.
    Uses modern 2025+ heuristics to avoid creating new dust outputs.
    """
    if not selected_utxos:
        raise ValueError("No UTXOs selected")

    total_in = sum(u["value"] for u in selected_utxos)
    if total_in <= 0:
        raise ValueError("Total input value is zero or negative")

    input_weight = sum(u["input_weight"] for u in selected_utxos)
    input_count = len(selected_utxos)

    # Base SegWit vsize calculation (accurate for P2WPKH/Taproot inputs)
    base_vsize = (input_weight + 43 * 4 + 31 * 4 + 10 * 4 + input_count) // 4 + 10

    # Conservative fallback (covers worst-case witness estimation)
    conservative_vsize = (input_weight + 150 + input_count * 60) // 4

    vsize = max(base_vsize, conservative_vsize)

    # Fee with minimum ~1 sat/vB effective rate
    fee = max(600, int(vsize * fee_rate))

    remaining_after_fee = total_in - fee
    if remaining_after_fee < 0:
        raise ValueError(f"Fee exceeds total input value ({fee:,} > {total_in:,} sats)")

    # ── 2025 best practice: avoid creating new dust ─────────────────────────────
    MIN_REASONABLE_CHANGE = 10_000   # ~0.0001 BTC — considered dust by most wallets today
    DUST_THRESHOLD = 546             # Classic dust limit (not used for change decision)

    change_amt = remaining_after_fee if remaining_after_fee >= MIN_REASONABLE_CHANGE else 0

    # Absorb tiny remainder into fee (cleanup mode)
    if remaining_after_fee > 0 and change_amt == 0:
        fee += remaining_after_fee

    return TxEconomics(
        total_in=total_in,
        vsize=vsize,
        fee=fee,
        change_amt=change_amt,
    )


# ── Transaction serialization helpers ───────────────────────────────────────────

def encode_varint(i: int) -> bytes:
    """Encode variable-length integer (CompactSize) per Bitcoin protocol."""
    if i < 0xfd:
        return bytes([i])
    if i < 0x10000:
        return b'\xfd' + i.to_bytes(2, 'little')
    if i < 0x100000000:
        return b'\xfe' + i.to_bytes(4, 'little')
    return b'\xff' + i.to_bytes(8, 'little')


@dataclass
class TxIn:
    """Unsigned transaction input (for PSBT)."""
    prev_tx: bytes
    prev_index: int
    sequence: int = 0xfffffffd   # RBF enabled by default

    def serialize(self) -> bytes:
        """Serialize input for unsigned tx (empty scriptSig)."""
        return (
            self.prev_tx[::-1] +                     # txid reversed
            self.prev_index.to_bytes(4, 'little') +
            b'\x00' +                                 # empty scriptSig length
            self.sequence.to_bytes(4, 'little')
        )


@dataclass
class TxOut:
    """Transaction output."""
    amount: int
    script_pubkey: bytes

    def serialize(self) -> bytes:
        """Serialize output (amount + scriptPubKey)."""
        return (
            self.amount.to_bytes(8, 'little') +
            encode_varint(len(self.script_pubkey)) +
            self.script_pubkey
        )


@dataclass
class Tx:
    """Minimal unsigned transaction structure for PSBT creation."""
    version: int = 2
    tx_ins: List[TxIn] = field(default_factory=list)
    tx_outs: List[TxOut] = field(default_factory=list)
    locktime: int = 0

    def serialize_unsigned(self) -> bytes:
        """
        Serialize transaction in legacy format (no witness marker, empty scriptSigs).
        Required format for PSBT global unsigned transaction field.
        """
        return (
            self.version.to_bytes(4, 'little') +
            encode_varint(len(self.tx_ins)) +
            b''.join(tx_in.serialize() for tx_in in self.tx_ins) +
            encode_varint(len(self.tx_outs)) +
            b''.join(tx_out.serialize() for tx_out in self.tx_outs) +
            self.locktime.to_bytes(4, 'little')
        )


# =========================
# PSBT Creation
# =========================

def create_psbt(tx: Tx, utxos: list[dict]) -> tuple[str, str]:
    """
    Build a hardware-signable PSBT (P2WPKH + Taproot only).
    Uses minimal UTXO data (txid/vout/value/scriptPubKey).
    BIP32 derivation is optional — added only when all fields are present.
    """
    import base64

    def encode_varint(i: int) -> bytes:
        if i < 0xfd:
            return bytes([i])
        if i <= 0xffff:
            return b'\xfd' + i.to_bytes(2, "little")
        if i <= 0xffffffff:
            return b'\xfe' + i.to_bytes(4, "little")
        return b'\xff' + i.to_bytes(8, "little")

    def write_kv(psbt: bytearray, key_type: int, key_data: bytes, value: bytes):
        key = bytes([key_type]) + key_data
        psbt += encode_varint(len(key))
        psbt += key
        psbt += encode_varint(len(value))
        psbt += value

    def write_bip32_derivation(psbt: bytearray, pubkey: bytes, fingerprint: bytes, path_bytes: bytes):
        write_kv(psbt, 0x06, pubkey, fingerprint + path_bytes)

    # ── Validation ──────────────────────────────────────────────────────────────
    if not tx.tx_ins:
        raise ValueError("Transaction has no inputs")
    if not tx.tx_outs:
        raise ValueError("Transaction has no outputs")
    if len(tx.tx_ins) != len(utxos):
        raise ValueError(f"Input count mismatch: tx has {len(tx.tx_ins)} inputs, but {len(utxos)} UTXOs provided")

    raw_tx = tx.serialize_unsigned()
    psbt = bytearray(b"psbt\xff")

    # Global: unsigned transaction
    write_kv(psbt, 0x00, b"", raw_tx)
    psbt += b"\x00"  # end global map

    # ── Input maps ──────────────────────────────────────────────────────────────
    for u in utxos:
        stype = u.get("script_type", "").lower()
        if stype not in ("p2wpkh", "p2tr", "taproot"):
            raise ValueError(f"Unsupported script type for PSBT: {stype}")

        # ── Per-UTXO sanity checks ──────────────────────────────────────────────
        txid_short = (u.get("txid", "unknown")[:8] + "...") if u.get("txid") else "?"
        vout = u.get("vout", "?")
        ident = f"txid={txid_short} vout={vout}"

        val = u.get("value", 0)
        if val <= 0:
            raise ValueError(f"Invalid UTXO value <= 0 sats ({ident})")
        if val > 21_000_000 * 100_000_000:
            raise ValueError(f"Impossible UTXO value > 21M BTC ({val:,} sats) ({ident})")
        if "scriptPubKey" not in u:
            raise ValueError(f"Missing scriptPubKey ({ident})")
        if not isinstance(u["scriptPubKey"], (bytes, str)):
            raise ValueError(f"Invalid scriptPubKey type (must be bytes or hex str) ({ident})")

        # Witness UTXO (required for segwit/taproot)
        spk = u["scriptPubKey"]
        if isinstance(spk, str):
            spk = bytes.fromhex(spk)

        witness_utxo = (
            val.to_bytes(8, "little") +
            encode_varint(len(spk)) +
            spk
        )
        write_kv(psbt, 0x01, b"", witness_utxo)

        # ALWAYS write SIGHASH_ALL — required by most hardware wallets
        write_kv(psbt, 0x03, b"", bytes([0x01, 0x00, 0x00, 0x00]))

        # Optional BIP-32 derivation (only if full data present)
        if all(k in u for k in ["pubkey", "fingerprint", "full_derivation_path"]):
            try:
                pubkey = bytes.fromhex(u["pubkey"])
                fingerprint = bytes.fromhex(u["fingerprint"])
                path_str = u["full_derivation_path"].replace("m/", "")
                path_bytes = b""
                for part in path_str.split("/"):
                    hardened = part.endswith("'")
                    n = int(part.rstrip("'"))
                    if hardened:
                        n |= 0x80000000
                    path_bytes += n.to_bytes(4, "little")

                write_bip32_derivation(psbt, pubkey, fingerprint, path_bytes)
            except Exception as e:
                log.warning(f"Failed to add BIP32 derivation ({ident}): {e}")

        psbt += b"\x00"  # end input map

    # ── Output maps (empty for unsigned PSBT) ──────────────────────────────────
    for _ in tx.tx_outs:
        psbt += b"\x00"

    # Final PSBT separator
    psbt += b"\x00"

    psbt_b64 = base64.b64encode(psbt).decode("ascii")
    return psbt_b64, ""


# =========================
# UTXO Fetching Functions
# =========================

def get_utxos_with_timeout(addr: str, dust: int, timeout_sec: int = 60) -> List[dict]:
    """
    Fetch UTXOs with a hard per-address timeout to prevent indefinite hangs
    on large/rate-limited addresses.
    """
    def inner_fetch():
        return get_utxos(addr, dust)  # your existing function

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(inner_fetch)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            log.warning(f"UTXO fetch timed out after {timeout_sec}s for address: {addr}")
            return []
        except Exception as e:
            log.error(f"Unexpected error fetching UTXOs for {addr}: {e}")
            return []
	
def get_utxos(addr: str, dust: int = 546) -> List[dict]:
    """
    Fetch confirmed UTXOs for a Bitcoin address from multiple public APIs.
    Filters out dust below threshold. Returns early on first successful source.
    """
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
                r = safe_get(url, timeout=8)
                if not r:
                    continue

                if r.status_code == 200:
                    try:
                        data = r.json()
                        utxos = []
                        items = data if isinstance(data, list) else data.get("utxos", [])
                        for item in items:
                            val = int(item.get("value", 0))
                            if val > dust:
                                utxos.append({
                                    "txid": item["txid"],
                                    "vout": item["vout"],
                                    "value": val,
                                    "address": addr,
                                    "confirmed": (
                                        item.get("status", {}).get("confirmed", True)
                                        if isinstance(item.get("status"), dict) else True
                                    )
                                })
                        if utxos:
                            log.debug(f"Found {len(utxos)} UTXOs from {url}")
                            return utxos
                    except Exception as e:
                        log.warning(f"JSON parse error from {url}: {e}")

                elif r.status_code == 429:
                    log.warning(f"Rate limited (429) on {url}, attempt {attempt+1}")
                    time.sleep(5 * (attempt + 1))

                else:
                    log.debug(f"Bad status {r.status_code} from {url}")

            except Exception as e:
                log.warning(f"Request error on {url}: {type(e).__name__}: {e}")

            time.sleep(1.0)  # polite delay between requests

        time.sleep(5)  # longer delay between full API rotation

    log.warning(f"No UTXOs found after retries for address: {addr}")
    return []
# =================
# Analyze functions
# ================

@dataclass(frozen=True)
class AnalyzeParams:
    fee_rate: int
    future_fee_rate: int
    dust_threshold: int
    strategy: str
    offline_mode: bool
    addr_input: str
    manual_utxo_input: str
    scan_source: str


def _sanitize_analyze_inputs(
    addr_input: str,
    strategy: str,
    dust_threshold: Any,
    fee_rate_slider: Any,
    future_fee_slider: Any,
    offline_mode: Any,
    manual_utxo_input: str,
) -> AnalyzeParams:
    """
    Normalize and clamp all analyze() inputs into a deterministic frozen params object.
    Handles Gradio edge cases (None, empty strings, floats).
    """
    # Safe int conversion with fallback to default
    fee_rate = max(1, min(300, int(float(fee_rate_slider or 15))))
    future_fee_rate = max(5, min(500, int(float(future_fee_slider or 60))))
    dust_threshold = max(0, min(10000, int(float(dust_threshold or 546))))

    scan_source = (addr_input or "").strip()

    return AnalyzeParams(
        fee_rate=fee_rate,
        future_fee_rate=future_fee_rate,
        dust_threshold=dust_threshold,
        strategy=strategy,
        offline_mode=bool(offline_mode),
        addr_input=scan_source,
        manual_utxo_input=(manual_utxo_input or "").strip(),
        scan_source=scan_source,
    )


def _collect_manual_utxos(params: AnalyzeParams) -> List[Dict]:
    """
    Parse offline/manual UTXO input lines into normalized dicts.
    Format: txid:vout:value_in_sats[:address]
    Skips invalid lines, dust, and comments (#).
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

            # Optional address (for change or labeling)
            addr = parts[3].strip() if len(parts) >= 4 else "unknown (manual)"

            if value <= params.dust_threshold:
                continue

            # Derive scriptPubKey + weight from address (fallback to P2WPKH)
            _, meta = address_to_script_pubkey(addr)
            input_weight = meta.get("input_vb", 68) * 4

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
            log.debug(f"Skipped invalid manual UTXO line {line_num}: {e}")

    if skipped:
        log.info(f"Manual UTXO parse: skipped {skipped} invalid/malformed lines")

    return utxos


def _collect_online_utxos(params: AnalyzeParams) -> Tuple[List[Dict], str, List[Any]]:
    """
    Fetch UTXOs from online APIs for provided addresses.
    Returns (utxos, debug_message, _unused).
    """
    if not params.addr_input:
        return [], "No addresses provided", []

    utxos = []
    debug_lines = []

    entries = [e.strip() for e in params.addr_input.splitlines() if e.strip()]

    for entry in entries:
        if not entry.startswith(("bc1q", "bc1p", "1", "3")):
            debug_lines.append(f"Skipped invalid entry: {entry[:20]}...")
            continue

        entry_utxos = get_utxos_with_timeout(entry, params.dust_threshold, timeout_sec=60)
        count = len(entry_utxos)

        if count > 0:
            source_label = entry  # full address as source
            debug_lines.append(f"Address {entry}: {count} UTXOs found")

            for u in entry_utxos:
                u["source"] = source_label
                u["address"] = entry

            utxos.extend(entry_utxos)
        else:
            debug_lines.append(f"Address {entry}: no UTXOs")

    debug_msg = "\n".join(debug_lines) if debug_lines else "No valid addresses or UTXOs found"
    return utxos, debug_msg, []


def _classify_utxo(
    value: int,
    input_weight: int,
    script_type_from_meta: str = ""
) -> Tuple[str, str, str, bool]:
    """
    Classify UTXO health & recommendation based on weight + value.
    Returns (script_type, health, recommendation, is_legacy).
    """
    # Step 1: Base classification from weight (fallback)
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

    # Step 2: Prefer metadata type (more accurate)
    if script_type_from_meta and script_type_from_meta in (
        "Taproot", "P2WPKH", "P2SH-P2WPKH", "P2PKH", "Legacy"
    ):
        script_type = script_type_from_meta
        is_legacy = script_type in ("P2PKH", "Legacy")

        # Reset defaults based on corrected type
        if script_type in ("Taproot", "P2WPKH"):
            default_health = "OPTIMAL"
            default_rec = "KEEP"
        elif script_type == "P2SH-P2WPKH":
            default_health = "MEDIUM"
            default_rec = "CAUTION"
        else:
            default_health = "HEAVY"
            default_rec = "PRUNE"

    # Step 3: Value overrides (highest priority)
    if value < 10_000:
        return script_type, "DUST", "PRUNE", is_legacy

    if value > 100_000_000 and is_legacy:
        return script_type, "CAREFUL", "OPTIONAL", is_legacy

    # Fallback to base classification
    return script_type, default_health, default_rec, is_legacy


def _enrich_utxos(raw_utxos: list[dict], params: AnalyzeParams) -> list[dict]:
    """
    Enrich raw UTXOs with script metadata, weight, health, recommendation,
    scriptPubKey, optional Taproot internal key, and legacy flag.
    """
    enriched = []

    for u in raw_utxos:
        addr = u.get("address", "")

        # Derive scriptPubKey + metadata
        script_pubkey, meta = address_to_script_pubkey(addr)
        input_weight = meta.get("input_vb", 68) * 4  # fallback to P2WPKH

        script_type_from_meta = meta.get("type", "")

        # Classify
        script_type, health, recommend, is_legacy = _classify_utxo(
            u["value"], input_weight, script_type_from_meta
        )

        # Normalize Taproot naming
        if script_type.upper() in ("TAPROOT", "P2TR"):
            script_type = "Taproot"

        # Optional Taproot internal key
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
            "scriptPubKey": script_pubkey,          # bytes — needed for PSBT
            "tap_internal_key": tap_internal_key,   # optional
            "is_legacy": is_legacy,                 # for table styling / auto-disable
            # "selected" set later by strategy or user
        })

    return enriched
    
# ── Apply pruning strategy to enriched UTXOs ────────────────────────────────────
def _apply_pruning_strategy(enriched: List[Dict], strategy: str) -> List[Dict]:
    """
    Apply deterministic pruning strategy to enriched UTXOs.
    - Excludes legacy/unsupported from pruning calculation
    - Prunes only supported UTXOs (priority: dust → heavy → optimal)
    - Returns list with pruned supported first, unsupported last (always unselected)
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

    # Apply selection flags
    result_supported = []
    for i, u in enumerate(by_health):
        new_u = dict(u)
        new_u["selected"] = i < prune_count
        result_supported.append(new_u)

    # Step 3: Combine — supported (pruned) first, unsupported last
    result = result_supported + unsupported

    # Log summary (no print)
    legacy_skipped = len(unsupported)
    if legacy_skipped > 0:
        log.info(
            f"Excluded {legacy_skipped} legacy/unsupported from pruning. "
            f"Pruned {prune_count} of {total_supported} supported UTXOs."
        )

    return result


# ── Build DataFrame rows from enriched UTXOs ────────────────────────────────────
def _build_df_rows(enriched: List[Dict]) -> tuple[List[List], bool]:
    """
    Convert enriched UTXOs into Gradio DataFrame rows.
    - Forces legacy/nested unselected with strong warnings
    - Preserves 'selected' from strategy or restore
    - Fully responsive health badges & type display
    Returns (rows, has_unsupported)
    """
    rows: List[List] = []
    has_unsupported = False

    for u in enriched:
        if not isinstance(u, dict):
            continue

        script_type = u.get("script_type", "").strip()
        selected = bool(u.get("selected", False))
        inferred = bool(u.get("script_type_inferred", False))
        is_legacy = bool(u.get("is_legacy", False))

        # Force legacy unselected (safety + UX)
        if is_legacy:
            selected = False
            has_unsupported = True

        supported = script_type in ("P2WPKH", "Taproot", "P2TR")

        # Health badge HTML
        if not supported or is_legacy:
            has_unsupported = True
            if is_legacy or script_type in ("P2PKH", "Legacy"):
                health_html = (
                    '<div class="health health-legacy" style="color:#ff4444;font-weight:bold;background:rgba(255,68,68,0.12);padding:6px;border-radius:6px;">'
                    '<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ LEGACY</span><br>'
                    '<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported in PSBT — migrate first</small>'
                    '</div>'
                )
            elif script_type in ("P2SH-P2WPKH", "Nested"):
                health_html = (
                    '<div class="health health-nested" style="color:#ff9900;font-weight:bold;background:rgba(255,153,0,0.12);padding:6px;border-radius:6px;">'
                    '<span style="font-size:clamp(1rem,4vw,1.2rem);">⚠️ NESTED</span><br>'
                    '<small style="font-size:clamp(0.8rem,3vw,0.9rem);">Not supported yet</small>'
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
            health = u.get("health", "OPTIMAL")
            recommend = u.get("recommend", "")
            health_html = (
                f'<div class="health health-{health.lower()}" style="padding:6px;border-radius:6px;">'
                f'<span style="font-size:clamp(1rem,4vw,1.2rem);">{health}</span><br>'
                f'<small style="font-size:clamp(0.85rem,3vw,0.95rem);">{recommend}</small>'
                '</div>'
            )

        # Friendly type display
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

        rows.append([
            selected,
            u.get("source", "Single"),
            u.get("txid", "unknown"),
            health_html,
            u.get("value", 0),
            u.get("address", "unknown"),
            u.get("input_weight", 0),
            display_type,
            u.get("vout", 0),
        ])

    return rows, has_unsupported


# ── Freeze enriched state for immutability ──────────────────────────────────────
def _freeze_enriched(
    enriched: List[Dict],
    *,
    strategy: str,
    scan_source: str,
) -> tuple:
    """
    Freeze enriched UTXOs into immutable tuple + metadata.
    Enforces your canonical state model: (meta, tuple[dict]).
    """
    frozen_utxos = tuple(copy.deepcopy(u) for u in enriched)

    meta = {
        "strategy": strategy,
        "scan_source": scan_source,
        "timestamp": int(time.time()),
    }

    return (meta, frozen_utxos)

# =================
# Analyze functions
# =================

def analyze(
    addr_input: str,
    strategy: str,
    dust_threshold: Any,
    fee_rate_slider: Any,
    future_fee_slider: Any,
    offline_mode: Any,
    manual_utxo_input: str,
):
    """
    Main entrypoint: sanitize inputs → collect UTXOs → enrich → prune → build UI outputs.
    Enforces the canonical state model (immutable enriched_state tuple).
    """
    log.info("analyze() started")

    # 1. Sanitize & normalize all inputs
    params = _sanitize_analyze_inputs(
        addr_input=addr_input,
        strategy=strategy,
        dust_threshold=dust_threshold,
        fee_rate_slider=fee_rate_slider,
        future_fee_slider=future_fee_slider,
        offline_mode=offline_mode,
        manual_utxo_input=manual_utxo_input,
    )

    # 2. Collect raw UTXOs (online or offline)
    if params.offline_mode:
        raw_utxos = _collect_manual_utxos(params)
        scan_debug = "Offline mode — manual UTXOs only"
        log.debug(f"Offline mode: collected {len(raw_utxos)} manual UTXOs")
    else:
        raw_utxos, scan_debug, _ = _collect_online_utxos(params)
        log.debug(f"Online mode: collected {len(raw_utxos)} raw UTXOs")
        if raw_utxos:
            log.debug(f"First UTXO address: {raw_utxos[0].get('address', 'unknown')}")

    # Early exit if nothing found
    if not raw_utxos:
        log.info("No UTXOs found — returning empty state")
        
        addr = params.addr_input.strip()
        is_legacy_attempt = bool(addr and addr[0] in ('1', '3'))
        
        if is_legacy_attempt:
            # Legacy/nested case — use the orange legacy-specific banner
            return _analyze_empty(
                scan_source=params.scan_source,
                is_legacy_attempt=True
            )
        
        else:
            # Non-legacy case: probably timeout, rate-limit, or empty modern address
            timeout_msg = """
            <div style='
                color:#aaffcc !important;
                padding: clamp(24px, 5vw, 40px) !important;
                background:#001122 !important;
                border:3px solid #00ccff !important;
                border-radius:16px !important;
                text-align:center !important;
                max-width:95% !important;
                margin: clamp(20px, 5vw, 40px) auto !important;
                box-shadow:0 0 40px rgba(0,204,255,0.4) !important;
            '>
                <div style='
                    color:#00ccff !important;
                    font-size: clamp(1.4rem, 5vw, 1.8rem) !important;
                    font-weight:900 !important;
                    margin-bottom: clamp(12px, 3vw, 20px) !important;
                '>
                    ⚠️ Fetch timed out or rate-limited
                </div>
                
                <div style='
                    color:#88ffdd !important;
                    font-size: clamp(1rem, 3.8vw, 1.3rem) !important;
                    line-height:1.6 !important;
                '>
                    The address may be very large, or public APIs are currently rate-limiting / slow.<br><br>
                    
                    <span style='font-weight:900 !important; color:#ffffff !important;'>Options:</span><br>
                    • Try again in a few minutes (APIs often recover)<br>
                    • Use offline mode and paste raw UTXOs manually<br>
                    • Check the address on a block explorer first<br><br>
                    
                    <small style='color:#66ccff !important;'>
                        Tip: For very large wallets, offline mode is usually faster and more reliable.
                    </small>
                </div>
            </div>
            """
            
            return (
                gr.update(value=[]),                   # 0: Empty DataFrame
                (),                                    # 1: Empty enriched_state
                gr.update(value=timeout_msg),          # 2: Timeout/rate-limit warning
                gr.update(visible=False),              # 3
                gr.update(visible=False),              # 4
                params.scan_source,                    # 5
                "",                                    # 6
                gr.update(visible=False),              # 7
                gr.update(visible=True),               # 8: Keep analyze button
            )

    # 3. Enrich UTXOs with metadata, health, script info
    enriched = _enrich_utxos(raw_utxos, params)
    log.debug(f"Enriched {len(enriched)} UTXOs")

    # Invariant check: every UTXO must have script_type
    if any("script_type" not in u for u in enriched):
        log.error("Missing 'script_type' in enriched UTXOs — invariant violation")
        raise RuntimeError("Missing 'script_type' in enriched UTXOs — invariant violation")

    # 4. Apply pruning strategy (sets 'selected' flags)
    enriched_pruned = _apply_pruning_strategy(enriched, params.strategy)
    log.debug(
        f"After pruning strategy: {len(enriched_pruned)} UTXOs total, "
        f"{sum(1 for u in enriched_pruned if u['selected'])} selected"
    )

    # Safety assertions (protect canonical model invariants)
    assert len(enriched_pruned) >= MIN_KEEP_UTXOS, "Too few UTXOs after pruning"
    assert any(not u["selected"] for u in enriched_pruned), "All UTXOs selected — invalid prune"
    assert params.strategy in PRUNING_RATIOS, f"Unknown strategy: {params.strategy}"

    # 5. Build DataFrame rows for display
    df_rows, has_unsupported = _build_df_rows(enriched_pruned)
    log.debug(f"Built {len(df_rows)} table rows, has_unsupported={has_unsupported}")

    # Build warning banner if needed
    warning_banner = ""
    if has_unsupported:
        # Legacy/nested warning — you can expand this later
        warning_banner = (
            "<div style='color:#ffdd88;padding:20px;background:#332200;border:3px solid #ff9900;border-radius:14px;text-align:center;'>"
            "⚠️ Some inputs are legacy or nested — not included in PSBT<br>"
            "Only Native SegWit (bc1q…) and Taproot (bc1p…) are supported for pruning."
            "</div>"
        )

    # 6. Freeze the enriched state (immutable tuple)
    frozen_state = _freeze_enriched(
        enriched_pruned,
        strategy=params.strategy,
        scan_source=params.scan_source,
    )

    # Return unified success state (exactly 9 outputs for Gradio)
    return _analyze_success(
        df_rows=df_rows,
        frozen_state=frozen_state,
        scan_source=params.scan_source,
        warning_banner=warning_banner,
    )


def _analyze_success(
    df_rows,
    frozen_state,
    scan_source,
    warning_banner: str = ""
):
    """Unified success return — exactly 9 outputs to match Gradio .click() handler."""
    return (
        gr.update(value=df_rows),              # 0: DataFrame rows (UTXO table)
        frozen_state,                          # 1: enriched_state (frozen tuple)
        gr.update(value=warning_banner),       # 2: Legacy/nested warning banner
        gr.update(visible=True),               # 3: Show generate_row (PSBT button)
        gr.update(visible=True),               # 4: Show import_file (JSON load area)
        scan_source,                           # 5: scan_source state
        "",                                    # 6: Reserved/placeholder (was debug/status)
        gr.update(visible=True),               # 7: Show load_json_btn
        gr.update(visible=False),              # 8: Hide analyze_btn after success
    )


def _analyze_empty_legacy_warning(scan_source: str = "") -> tuple:
    """Special empty state for legacy/nested addresses that failed to load UTXOs."""
    legacy_msg = """
    <div style='
        color:#ffdd88 !important;
        padding: clamp(24px, 5vw, 40px) !important;
        background:#332200 !important;
        border:3px solid #ff9900 !important;
        border-radius:16px !important;
        text-align:center !important;
        max-width:95% !important;
        margin: clamp(20px, 5vw, 40px) auto !important;
        box-shadow:0 0 40px rgba(255,153,0,0.5) !important;
    '>
        <div style='
            color:#ffff66 !important;
            font-size: clamp(1.4rem, 5vw, 1.8rem) !important;
            font-weight:900 !important;
            margin-bottom: clamp(12px, 3vw, 20px) !important;
        '>
            ⚠️ Large Legacy / Nested Address Detected (1… or 3…)
        </div>
        
        <div style='
            color:#ffcc88 !important;
            font-size: clamp(1rem, 3.8vw, 1.3rem) !important;
            line-height:1.6 !important;
        '>
            This address appears to have thousands of UTXOs (common for old wallets or exchanges).<br><br>
            
            Public block explorers usually refuse, timeout, or rate-limit queries for such large legacy/nested addresses.<br><br>
            
            <strong style='color:#ffffff !important;'>Important:</strong><br>
            Even if the full list loaded, <strong>legacy (1…) and nested (3…) inputs cannot be pruned</strong> with this tool — they are not supported in the generated PSBT.<br><br>
            
            To prune or consolidate:<br>
            • Migrate to a modern address first (bc1q… Native SegWit or bc1p… Taproot)<br>
            • Then re-analyze here<br><br>
            
            <small style='color:#ffaa66 !important;'>
                Tip: Use offline mode and paste even one example UTXO line to see the legacy warning and table styling.
            </small>
        </div>
    </div>
    """
    
    return (
        gr.update(value=[]),                   # 0: Empty DataFrame
        (),                                    # 1: Empty enriched_state
        gr.update(value=legacy_msg),           # 2: Special legacy warning banner
        gr.update(visible=False),              # 3: Hide generate_row
        gr.update(visible=False),              # 4: Hide import_file
        scan_source,                           # 5: Preserve scan_source
        "",                                    # 6: Reserved/placeholder
        gr.update(visible=False),              # 7: Hide load_json_btn on failure
        gr.update(visible=True),               # 8: Keep analyze_btn visible (retry)
    )
    

def _analyze_empty(scan_source: str = "", is_legacy_attempt: bool = False) -> tuple:
    """
    Empty/failure state return — exactly 9 outputs.
    Can show legacy-specific warning if we suspect it's a large legacy/nested address.
    """
    if is_legacy_attempt and scan_source.strip().startswith(('1', '3')):
        return _analyze_empty_legacy_warning(scan_source)
    
    # Default generic empty state (your original)
    return (
        gr.update(value=[]),                   # 0
        (),                                    # 1
        gr.update(value=""),                   # 2: No warning banner
        gr.update(visible=False),              # 3
        gr.update(visible=False),              # 4
        scan_source,                           # 5
        "",                                    # 6
        gr.update(visible=False),              # 7
        gr.update(visible=True),               # 8
    )
# ====================
# generate_summary_safe() — Refactored for Clarity
# ====================

def _render_locked_state() -> Tuple[str, gr.update]:
    """Locked state message — shown after successful PSBT generation."""
    return (
        "<div style='"
        "text-align:center !important;"
        "padding: clamp(40px, 10vw, 80px) !important;"
        "color:#dddddd !important;"                     # Brighter gray for visibility
        "font-size: clamp(1.2rem, 5vw, 1.8rem) !important;"
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
    Resolve selected UTXOs from Gradio DataFrame checkboxes.
    Prioritizes fresh UI state over frozen flags; strict in offline mode.
    """
    if not utxos:
        return None, 0, "NO_UTXOS"

    utxos = [u for u in utxos if isinstance(u, dict)]
    if not utxos:
        return None, 0, "NO_UTXOS"

    selected_indices = []
    if isinstance(df_rows, list) and df_rows:
        for i, row in enumerate(df_rows):
            if not row or len(row) <= CHECKBOX_COL:
                continue
            checkbox_val = row[CHECKBOX_COL]
            is_checked = checkbox_val in (True, 1, "true", "True", "1") or bool(checkbox_val)
            if is_checked and i < len(utxos):
                selected_indices.append(i)

    has_valid_df = df_rows and len(df_rows) == len(utxos)

    if has_valid_df:
        selected_utxos = [utxos[i] for i in selected_indices]
    else:
        selected_utxos = [u for u in utxos if u.get("selected", False)]

    if offline_mode and not selected_indices:
        selected_utxos = []

    pruned_count = len(selected_utxos)
    if pruned_count == 0:
        return None, 0, "NO_SELECTION"

    return selected_utxos, pruned_count, None


def _compute_privacy_metrics(selected_utxos: List[dict], total_utxos: int) -> Tuple[int, str]:
    """Compute privacy score and UI color."""
    score = calculate_privacy_score(selected_utxos, total_utxos)
    color = "#0f0" if score >= 70 else "#ff9900" if score >= 40 else "#ff3366"
    return score, color


def _compute_economics_safe(selected_utxos: List[dict], fee_rate: int) -> Optional[TxEconomics]:
    """Safe economics calculation — logs failure and returns None."""
    try:
        return estimate_tx_economics(selected_utxos, fee_rate)
    except ValueError as e:
        log.warning(f"Economics failed: {e}")
        return None


def _render_small_prune_warning(econ: TxEconomics, fee_rate: int) -> str:
    """Warning when pruned value is too small for meaningful change output."""
    remainder = econ.total_in - econ.fee
    current_fee = econ.fee

    if remainder >= 15000:
        return ""

    if remainder < 8000:
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
      Post-fee remainder (~{remainder:,} sats) is small.<br>
      Pruned value will likely be fully or partially absorbed into miner fees.<br><br>
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
        💡 For reliable change output, aim for:<br>
        • Value Pruned > ~5× Current Fee (good change)<br>
        • Value Pruned > ~10× Current Fee (very comfortable)<br><br>
        This prune: <span style="color:#ffffff !important; font-weight:800 !important;">{sats_to_btc_str(econ.total_in)}</span> value and 
        <span style="color:#ffffff !important; font-weight:800 !important;">{current_fee:,} sats</span> fee<br>
        Ratio: <span style="color:#ffffff !important; font-weight:800 !important;">{ratio}×</span> current fee
      </div><br>
      <small style="color:#88ffcc !important; font-size: clamp(0.9rem, 3vw, 1rem) !important;">
        💡 Pro tip: Bigger prune (relative to fee) → more change back. Small prunes = cleanup only.
      </small>
    </div>
    """


def _render_pruning_explanation(pruned_count: int, remaining_utxos: int) -> str:
    """Educational explanation of pruning benefits."""
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
    locked,
    strategy,
    dest_value,
    offline_mode,
) -> tuple:
    """
    Generate main status/summary HTML — the central UI feedback loop.
    Returns (status_html, generate_row_visibility).
    """
    if locked:
        return _render_locked_state()

    # Extract UTXOs from frozen state
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

    # Hard errors
    if error == "NO_UTXOS":
        return no_utxos_msg, gr.update(visible=False)
    if error == "NO_SELECTION":
        return select_msg, gr.update(visible=False)

    # Filter to supported inputs (offline-safe)
    supported_selected = [
        u for u in selected_utxos
        if u.get("script_type") in ("P2WPKH", "Taproot", "P2TR")
    ]

    offline_inferred_count = sum(
        1 for u in supported_selected if u.get("script_type_inferred", False)
    )

    pruned_count = len(supported_selected)
    if pruned_count == 0:
        return select_msg, gr.update(visible=False)

    remaining_utxos = total_utxos - pruned_count

    privacy_score, score_color = _compute_privacy_metrics(supported_selected, total_utxos)
    econ = _compute_economics_safe(supported_selected, fee_rate)

    if econ is None:
        return (
            "<div style='text-align:center; padding:30px; background:#440000; border:2px solid #ff3366; border-radius:16px; color:#ffaa88; max-width:95%; margin:0 auto;'>"
            "<strong style='color:#ff3366; font-size:1.8rem;'>Transaction Invalid</strong><br><br>"
            "Could not compute economics — please re-analyze."
            "</div>",
            gr.update(visible=False)
        )

    # Pre-prune wallet size estimate
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

    # Future savings estimate
    sats_saved = max(0, econ.vsize * (future_fee_rate - fee_rate))

    # Render components
    small_warning = _render_small_prune_warning(econ, fee_rate)
    cioh_warning = get_cioh_warning(
        pruned_count,
        len({u["address"] for u in supported_selected}),
        privacy_score
    )
    pruning_explanation = _render_pruning_explanation(pruned_count, remaining_utxos)

    strategy_label = strategy.split(" — ")[0] if " — " in strategy else "Recommended"

    # Change output message
    if econ.change_amt > 0:
        change_line = (
            f"💧 Expected output: "
            f"<span style='color:#0f0 !important;font-weight:800 !important;'>{econ.change_amt:,} sats</span> "
            "change sent to standard address"
        )
    else:
        if offline_mode and not any(u.get("address") for u in utxos):
            change_line = (
                f"💧 <span style='color:#ff9900 !important;font-weight:800 !important;'>No change output</span> "
                "(all remaining value absorbed into fees — full cleanup)"
            )
        else:
            change_line = (
                f"💧 <span style='color:#ff9900 !important;font-weight:800 !important;'>No change output</span> "
                "(remaining value below dust threshold — absorbed into fee)"
            )

    # Offline address warning (fallback if no bc1q/bc1p address found)
    offline_address_warning = ""
    if offline_mode:
        has_modern_addr = any(
            u.get("address")
            and isinstance(u["address"], str)
            and u["address"].strip().startswith(('bc1q', 'bc1p'))
            for u in utxos
        )
        if not has_modern_addr:
            offline_address_warning = (
                "<div style='"
                "color:#ffdd88 !important; background:#332200 !important; "
                "padding:16px; margin:20px 0; border:3px solid #ff9900 !important; "
                "border-radius:14px; text-align:center; font-weight:700;"
                "'>"
                "⚠️ Offline mode: No modern change address detected<br><br>"
                "Include at least one bc1q… or bc1p… address in your pasted UTXOs to receive change.<br>"
                "Without it, all remaining value will be absorbed into fees (full cleanup)."
                "</div>"
            )

    # Offline inferred script badge (if any)
    offline_badge = ""
    if offline_mode and offline_inferred_count > 0:
        offline_badge = (
            "<div style='"
            "color:#bb86fc !important; background:rgba(187,134,252,0.15) !important; "
            "padding:12px; margin:12px 0; border:2px solid #bb86fc !important; "
            "border-radius:12px; text-align:center; font-weight:700;"
            "'>"
            f"⚠️ {offline_inferred_count} script type(s) inferred from offline input<br>"
            "Review addresses carefully — accuracy depends on your paste."
            "</div>"
        )

    # Main status box HTML
    status_box_html = offline_address_warning + f"""
    <div style="
        text-align:center !important;
        margin:clamp(30px, 8vw, 60px) auto 20px auto !important;
        padding:clamp(24px, 6vw, 40px) !important;
        background: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(12px) !important;
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
              💡 After pruning: your full wallet spend size drops to roughly 
              <span style="color:#aaffcc !important;font-weight:700 !important;">~{pre_vsize - econ.vsize + 200:,} vB</span>
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
              <span style="color:#0f0 !important;font-weight:800 !important;">{econ.fee:,} sats @ {fee_rate} s/vB</span>
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
    """
    Safely extract currently selected UTXOs from the frozen enriched_state.
    Respects the canonical model: only uses 'selected' flags from immutable tuple.
    """
    if not enriched_state:
        return []

    # Unpack frozen state (invariant: always (meta, tuple[dict]))
    _, utxos = enriched_state if isinstance(enriched_state, tuple) and len(enriched_state) == 2 else (None, enriched_state)

    return [u for u in utxos if u.get("selected", False)]


def _create_psbt_snapshot(
    selected_utxos: List[dict],
    scan_source: str,
    dest_override: Optional[str],
    fee_rate: int,
    future_fee_rate: int,
) -> dict:
    """
    Create deterministic, audit-friendly JSON snapshot of user selection.
    Used for export, reload, and fingerprint verification.
    """
    if not selected_utxos:
        raise ValueError("No UTXOs selected for snapshot")

    # Sort for deterministic order
    sorted_utxos = sorted(selected_utxos, key=lambda u: (u["txid"], u["vout"]))

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
        for u in sorted_utxos
    ]

    snapshot = {
        "version": 1,
        "timestamp": int(time.time()),
        "scan_source": scan_source.strip(),
        "dest_addr_override": dest_override.strip() if dest_override else None,
        "fee_rate": fee_rate,
        "future_fee_rate": future_fee_rate,
        "inputs": clean_inputs,
    }

    # Deterministic fingerprint (SHA-256 of canonical JSON)
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    snapshot["fingerprint"] = fingerprint
    snapshot["fingerprint_short"] = fingerprint[:16].upper()

    return snapshot


def _persist_snapshot(snapshot: dict) -> str:
    """Write snapshot to temporary file for Gradio download."""
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

    log.debug(f"Snapshot persisted to: {tmp_file.name}")
    return tmp_file.name


def on_generate(
    dest_value: str,
    fee_rate: int,
    future_fee_rate: int,
    enriched_state: tuple,
    scan_source: str,
) -> tuple:
    """
    Freeze user intent on "Generate" click.
    Returns snapshot (for JSON export), selected UTXOs, locked flag, and file path.
    """
    log.info("on_generate called")

    if not enriched_state:
        log.info("enriched_state is empty — nothing to generate")
        return None, [], gr.update(value=False), None

    selected_utxos = _extract_selected_utxos(enriched_state)
    log.info(f"Selected UTXOs count: {len(selected_utxos)}")

    if not selected_utxos:
        log.info("Generate attempted with no UTXOs selected")
        return None, [], gr.update(value=False), None

    log.debug(f"Destination override: {dest_value!r}")

    try:
        log.debug("Creating snapshot...")
        snapshot = _create_psbt_snapshot(
            selected_utxos=selected_utxos,
            scan_source=scan_source,
            dest_override=dest_value,
            fee_rate=fee_rate,
            future_fee_rate=future_fee_rate,
        )

        log.debug("Snapshot created — persisting to file...")
        file_path = _persist_snapshot(snapshot)

        log.info("Snapshot persisted — locking UI")
        return snapshot, selected_utxos, gr.update(value=True), file_path

    except Exception as e:
        log.error(f"Snapshot creation failed: {e}", exc_info=True)
        return None, [], gr.update(value=False), None
        
# ====================
# generate_psbt() helpers
# ====================

def _render_no_snapshot() -> str:
    """Error message when no snapshot exists (generate not clicked yet)."""
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
    """Error message when no UTXOs are selected for pruning."""
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
    """
    Frozen parameters extracted from snapshot for PSBT generation.
    Ensures deterministic, immutable config during tx building.
    """
    inputs: List[dict]
    scan_source: str
    dest_override: Optional[str]
    fee_rate: int
    fingerprint_short: str
    full_spend_no_change: bool = False


def _extract_psbt_params(snapshot: dict) -> PsbtParams:
    """Extract frozen parameters from saved snapshot."""
    return PsbtParams(
        inputs=snapshot["inputs"],
        scan_source=snapshot["scan_source"],
        dest_override=snapshot.get("dest_addr_override"),
        fee_rate=snapshot["fee_rate"],
        fingerprint_short=snapshot["fingerprint_short"],
        full_spend_no_change=snapshot.get("full_spend_no_change", False),
    )


def _resolve_destination(dest_override: Optional[str], scan_source: str) -> Union[bytes, str]:
    """
    Resolve final destination address to scriptPubKey (bytes) or return user-facing error HTML.

    Logic:
    - If override provided → use it
    - Else → fall back to scan_source
    - If neither → return empty bytes (absorb remainder to fee = full cleanup)
    - Only allow modern outputs: P2WPKH (bc1q...) or Taproot (bc1p...)
    - Legacy (1...) or Nested SegWit (3...) are rejected with clear error

    Returns:
        bytes: valid scriptPubKey if modern address
        str: HTML error message if invalid / unsupported
    """
    override_clean = (dest_override or "").strip()
    source_clean = scan_source.strip()

    final_dest = override_clean if override_clean else source_clean

    # No destination → full cleanup (absorb change into fee)
    if not final_dest:
        return b''

    try:
        spk, meta = address_to_script_pubkey(final_dest)

        typ = meta.get('type', 'unknown')

        # Enforce modern change output only
        if typ not in ('P2WPKH', 'Taproot', 'P2TR'):
            return (
                "<div style='"
                "color:#ff3366 !important; "
                "padding: clamp(20px, 5vw, 32px) !important; "
                "background:#330000 !important; "
                "border:3px solid #ff3366 !important; "
                "border-radius:14px !important; "
                "box-shadow:0 0 40px rgba(255,51,102,0.5) !important; "
                "text-align:center !important; "
                "max-width:95% !important; "
                "margin:20px auto !important;"
                "'>"
                "<div style='font-size:clamp(1.3rem, 5vw, 1.7rem); font-weight:900;'>"
                "Change address must be modern"
                "</div><br>"
                "Only <span style='font-weight:900;'>bc1q… (Native SegWit)</span> or "
                "<span style='font-weight:900;'>bc1p… (Taproot)</span> allowed.<br><br>"
                f"Detected: <span style='font-weight:900;'>{typ}</span><br>"
                "Legacy (1…) or Nested (3…) outputs create very expensive change in 2025+ fee environment.<br><br>"
                "Please use a modern address for change output."
                "</div>"
            )

        # Success — log for traceability (can be removed later if desired)
        log.info("Resolved change address %s → %s (spk len=%d)", final_dest, typ, len(spk))

        return spk

    except Exception as e:
        log.error("Failed to resolve destination '%s': %s", final_dest, str(e), exc_info=True)
        return (
            "<div style='color:#ff3366 !important; padding:20px; background:#330000; border:3px solid #ff3366; border-radius:12px; text-align:center;'>"
            "Invalid destination address<br><br>"
            "Must be a valid modern Bitcoin address:<br>"
            "• <span style='font-weight:900;'>bc1q…</span> — Native SegWit<br>"
            "• <span style='font-weight:900;'>bc1p…</span> — Taproot<br><br>"
            "Legacy (1…) and Nested (3…) are not allowed for change."
            "</div>"
        )

def _build_unsigned_tx(
    inputs: list[dict],
    econ: TxEconomics,
    dest_spk: bytes,
    params: PsbtParams,
) -> tuple[Tx, list[dict], str, bool]:
    """
    Construct unsigned transaction and prepare UTXO info for PSBT.

    Returns:
        tx: Unsigned Tx object
        utxos_for_psbt: Prepared UTXO dicts for PSBT input maps
        no_change_warning: HTML warning (if no change address or absorbed)
        has_change_output: True if a real change output was added
    """
    tx = Tx()
    utxos_for_psbt: list[dict[str, Any]] = []

    for u in inputs:
        try:
            txid_bytes = bytes.fromhex(u["txid"])
            vout = int(u["vout"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid txid/vout in input: {e}")

        tx.tx_ins.append(TxIn(txid_bytes, vout))

        # Prepare UTXO for PSBT witness data
        _, meta = address_to_script_pubkey(u["address"])
        spk = meta.get("scriptPubKey") or address_to_script_pubkey(u["address"])[0]

        utxos_for_psbt.append({
            "value": u["value"],
            "scriptPubKey": spk,
            "script_type": u.get("script_type", "unknown"),
        })

    # Change / remainder logic
    change_amt = econ.change_amt
    if params.full_spend_no_change:
        change_amt = econ.total_in - econ.fee  # Force full spend (absorb everything)

    no_change_warning = ""
    has_change_output = False

    if change_amt > 0:
        if dest_spk:
            # Normal change output
            tx.tx_outs.append(TxOut(change_amt, dest_spk))
            has_change_output = True
        else:
            # No destination address — absorb remainder
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
                "To receive change, include at least one bc1q… or bc1p… address in your pasted UTXOs."
                "</div>"
            )

    return tx, utxos_for_psbt, no_change_warning, has_change_output
	
# ====================
# QR & PSBT HTML composition
# ====================

def _generate_qr(psbt_b64: str) -> Tuple[str, str]:
    """
    Generate QR code HTML for PSBT with graceful fallback for large payloads.
    Returns (qr_html, qr_warning).
    """
    # QR version 40 max ~2953 chars — safe threshold
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
            "<span style='color:#ffff66 !important; font-size: clamp(1.4rem, 6vw, 1.8rem) !important; font-weight:900 !important; text-shadow:0 0 35px #ffff00 !important;'>"
            "PSBT Too Large for QR Code"
            "</span><br><br>"
            f"<span style='color:#ffddaa !important; font-size: clamp(1rem, 3.8vw, 1.2rem) !important;'>"
            f"Size: {len(psbt_b64):,} characters"
            "</span><br><br>"
            "Use the <span style='color:#00ffff !important; font-size: clamp(1.2rem, 5vw, 1.5rem) !important; font-weight:900 !important; text-shadow:0 0 30px #00ffff !important;'>"
            "COPY PSBT"
            "</span> button below and paste directly into your wallet.<br><br>"
            "<span style='color:#aaffff !important; font-size: clamp(0.95rem, 3.5vw, 1.1rem) !important;'>"
            "Sparrow • Coldcard • Electrum • UniSat • Nunchuk • OKX"
            "</span>"
            "</div>"
        )
        return qr_html, qr_warning

    # Normal QR generation
    error_correction = (
        qrcode.constants.ERROR_CORRECT_L if len(psbt_b64) > 2600
        else qrcode.constants.ERROR_CORRECT_M
    )
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
    no_change_warning: str = "",
) -> str:
    """
    Compose complete PSBT output HTML (fingerprint, QR, raw PSBT, wallet notes).
    Handles change status, warnings, and large-PSBT fallbacks gracefully.
    """
    # Change status message
    change_message = ""
    if has_change_output:
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
        change_message = ""  # Big warning already present — no extra note
    else:
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

    # QR section (only if QR generated)
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

    qr_feedback = qr_warning if qr_warning else ""

    # Full PSBT output HTML
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

        <!-- Wallet support list -->
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

# ====================
# Final PSBT Generation & Summary Bridge
# ====================

def generate_psbt(psbt_snapshot: dict, full_selected_utxos: list[dict], df_rows) -> str:
    """
    Orchestrate PSBT generation using snapshot (params) and full enriched UTXOs.
    Returns full PSBT HTML output (QR, raw text, warnings, fingerprint).
    """
    if not psbt_snapshot:
        return _render_no_snapshot()

    if not full_selected_utxos:
        return _render_no_inputs()

    # Safety check: nothing actually selected
    if not any(row and row[CHECKBOX_COL] for row in df_rows if row):
        return (
            "<div style='"
            "color:#ff9900 !important;"
            "background:rgba(51,34,0,0.6) !important;"
            "border:3px solid #ff9900 !important;"
            "border-radius:14px !important;"
            "padding: clamp(24px,6vw,40px) !important;"
            "margin:20px 0 !important;"
            "text-align:center !important;"
            "font-size: clamp(1.1rem,4vw,1.3rem) !important;"
            "font-weight:700 !important;"
            "box-shadow:0 0 25px rgba(255,153,0,0.5) !important;"
            "'>"
            "⚠️ Nothing selected yet<br><br>"
            "Start over and check at least one UTXO in the table to generate the PSBT.<br>"
            "Your coins are waiting — just pick which ones to prune!"
            "</div>"
        )

    # Safe param extraction
    try:
        params = _extract_psbt_params(psbt_snapshot)
    except Exception as e:
        log.error(f"Failed to extract PSBT params: {e}", exc_info=True)
        return (
            "<div style='"
            "color:#ff6666 !important;"
            "text-align:center !important;"
            "padding: clamp(30px, 8vw, 60px) !important;"
            "background:#440000 !important;"
            "border-radius:18px !important;"
            "box-shadow:0 0 50px rgba(255,51,102,0.5) !important;"
            "font-size: clamp(1.2rem, 4.5vw, 1.6rem) !important;"
            "line-height:1.7 !important;"
            "max-width:90% !important;"
            "margin:0 auto !important;"
            "'>"
            "<span style='color:#ff3366 !important;font-size: clamp(1.4rem, 5vw, 1.8rem) !important;font-weight:900 !important;'>"
            "Invalid or corrupted snapshot"
            "</span><br><br>"
            "Please click <strong style='color:#ffaa66 !important;'>GENERATE</strong> again."
            "</div>"
        )

    # Use full enriched UTXOs
    all_inputs = full_selected_utxos

    # Filter to supported types only
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
            "Only <strong style='color:#00ffff !important;'>Native SegWit (bc1q…)</strong> and "
            "<strong style='color:#00ffff !important;'>Taproot (bc1p…)</strong> inputs can be pruned.<br><br>"
            "Legacy and Nested inputs were automatically skipped."
            "</div>"
        )

    legacy_excluded = len(supported_inputs) < len(all_inputs)

    # Strict input value sanity checks (protect against dust attacks / bad data)
    MAX_BTC = 21_000_000
    MAX_SATS = MAX_BTC * 100_000_000

    for u in supported_inputs:
        val = u.get("value", 0)
        txid_short = (u.get("txid", "unknown")[:12] + "...") if u.get("txid") else "?"
        vout = u.get("vout", "?")
        ident = f"txid={txid_short} vout={vout}"

        if val <= 0:
            return (
                f"<div style='color:#ff3366 !important; padding:20px; background:#330000; border:3px solid #ff3366; border-radius:12px; text-align:center;'>"
                f"Invalid UTXO value ≤ 0 sats detected<br>"
                f"<strong>{ident}</strong><br><br>"
                "This is likely a paste error or corrupted data. Please re-analyze or correct the UTXO value."
                "</div>"
            )

        if val > MAX_SATS:
            return (
                f"<div style='color:#ff3366 !important; padding:20px; background:#330000; border:3px solid #ff3366; border-radius:12px; text-align:center;'>"
                f"Impossible UTXO value > {MAX_BTC:,} BTC detected<br>"
                f"<strong>{val:,} sats — {ident}</strong><br><br>"
                "This exceeds the total Bitcoin supply. Likely bad input data. Please verify and re-analyze."
                "</div>"
            )

        if "scriptPubKey" not in u or not u.get("scriptPubKey"):
            return (
                f"<div style='color:#ff3366 !important; padding:20px; background:#330000; border:3px solid #ff3366; border-radius:12px; text-align:center;'>"
                f"Missing or empty scriptPubKey for input<br>"
                f"<strong>{ident}</strong><br><br>"
                "Cannot build safe PSBT without scriptPubKey. Please re-analyze or correct the UTXO entry."
                "</div>"
            )

    # Proceed — inputs look sane
    dest_result = _resolve_destination(params.dest_override, params.scan_source)
    if isinstance(dest_result, str):
        return dest_result
    dest_spk = dest_result

    try:
        econ = estimate_tx_economics(supported_inputs, params.fee_rate)
    except ValueError as e:
        log.warning(f"Economics failed in generate_psbt: {e}")
        return (
            "<div style='color:#ff6666 !important; text-align:center; padding:30px; background:#440000; border-radius:18px; box-shadow:0 0 50px rgba(255,51,102,0.5) !important;'>"
            "Invalid transaction economics — please re-analyze."
            "</div>"
        )

    # Build unsigned tx
    tx, utxos_for_psbt, no_change_warning, has_change_output = _build_unsigned_tx(
        supported_inputs,
        econ,
        dest_spk,
        params,
    )

    if len(utxos_for_psbt) != len(tx.tx_ins):
        return (
            "<div style='color:#ff6666 !important; text-align:center; padding:30px; font-size:1.2rem; font-weight:700;'>"
            "Internal error: Input/UTXO count mismatch — please report this bug."
            "</div>"
        )

    # Generate PSBT & QR
    psbt_b64, _ = create_psbt(tx, utxos_for_psbt)
    qr_html, qr_warning = _generate_qr(psbt_b64)

    # Additional warnings
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

    # Final HTML composition
    return _compose_psbt_html(
        fingerprint=params.fingerprint_short,
        qr_html=qr_html,
        qr_warning=qr_warning,
        psbt_b64=psbt_b64,
        extra_note=extra_note,
        has_change_output=has_change_output,
        no_change_warning=no_change_warning,
    )


def analyze_and_show_summary(
    addr_input,
    strategy,
    dust_threshold,
    fee_rate_slider,
    future_fee_slider,
    offline_mode,
    manual_utxo_input,
    locked,
    dest_value,
):
    """Bridge analyze() → summary rendering for UI update."""
    log.debug("analyze_and_show_summary STARTED")

    # Run core analyze
    df_update, enriched_new, warning_banner, gen_row_vis, import_vis, scan_source_new, status_box_html, load_btn_vis, analyze_btn_vis = analyze(
        addr_input,
        strategy,
        dust_threshold,
        fee_rate_slider,
        future_fee_slider,
        offline_mode,
        manual_utxo_input,
    )

    # Extract fresh rows from Gradio DataFrame update payload
    df_rows = []
    if hasattr(df_update, "value"):
        df_rows = df_update.value
    elif isinstance(df_update, dict):
        df_rows = df_update.get("value", [])
    
    # Safety net: always ensure list
    if not isinstance(df_rows, list):
        log.warning(f"df_rows from analyze() was not a list: {type(df_rows).__name__}")
        df_rows = []

    log.debug(f">>> rows built: {len(df_rows)} UTXOs")
    
    # Generate summary with fresh data
    status_box_html, generate_row_visibility = generate_summary_safe(
        df_rows,
        enriched_new,
        fee_rate_slider,
        future_fee_slider,
        locked,
        strategy,
        dest_value,
        offline_mode,
    )

    # Return in exact order expected by Gradio .click()
    return (
        df_update,
        enriched_new,
        warning_banner,
        gen_row_vis,
        import_vis,
        scan_source_new,
        status_box_html,
        load_btn_vis,
        analyze_btn_vis,
    )

# ====================
# UI Helper Functions
# ====================

def fresh_empty_dataframe():
    """Return a fresh, truly empty Gradio DataFrame with correct headers & settings."""
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
        row_count=(5, "dynamic"),      # Start with 5 rows, grow dynamically
        max_height=500,
        max_chars=None,
        label=" ",
        static_columns=[1, 2, 3, 4, 5, 6, 7, 8],  # Keep non-PRUNE columns fixed
        column_widths=["120px", "360px", "380px", "120px", "140px", "380px", "130px", "105px", "80px"]
    )


def process_uploaded_file(file):
    """
    Process uploaded JSON selection file.
    Returns parsed dict or empty {} on failure.
    """
    if not file:
        log.debug("No file provided for upload")
        return {}

    # Quick safety: reject very large files to prevent hangs / memory issues
    MAX_FILE_SIZE = 1_000_000  # 1 MB — more than enough for any realistic selection snapshot
    try:
        file_size = os.path.getsize(file.name)
        if file_size > MAX_FILE_SIZE:
            log.warning(f"Uploaded file rejected — too large ({file_size:,} bytes > {MAX_FILE_SIZE:,})")
            return {}
    except Exception as e:
        log.warning(f"Failed to check file size: {e}")
        return {}

    try:
        log.debug(f"Processing uploaded file: {file.name} ({file_size:,} bytes)")
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                log.warning("Uploaded file is empty")
                return {}
            parsed = json.loads(content)
            log.debug(f"JSON parsed OK — inputs count: {len(parsed.get('inputs', []))}")
            return parsed
    except json.JSONDecodeError as e:
        log.warning(f"JSON decode error in uploaded file: {e}")
        return {}
    except FileNotFoundError:
        log.error(f"File not found: {file.name}")
        return {}
    except Exception as e:
        log.error(f"Error processing uploaded file: {type(e).__name__}: {e}", exc_info=True)
        return {}


# ── UI status banner (theme + offline mode) ─────────────────────────────────────
def update_status_and_ui(offline: bool, dark: bool) -> str:
    """Generate responsive status banner showing theme & connection mode."""
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


def offline_toggle_handler(offline: bool, dark: bool) -> tuple:
    """
    Handle offline toggle:
    - Show/hide manual UTXO box
    - Lock/unlock addr_input + dest
    - Update addr placeholder
    - Update dest placeholder (new: offline message)
    - Clear values when going offline
    - Update status banner
    """
    manual_box_vis = gr.update(visible=offline)
    
    # Explicitly lock both when offline is ON
    addr_interactive = gr.update(interactive=not offline)
    dest_interactive = gr.update(interactive=not offline)
    
    addr_placeholder = (
        "Offline mode active — paste raw UTXOs below (txid:vout:value[:address])\n"
        "Include at least one bc1q… or bc1p… address for change output."
        if offline
        else "Paste a single modern Bitcoin address (bc1q…, bc1p…, 1…, or 3…)\n"
             "Only the first valid address is used."
    )
    
    dest_placeholder = (
        "Offline mode active — change address set via manual UTXO paste"
        if offline
        else "Paste Bitcoin address for change output (optional)"
    )
    
    # Clear values when switching to offline
    addr_value = "" if offline else gr.update()
    dest_value = "" if offline else gr.update()
    
    status_html = update_status_and_ui(offline, dark)

    return (
        manual_box_vis,         # 0
        addr_value,             # 1
        addr_interactive,       # 2
        addr_placeholder,       # 3
        dest_value,             # 4
        dest_interactive,       # 5
        dest_placeholder,       # 6: NEW - dest placeholder
        status_html             # 7
    )
# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks(
    title="Ωmega Pruner v11 — Forged Anew"
) as demo:
    
    # ── Full-screen animated background + Hero Banner ───────────────────────────────
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
                background: linear-gradient(135deg, rgba(247,147,26,0.28), rgba(0, 120, 255,0.12));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                color: transparent;
                text-shadow:
                    0 0 80px rgba(247,147,26,0.55),
                    0 0 140px rgba(0, 120, 255,0.35);
                animation: omega-breath 28s infinite ease-in-out;
                user-select: none;
                line-height: 1;
                opacity: 0.96;
            ">Ω</span>
        </span>
    </div>

    <div style="
        display: flex;
        justify-content: center;
        margin: clamp(8px, 2.5vw, 20px) auto 30px auto;
    ">
        <div style="
            text-align: center;
            padding: clamp(40px, 7vw, 70px) clamp(15px, 4vw, 30px) clamp(30px, 6vw, 50px);
            background: linear-gradient(rgba(0,0,0,0.42), rgba(0, 30, 80,0.02));
            backdrop-filter: blur(10px);
            border: clamp(4px, 2vw, 8px) solid #f7931a;
            box-shadow: 
                inset 0 0 80px rgba(0, 100, 255, 0.25),
                inset 0 0 20px rgba(0, 120, 255, 0.15),
                0 0 80px rgba(247,147,26,0.4);
            border-radius: clamp(16px, 5vw, 24px);
            max-width: 1200px;
            width: 95vw;
        ">
            <!-- Reclaim Sovereignty -->
            <div style="
                color: #ffcc00;
                font-size: clamp(2.8rem, 11vw, 5.2rem);
                font-weight: 900;
                letter-spacing: clamp(2px, 1.8vw, 12px);
                text-shadow:
                    0 0 50px #ffcc00,
                    0 0 100px #ffaa00,
                    0 0 150px rgba(255,204,0,0.9),
                    -2px -2px 0 #ffffff,
                    2px -2px 0 #ffffff,
                    -2px  2px 0 #ffffff,
                    2px  2px 0 #ffffff;
                line-height: 1;
                margin: 0 auto clamp(28px, 5.5vw, 44px) auto;
                transform: translateX(-0.03em); /* optical centering */
            ">
                Reclaim Sovereignty
            </div>

            <!-- ΩMEGA PRUNER -->
            <div style="
                color: #e65c00;
                font-size: clamp(2.4rem, 9vw, 4.8rem);
                font-weight: 900;
                letter-spacing: clamp(2px, 1.5vw, 12px);
                text-shadow:
                    0 0 25px #e65c00,
                    0 0 50px #c94a00,
                    0 0 75px rgba(230,92,0,0.9),
                    0 130px rgba(220, 0, 60, 0.85),   /* very strong red core */
                    0 0 170px rgba(200, 0, 0, 0.7),     /* intense crimson mid */
                    0 0 220px rgba(180, 0, 0, 0.5);     /* wide red outer glow */
                margin: 4px auto clamp(26px, 5.5vw, 44px) auto;
            ">
                ΩMEGA PRUNER
            </div>

            <!-- NUCLEAR COIN CONTROL -->
            <div style="
                color: #0f0;
                font-size: clamp(1.8rem, 7vw, 3.2rem);
                font-weight: 900;
                letter-spacing: clamp(3px, 1.2vw, 6px);
                text-shadow: 0 0 35px #0f0, 0 0 70px #0f0;
                margin: clamp(20px, 5vw, 35px) 0;
            ">
                NUCLEAR COIN CONTROL
            </div>

            <!-- Version -->
            <div style="
                color: #00ffaa;
                font-size: clamp(1rem, 3.5vw, 1.2rem);
                letter-spacing: clamp(1px, 0.8vw, 3px);
                text-shadow: 0 0 12px #00ffaa;
                margin: clamp(15px, 4vw, 25px) 0;
            ">
                FORGED ANEW — v11
            </div>

        <!-- Body text -->
        <div style="
            color: #ddd;
            font-size: clamp(1.1rem, 3.8vw, 1.4rem);
            line-height: 1.6;
            max-width: 900px;
            margin: clamp(30px, 6vw, 45px) auto;
            padding: 0 clamp(10px, 3vw, 20px);
        ">
            Pruning isn’t just about saving sats today — it’s a deliberate step toward taking
            <strong style="color:#0f0;">full strategic control</strong> of your Bitcoin.<br><br>

            By pruning inefficient UTXOs, you:<br>
            • <strong style="color:#00ff9d;">Slash fees</strong> during high-congestion periods<br>
            • <strong style="color:#00ff9d;">Reduce future costs</strong> with a cleaner UTXO set<br>
            • <strong style="color:#00ff9d;">Optimize your stack</strong> for speed, savings and privacy<br><br>

            <strong style="color:#f7931a; font-size: clamp(1.3rem, 4.5vw, 1.7rem); font-weight:900;">
                Prune smarter. Win forever.
            </strong>
        </div>

        <!-- Arrow -->
        <div id="hero-arrow" style="
            font-size: clamp(2.5rem, 7vw, 4rem);
            color: #f7931a;
            opacity: 0;
            margin-top: clamp(20px, 5vw, 40px);
            animation: 
                arrow-fade-in 1.8s ease-out forwards,
                arrow-pulse-bounce 5s ease-in-out infinite 2s;
            text-shadow: 0 0 30px #f7931a, 0 0 60px #f7931a;
        ">
            ↓
        </div>
    </div>
</div>

<style>
    @keyframes arrow-fade-in {
        0%   { opacity: 0; transform: translateY(-40px) scale(0.8); }
        100% { opacity: 0.92; transform: translateY(0) scale(1); }
    }

    @keyframes arrow-pulse-bounce {
        0%, 100% { 
            transform: translateY(0) scale(1); 
            opacity: 0.92; 
            text-shadow: 0 0 30px #f7931a, 0 0 60px #f7931a; 
        }
        50% { 
            transform: translateY(12px) scale(1.08); 
            opacity: 1.0; 
            text-shadow: 0 0 50px #f7931a, 0 0 100px #f7931a; 
        }
    }

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
        }

        .omega-symbol {
            will-change: opacity;
        }

        /* Fee preset buttons glow */
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

        /* Slider halo effect */
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

        /* Locked badge animation */
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

    # ── Health badge & legacy row styling ───────────────────────────────────────────
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

    # ── Global dark mode, nuclear checkboxes ────────────────
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

        /* Light mode fallback for checkboxes */
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

        /* DataFrame column wrapping for long strings (TXID, address, source) */
        .gr-dataframe th:nth-child(1) {
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow: visible !important;
            line-height: 1.2 !important;
            padding: 8px 4px !important;
            font-size: clamp(0.9rem, 3vw, 1rem) !important;
        }

        .gr-dataframe td:nth-child(2),  /* Source */
        .gr-dataframe td:nth-child(3),  /* TXID */
        .gr-dataframe td:nth-child(6) { /* Address */
            white-space: normal !important;
            word-break: break-all !important;
            overflow-wrap: break-word !important;
            hyphens: auto;
            font-family: monospace !important;
            font-size: 0.95rem !important;
            line-height: 1.35 !important;
            padding: 8px 6px !important;
        }

        .gr-dataframe td:nth-child(2),
        .gr-dataframe td:nth-child(3),
        .gr-dataframe td:nth-child(6) {
            max-width: none !important;
            overflow: visible !important;
        }

        .gr-dataframe td:nth-child(2):hover,
        .gr-dataframe td:nth-child(3):hover,
        .gr-dataframe td:nth-child(6):hover {
            background: rgba(0, 255, 136, 0.12) !important;
            cursor: pointer !important;
        }

        .gr-dataframe td {
            white-space: normal !important;
            overflow: visible !important;
        }
    </style>
    """)

    # Placeholder for dynamic prune conditions badge (populated by load or timer)
    prune_badge = gr.HTML("")
   
    # =============================
    # — BACKGROUND FEE CACHE REFRESH —
    # =============================
    def refresh_fees_periodically():
        """Background daemon thread to keep fee cache warm every 30 seconds."""
        while True:
            time.sleep(30)
            try:
                get_live_fees()  # Refresh cache without blocking main thread
            except Exception as e:
                log.warning(f"Background fee refresh error: {e}")

    # Start daemon thread only once (guard against hot-reload / multiple imports)
    if not hasattr(threading, "_fee_refresh_started"):
        fee_refresh_thread = threading.Thread(
            target=refresh_fees_periodically,
            daemon=True,
            name="FeeCacheRefresher"
        )
        fee_refresh_thread.start()
        threading._fee_refresh_started = True
        log.info("Background fee cache refresher thread started")

    # =============================
    # — LOCK-SAFE FEE PRESET FUNCTION —
    # =============================
    def apply_fee_preset_locked(locked: bool, preset: str):
        """
        Apply a fee preset (fastest, half_hour, hour, economy) to the current fee slider.
        Safe to call when UI is locked — returns no-op updates.
        """
        if locked:
            return gr.update(), gr.update()

        fees = get_live_fees() or {
            "fastestFee": 10,
            "halfHourFee": 6,
            "hourFee": 3,
            "economyFee": 1,
        }

        rate_map = {
            "fastest":   fees.get("fastestFee",   10),
            "half_hour": fees.get("halfHourFee",  6),
            "hour":      fees.get("hourFee",      3),
            "economy":   fees.get("economyFee",   1),
        }

        new_rate = rate_map.get(preset, 3)  # Default to 3 s/vB if preset unknown

        return gr.update(value=new_rate), gr.update()


    def finalize_generate_ui():
        """
        Completely lock the UI after successful PSBT generation.
        Disables inputs/sliders/toggles/buttons, hides unnecessary elements,
        shows export area and locked badge.
        Returns 22-tuple matching Gradio output order.
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
        # Modern Bitcoin Optimization Note
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
                Fully offline operation — no API calls, perfect for cold wallets.
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
                Pruning Strategy & Fee Dynamics
              </div>

              <div style="
                  color:#aaffcc !important;
                  font-size: clamp(1rem, 3.8vw, 1.2rem) !important;
                  line-height:1.7 !important;
                  text-shadow:0 2px 4px rgba(0,0,0,0.8) !important;
              ">
                Compare current vs future fees and choose your privacy–cost tradeoff
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
                manual_box_row,        # 0: manual visibility
                addr_input,            # 1: addr value
                addr_input,            # 2: addr interactive
                addr_input,            # 3: addr placeholder
                dest,                  # 4: NEW - dest value
                dest,                  # 5: NEW - dest interactive
                dest,                  # 6 placeholder
                mode_status            # 7 banner
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

<!-- Donation section – centered QRs + copy buttons (FINAL FIXED) -->
<div style="
    text-align: center !important;
    margin: clamp(40px, 10vw, 80px) auto 60px auto !important;
    padding: clamp(20px, 5vw, 40px) !important;
    background: rgba(0,0,0,0.5) !important;
    border-top: 2px solid #f7931a !important;
    max-width: 95vw !important;
    color: #ddd !important;
    font-size: clamp(0.9rem, 3vw, 1.1rem) !important;
">

    <!-- Header -->
    <div style="margin-bottom: 14px !important;">
        <strong style="
            color:#f7931a !important;
            font-size: clamp(1.1rem, 4vw, 1.3rem) !important;
        ">
            Support Ωmega Pruner
        </strong><br>
        <span style="color:#ccc !important;">
            If this tool saved you sats or helped your stack — show your love.
        </span>
    </div>

    <!-- QR row -->
    <div style="
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: clamp(20px, 5vw, 40px);
        flex-wrap: wrap;
    ">

        <!-- On-chain -->
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 180px;
        ">
            <img
                src="https://api.qrserver.com/v1/create-qr-code/?data=bitcoin:bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj&size=300x300&color=247-147-26&bgcolor=0-0-0"
                alt="Donate On-chain"
                style="
                    width: 180px;
                    height: 180px;
                    border: 2px solid #f7931a;
                    border-radius: 12px;
                    box-shadow: 0 0 20px rgba(247,147,26,0.5);
                    max-width: 45vw;
                "
            />

            <div style="
                font-weight: 700;
                color: #f7931a !important;
                margin: 8px 0 6px 0;
                font-size: clamp(0.95rem, 3.2vw, 1.1rem);
            ">
                On-chain Bitcoin
            </div>

            <div style="
                display: block;
                font-family: monospace;
                max-width: 180px;
                font-size: 0.9rem;
                color: #ffffff !important;
                opacity: 1 !important;
                filter: none !important;
                isolation: isolate;
            ">
                <span style="
                    display: inline-block;
                    max-width: 130px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    vertical-align: bottom;
                    color: #ffffff !important;
                    opacity: 1 !important;
                    text-decoration: none !important;
                    filter: none !important;
                ">
                    bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj
                </span>

                <button
                    onclick="navigator.clipboard.writeText('bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj').then(() => {
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
                        line-height:1.4 !important;
                        opacity:1 !important;
                    "
                >
                    Copy
                </button>
            </div>
        </div>

        <!-- Lightning -->
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 180px;
        ">
            <img
                src="https://api.qrserver.com/v1/create-qr-code/?data=lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c&size=300x300&color=0-255-136&bgcolor=0-0-0"
                alt="Donate Lightning (Bolt 12)"
                style="
                    width: 180px;
                    height: 180px;
                    border: 2px solid #00ff88;
                    border-radius: 12px;
                    box-shadow: 0 0 20px rgba(0,255,136,0.5);
                    max-width: 45vw;
                "
            />

            <div style="
                font-weight: 700;
                color: #00ff88 !important;
                margin: 8px 0 6px 0;
                font-size: clamp(0.95rem, 3.2vw, 1.1rem);
            ">
                Lightning (Bolt 12)
            </div>

            <div style="
                display: block;
                font-family: monospace;
                max-width: 180px;
                font-size: 0.9rem;
                color: #ffffff !important;
                opacity: 1 !important;
                filter: none !important;
                isolation: isolate;
            ">
                <span style="
                    display: inline-block;
                    max-width: 130px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    vertical-align: bottom;
                    color: #ffffff !important;
                    opacity: 1 !important;
                    text-decoration: none !important;
                    filter: none !important;
                ">
                    lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c
                </span>

                <button
                    onclick="navigator.clipboard.writeText('lno1zrxq8pjw7qjlm68mtp7e3yvxee4y5xrgjhhyf2fxhlphpckrvevh50u0qtj23mz69jm4duvpls79sak9um7pnarjzx5an0ggp9l9vpev2z8vqqsrnu7g8he7v8kphskcr2pxzgtp3saegcr7s6tx6qtzv9rk7mf46ngqqve0ewwdpupy07sswdf4lefwj4hm7r0rj3d4ckwt88e6h4zla3vlx7leegmyp03s8uph5f34atdkh7qkalp2q0qqkc9e82rrwrqfe9f3zm7yqmagnphm352u6kdwddrwalr0lefmjqqsm2trc6zazz083var6dulkm7w8c').then(() => {
                        this.innerText='COPIED';
                        this.style.color='#f7931a';
                        this.style.fontWeight='900';
                        this.style.textShadow='0 0 10px #f7931a';
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
                        line-height:1.4 !important;
                        opacity:1 !important;
                    "
                >
                    Copy
                </button>
            </div>
        </div>
    </div>

    <!-- Footer line -->
    <div style="
        margin-top: 24px;
        font-size: 0.9rem;
        color: #e6e6e6 !important;
        font-weight: 500;
    ">
        Thank you for supporting open-source Bitcoin tools.
    </div>
</div>

<!-- TAGLINE (outside support box, inside footer) -->
<div style="
    margin-top: clamp(40px, 8vw, 70px);
    text-align: center;
    isolation: isolate;
">
    <span style="
        color: #00ff00 !important;
        font-size: clamp(0.95rem, 3.8vw, 1.15rem) !important;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-shadow:
            0 0 15px #00ff00,
            0 0 30px #00ff00,
            0 0 6px #000,
            0 4px 10px #000,
            0 8px 20px #000000e6;
        opacity: 1 !important;
        filter: none !important;
    ">
        Prune smarter. Win forever. • Ω
    </span>
</div>

</div> <!-- END FOOTER -->
""",
    elem_id="omega_footer",
)
	
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False, debug=False, allowed_paths=["/"],)
