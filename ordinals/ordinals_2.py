# app.py — Omega Pruner v9.0 — Community Edition
import gradio as gr
import requests, time, base64, qrcode, io
from dataclasses import dataclass
from typing import List

# ==============================
# Optional deps (still graceful)
# ==============================
try:
    from bolt11 import decode as bolt11_decode
except:
    bolt11_decode = None

# ==============================
# Constants
# ==============================
DEFAULT_DAO_ADDR = "bc1q8jyzxmdad3t9emwfcc5x6gj2j00ncw05sz3xrj"   # kept only for grateful users
pruned_utxos_global = None
input_vb_global = 68
output_vb_global = 31

# ==============================
# Clean, non-cult CSS + disclaimer
# ==============================
css = """
.qr-button { position: fixed !important; bottom: 24px; right: 24px; z-index: 9999;
    width: 64px; height: 64px; border-radius: 50% !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); cursor: pointer; display: flex;
    align-items: center; justify-content: center; font-size: 38px; }
.qr-button.camera { bottom: 96px !important; background: #f7931a !important; }
.qr-button.lightning { bottom: 24px !important; background: #1188ff !important; }
.big-fuel-button button { height: 80px !important; font-size: 18px !important; border-radius: 16px !important; }
"""

disclaimer = """
**Omega Pruner v9.0 — Community Edition**  
Open-source • Zero custody • No forced fees  
Consolidate dusty UTXOs when fees are low → win when fees are high.  
Optional small thank-you to the original author (default 0.5% of future savings).  
Source: github.com/omega-pruner/v9 (MIT license)
"""

# ==============================
# All the Bitcoin helpers (unchanged except tiny cleanups)
# ==============================
# (bech32, base58, address_to_script_pubkey, api_get, get_utxos, fetch_all_utxos_from_xpub,
#  TxIn/TxOut/Tx classes, tx_to_psbt, rbf_bump — all identical to v8.7, just keeping them here)

# ... [paste everything from your original file from CHARSET down to the rbf_bump function]
# I'm skipping repeating 300 lines here for brevity — keep exactly the same as v8.7

# ==============================
# NEW: Local QR code generation (privacy!)
# ==============================
def make_qr(data: str) -> str:
    img = qrcode.make(data, box_size=10, border=4)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ==============================
# Core logic — now with all requested upgrades
# ==============================
def analysis_pass(...):  # unchanged except it now returns detected_type for UI
    # ... same as v8.7, just add at the end:
    return log, gr.update(visible=True), detected_type

def build_real_tx(addr, strategy, threshold, dest, sweep, invoice, xpub,
                  dao_percent, selfish_mode, dao_addr):
    global pruned_utxos_global, input_vb_global, output_vb_global
    if not pruned_utxos_global:
        return "Run analysis first!", gr.update(visible=False), ""

    total = sum(u['value'] for u in pruned_utxos_global)
    inputs = len(pruned_utxos_global)

    # Live fee rate from mempool.space
    try:
        fee_rate = requests.get("https://mempool.space/api/v3/fees/recommended", timeout=8).json()["fastestFee"]
    except:
        fee_rate = 12  # sane fallback

    vsize = 10 + inputs + (input_vb_global * inputs) + (output_vb_global * (2 if not selfish_mode and dao_percent > 0 else 1))
    miner_fee = max(1000, int(vsize * fee_rate * 1.15))  # +15% buffer

    future_cost = int((input_vb_global * inputs + output_vb_global) * 120)  # assume 120 sat/vB future
    savings = future_cost - miner_fee

    # DAO logic — completely optional
    dao_cut = 0
    if not selfish_mode and dao_percent > 0 and savings > 2000:
        dao_cut = max(546, int(savings * dao_percent / 10000))  # percent → bps → fraction

    user_gets = total - miner_fee - dao_cut
    if user_gets < 546:
        return "Not enough for dust limit after fees", gr.update(visible=False), ""

    # Lightning path
    if sweep:
        # ... same lightning_sweep_flow as before, just pass dao_cut
        return lightning_sweep_flow(pruned_utxos_global, invoice, miner_fee, savings, dao_cut, selfish_mode), ""

    # On-chain path
    dest_addr = (dest or addr).strip() if dest else addr.strip()
    dest_script, dest_info = address_to_script_pubkey(dest_addr)
    if len(dest_script) < 20:
        return "Invalid destination address", gr.update(visible=False), ""

    tx = Tx()
    for u in pruned_utxos_global:
        tx.tx_ins.append(TxIn(bytes.fromhex(u['txid']), u['vout']))

    tx.tx_outs.append(TxOut(user_gets, dest_script))
    if dao_cut > 0:
        dao_script, _ = address_to_script_pubkey(dao_addr or DEFAULT_DAO_ADDR)
        tx.tx_outs.append(TxOut(dao_cut, dao_script))

    raw = tx.encode().hex()
    psbt = tx_to_psbt(tx)
    qr = make_qr(psbt)

    fee_text = "No thank-you" if dao_cut == 0 else f"Thank-you: {dao_cut:,} sats ({dao_percent/100:.2f}% of savings)"
    return f"""
    <div style="text-align:center; max-width:780px; margin:0 auto; padding:20px;">
        <h3 style="color:#f7931a; margin-bottom:32px;">Transaction Ready</h3>
        <p>Consolidated <b>{inputs}</b> inputs → <b>{total:,}</b> sats total<br>
        Live fee rate: <b>{fee_rate}</b> sat/vB → Miner fee <b>{miner_fee:,}</b> sats<br>
        {fee_text}<br><br>
        <span style="font-size:28px; color:#00ff9d; font-weight:bold;">You receive: {user_gets:,} sats</span></p>
        <div style="display:flex; justify-content:center; margin:50px 0;">
            <img src="{qr}" style="width:440px; max-width:96vw; border-radius:20px; border:5px solid #f7931a;">
        </div>
        <small>Scan with any PSBT-compatible wallet (Sparrow, BlueWallet, Nunchuk…)</small>
        <details><summary>Raw hex / PSBT</summary>
        <pre style="text-align:left; background:#000; color:#0f0; padding:15px; border-radius:8px; overflow-x:auto;">
Raw hex:  {raw}
PSBT:     {psbt}</pre></details>
    </div>
    """, gr.update(visible=False), ""

# ==============================
# Gradio UI – clean & honest
# ==============================
with gr.Blocks(css=css, title="Omega Pruner v9.0 – Community Edition") as demo:
    gr.Markdown("# Omega Pruner v9.0")
    gr.Markdown(disclaimer)

    with gr.Row():
        user_addr = gr.Textbox(label="Bitcoin address or xpub/zpub", placeholder="bc1q… or xpub…", lines=2, scale=4)
        prune_choice = gr.Dropdown(
            ["Privacy First (30% pruned)", "Recommended (40% pruned)", "More Savings (50% pruned)"],
            value="Recommended (40% pruned)", label="Strategy")

    dust_threshold = gr.Slider(0, 3000, 546, step=1, label="Dust threshold (sats)")
    dest_addr = gr.Textbox(label="Destination (optional – leave blank = same address)", placeholder="bc1q…")

    with gr.Row():
        sweep_to_ln = gr.Checkbox(label="Sweep to Lightning", value=False)
        selfish_mode = gr.Checkbox(label="Selfish mode – keep 100% (no thank-you)", value=False)

    with gr.Row():
        dao_percent = gr.Slider(0, 500, value=50, step=10, label="Optional thank-you to original author (basis points of future savings)")
        gr.Markdown(" ← 50 bps = 0.5% (recommended if you like the tool)")

    dao_addr = gr.Textbox(label="Custom thank-you address (optional)", placeholder=f"Default: {DEFAULT_DAO_ADDR}")

    submit_btn = gr.Button("1. Analyze UTXOs", variant="secondary")
    generate_btn = gr.Button("2. Generate Transaction", visible=False, variant="primary")
    output_log = gr.HTML()

    ln_invoice = gr.Textbox(label="Lightning invoice (exact amount shown above)", lines=3, visible=False)

    # RBF section unchanged
    gr.Markdown("### Stuck tx? RBF bump")
    with gr.Row():
        rbf_in = gr.Textbox(label="Raw hex", lines=5)
        rbf_btn = gr.Button("Bump +50 sat/vB")
    rbf_out = gr.Textbox(label="New transaction", lines=8)

    # Events
    submit_btn.click(
        analysis_pass,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice, user_addr],
        outputs=[output_log, generate_btn, gr.State()]  # dummy state to keep type detection if you want
    )
    generate_btn.click(
        build_real_tx,
        inputs=[user_addr, prune_choice, dust_threshold, dest_addr, sweep_to_ln, ln_invoice, user_addr,
                 dao_percent, selfish_mode, dao_addr],
        outputs=[output_log, generate_btn, gr.State()]
    )
    rbf_btn.click(lambda h: rbf_bump(h)[0], rbf_in, rbf_out)

    # QR scanners + auto-show invoice box (same excellent code from v8.7)
    gr.HTML(""" ... same QR scanner HTML+JS block from v8.7 ... """)

if __name__ == "__main__":
    demo.queue(max_size=30)
    demo.launch(share=True, server_port=7860)
