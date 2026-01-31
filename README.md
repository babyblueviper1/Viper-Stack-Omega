# Ωmega Pruner v11.1 — Forged Anew  
**Reclaim Sovereignty**

Ωmega Pruner is a **fee-aware UTXO consolidation tool** built to surface **privacy tradeoffs**
before any transaction is constructed.

It is non-custodial and responsive to real network conditions.  
**No inputs are processed until the user explicitly chooses to analyze.**

> **Terminology note:**  
> “Pruning” here refers to *wallet-side UTXO consolidation*,  
> **not** Bitcoin Core’s node-level UTXO set pruning.

> **Design note:**  
> For a formal overview of the fee model, scope, and CIOH tradeoffs, see  
> **[`docs/design.md`](docs/design.md)**.  
>  
> A longer, text-heavy README with architecture diagrams and the threat model lives in  
> **[`docs/README.md`](docs/README.md)**.

---

## New in v11.1

- **Network Conditions Badge** — immediate fee-context snapshot  
- Current economy fee vs 1-day / 1-week / 1-month mined medians  
  (mempool.space mining data)  
- Clear vertical layout: **current fee → VS → medians**  
  (1-day → 1-week → 1-month)  
- Live BTC price, block height, and network hashrate  
- Next difficulty adjustment and halving countdown  
- **Instant insight:** assess whether consolidation conditions are favorable  
  *before pasting anything*

---

## Optimized for Modern Bitcoin

**Fully supported input types:**
- Native SegWit (`bc1q…`)
- Taproot (`bc1p…`)

**Legacy (`1…`) and Nested SegWit (`3…`) inputs** are displayed for transparency only and  
**cannot be consolidated** (faded rows, disabled checkboxes).

Spend or convert them separately before consolidation.

---

## Scope & Safety Model (Important)

- Single-address analysis only  
- **No** cross-wallet or multi-wallet mixing  
- **No** hidden aggregation — ever  
- Deterministic results → safer signing → reduced CIOH risk  

---

## Hardware Wallet & Taproot Notes

- Taproot inputs may require a derivation path for some hardware wallets  
- If no derivation path is provided, PSBTs remain valid but signing may fail on certain devices  
- A **non-blocking warning** appears when this condition is detected  
- **No automatic re-generation with corrected paths** is currently supported  
  — use a wallet that already knows the account (e.g., Sparrow) or recreate the transaction there

---

## Core Features

- Table-first interface — data loads instantly; decide before reading  
- Analysis-first flow — intent is evaluated before any commitment  
- Unambiguous labeling — no confusion between pre- and post-consolidation states  
- CIOH recovery guidance — warnings translated into concrete next steps  
- **True air-gapped / offline mode** 🔒 — paste raw UTXOs, zero API calls  
- Pure dark mode — full contrast, no haze  
- Deterministic selection export — JSON + cryptographic fingerprint  
- Live mempool fee oracle — Economy / 1h / 30m / Fastest presets  
- **Privacy Score (0–100)** — linkage, merge exposure, CIOH risk  
- Tiered CIOH warnings — color-coded and impossible to miss  
- “Consolidate now vs later” fee delta — see future regret in sats  
- Per-input weight (wu) — SegWit vs Taproot vs dust clearly marked  
- Live wallet footprint comparison — before / after cleanup  
- **NUCLEAR WALLET CLEANUP** confirmation step  
- 100% preview → PSBT fidelity  
- Zero custody • Full coin control • RBF • Taproot • Dust-resistant  

---

## Limitations

- Only Native SegWit and Taproot inputs can be consolidated  
- Legacy and Nested SegWit inputs cannot be included in PSBTs  
- No automatic derivation path inference for Taproot hardware signing  
- Single-address scope only — no batch or multi-wallet support  

---

**Consolidate smarter. Win forever. • Ω**

**Custom builds** → babyblueviperbusiness@gmail.com
