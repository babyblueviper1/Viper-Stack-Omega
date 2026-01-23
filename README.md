
# Ωmega Pruner v11.1 — Forged Anew  
**Reclaim Sovereignty**

Ωmega Pruner is a fee-aware UTXO pruning tool designed to help users make
**economically informed and privacy-conscious pruning decisions** under
changing network conditions.

> **Design note:**  
> For a technical overview of the fee model, threat assumptions, and CIOH
> tradeoffs, see  
> **[`docs/design.md`](docs/design.md)**

---
### New in v11.1

- **Pruning Conditions Badge** — real-time 1–10 score with glowing nuclear design  
- Current economy fee vs 1-day / 1-week / 1-month medians (mempool.space mining data)  
- Clear vertical layout: **current fee → VS → medians** (1-day → 1-week → 1-month)  
- Live BTC price + block height + hashrate  
- Next difficulty adjustment + halving countdown  
- **Instant insight**: know if now is prime pruning time — before pasting anything

### Optimized for Modern Bitcoin

**Fully supported input types:**
- Native SegWit (`bc1q…`)
- Taproot (`bc1p…`)

**Legacy (`1…`) and Nested SegWit (`3…`) inputs** are displayed for transparency only and **cannot be pruned** (faded, disabled checkboxes).  
Spend or convert them separately before consolidation.

### Scope & Safety Model (Important)

- Single-address analysis only  
- **No** cross-wallet or multi-wallet mixing  
- **No** hidden aggregation — ever  
- Deterministic results → safer signing → minimized CIOH risk

### Hardware Wallet & Taproot Notes

- Taproot inputs may require a derivation path for some hardware wallets  
- If no derivation path is provided, PSBTs are still valid but signing may fail on certain devices  
- A **non-blocking warning** appears when this condition is detected  
- **No re-generation with corrected path** is currently supported — use a wallet that already knows the account (e.g., Sparrow) or recreate the tx there

### Core Features

- Table-first interface — data loads instantly, act before reading  
- Unambiguous labeling — no confusion between pre- and post-prune states  
- CIOH recovery guidance — warnings translated into concrete next steps  
- **True air-gapped / offline mode** 🔒 — paste raw UTXOs, zero API calls  
- Pure dark nuclear mode — full contrast, no haze  
- Deterministic selection export — JSON + cryptographic fingerprint  
- Live mempool fee oracle — Economy / 1h / 30m / Fastest presets  
- **Privacy Score (0–100)** — linkage, merge exposure, CIOH risk  
- Tiered CIOH warnings — color-coded and impossible to miss  
- “Prune now vs later” fee delta — see future regret in sats  
- Per-input weight (wu) — SegWit vs Taproot vs dust clearly marked  
- Live wallet footprint comparison — before / after cleanup  
- **NUCLEAR WALLET CLEANUP** confirmation step  
- 100% preview → PSBT fidelity  
- Zero custody • Full coin control • RBF • Taproot • Dust-resistant  

**Custom builds** → babyblueviperbusiness@gmail.com

**Limitations**
- Only Native SegWit & Taproot inputs can be pruned  
- Legacy/Nested inputs cannot be included in PSBTs  
- No automatic derivation path inference for Taproot hardware signing  
- Single-address scope only — no batch/multi-wallet support

Technical design notes and threat model are documented in `docs/design.md`.

**Prune smarter. Win forever. • Ω**
