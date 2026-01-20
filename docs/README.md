# Ωmega Pruner v11.1 — Forged Anew

**Reclaim Sovereignty**

The purest UTXO consolidator ever built.  
Reborn in fire — stripped of pretense, refined to essence, honest to the core.

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** for advanced UTXO consolidation and long-term coin control.

**LIVE:** https://omega-pruner.onrender.com

**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

### What’s New in v11.1

- **Pruning Conditions Badge — LIVE**  
  Real-time 1–10 score with glowing nuclear design  
- Current economy fee vs dynamic medians:  
  • 1-day  
  • 1-week **(primary benchmark)**  
  • 1-month  
- Clear vertical layout: **Current → VS → Medians** (1-day → 1-week → 1-month)  
- Live BTC price + block height + hashrate  
- Next difficulty adjustment + halving countdown  
- Powered by **mempool.space** mining statistics  
- **Instant insight**: know if now is prime pruning time — before pasting anything

### Supported Address Types

Optimized for modern Bitcoin:

- **Native SegWit** (`bc1q…`) — P2WPKH  
- **Taproot** (`bc1p…`) — P2TR  

These deliver the best fee efficiency, privacy characteristics, and forward compatibility.

**Legacy (`1…`) and Nested SegWit (`3…`)** inputs are displayed for transparency only and **cannot be pruned** (faded, disabled checkboxes).  
Spend or migrate them separately before consolidation.

### Scope & Safety Model (Read This)

Strict **single-scope safety model**:

- One address **OR** one xpub per run  
- **No** cross-wallet or multi-wallet mixing  
- **No** hidden aggregation — ever  
- Deterministic results → safer signing → minimized CIOH risk

### Hardware Wallet & Taproot Notes

Ωmega Pruner always generates valid PSBTs — even without derivation metadata.

However:

- Some hardware wallets require Taproot derivation paths to authorize signing  
- If Taproot inputs are detected and hardware support is enabled without a path:  
  - A **non-blocking warning** is displayed  
  - PSBT generation still succeeds  
  - Signing may be refused by certain devices  
- **No re-generation with corrected path** is currently supported  
- Workaround: Import into a wallet that already knows the account (e.g., Sparrow) or recreate the transaction there

This preserves maximum flexibility while being honest about hardware limitations.

### Core Features

- Table-first interface — data loads instantly, act before reading  
- Unambiguous labeling — no confusion between pre- and post-prune states  
- PayJoin-aware analysis — invoice detection with CIOH-safe handling  
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

### Under the Hood — Canonical State Model

| Principle              | Implementation              | Why It Matters               |
|------------------------|-----------------------------|------------------------------|
| Single source of truth | Immutable enriched state    | No stale UI                  |
| Derived economics      | Live computation            | Perfect consistency          |
| Selection fingerprint  | Deterministic hash          | Provable intent              |

**Audit-proof. Deterministic. Unbreakable.**

### Philosophy

Most consolidators lie to you with half-implemented features.  
Ωmega Pruner tells the truth — and nothing but the truth.

**No keys. No signing. No silent failures. No fake privacy.**

### Ωmega Pruner — Custom Builds

Your treasury. Your rules.

- Custom integrations  
- Air-gapped / on-prem deployments  
- Branded dashboards  
- Dedicated support  

**By quote only**  
📧 babyblueviperbusiness@gmail.com

🎙 **Baby Blue Viper** — https://babyblueviper.com

**Ωmega Pruner v11.1 — Forged Anew**  
babyblueviper & the swarm • January 2026

**Prune smarter. Win forever.**

**Ω**
