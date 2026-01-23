# Ωmega Pruner v11.1 — Forged Anew

**Reclaim Sovereignty**

The purest UTXO consolidator ever built.  
Reborn in fire — stripped of pretense, refined to essence, honest to the core.

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** designed for
**fee-aware UTXO consolidation and long-term coin control** under real network
conditions.

> **Design note:**  
> A technical overview of the fee model, scope, and CIOH tradeoffs is available in  
> **[`docs/design.md`](docs/design.md)**

**LIVE:** https://omega-pruner.onrender.com  
**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

---

## What’s New in v11.1

- **Pruning Conditions Badge — LIVE**  
  Real-time 1–10 score reflecting current pruning conditions
- Current economy fee vs dynamic medians:
  - 1-day  
  - 1-week **(primary benchmark)**  
  - 1-month  
- Clear vertical comparison: **Current → VS → Medians**
- Live BTC price, block height, and network hashrate
- Next difficulty adjustment and halving countdown
- Powered by **mempool.space** mining statistics
- **Instant insight:** assess whether conditions favor pruning *before* pasting any data

---

## Supported Address Types

Optimized for modern Bitcoin script types:

- **Native SegWit** (`bc1q…`) — P2WPKH  
- **Taproot** (`bc1p…`) — P2TR  

These offer the best fee efficiency, privacy characteristics, and forward compatibility.

**Legacy (`1…`) and Nested SegWit (`3…`)** inputs are shown for transparency only and  
**cannot be pruned** (faded, disabled).  
Spend or migrate them separately before consolidation.

---

## Scope & Safety Model (Read This)

Ωmega Pruner enforces a strict **single-scope safety model**:

- One address **or** one xpub per run
- **No** cross-wallet or multi-wallet mixing
- **No** hidden aggregation — ever
- Deterministic selection → predictable signing → minimized CIOH risk

This constraint is intentional and central to the tool’s safety guarantees.

---

## Hardware Wallet & Taproot Notes

Ωmega Pruner always generates valid PSBTs — even without derivation metadata.

However:

- Some hardware wallets require explicit Taproot derivation paths to sign
- If Taproot inputs are detected and hardware signing is enabled without a path:
  - A **non-blocking warning** is displayed
  - PSBT generation proceeds normally
  - Signing may be refused by certain devices
- No automatic re-generation with corrected paths is currently supported

**Workaround:** Import the PSBT into a wallet that already knows the account
(e.g., Sparrow), or recreate the transaction there.

This preserves flexibility while remaining explicit about hardware limitations.

---

## Core Features

- Table-first interface — data loads instantly; act before reading
- Unambiguous labeling — no confusion between pre- and post-prune states
- CIOH recovery guidance — warnings translated into concrete next steps
- **True air-gapped / offline mode** 🔒 — paste raw UTXOs, zero API calls
- Pure dark nuclear mode — maximum contrast, zero haze
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

---

## Under the Hood — Canonical State Model

| Principle              | Implementation           | Why It Matters            |
|------------------------|--------------------------|---------------------------|
| Single source of truth | Immutable enriched state | No stale or desynced UI   |
| Derived economics      | Live computation         | Perfect internal coherence|
| Selection fingerprint  | Deterministic hash       | Provable user intent      |

**Audit-friendly. Deterministic. Explicit.**

---

## Philosophy

Most consolidators hide complexity or paper over tradeoffs.  
Ωmega Pruner does neither.

**No keys. No signing. No silent failures. No fake privacy.**

---

## Ωmega Pruner — Custom Builds

Your treasury. Your rules.

- Custom integrations
- Air-gapped / on-prem deployments
- Branded dashboards
- Dedicated support

**By quote only**  
📧 babyblueviperbusiness@gmail.com

🎙 **Baby Blue Viper** — https://babyblueviper.com

---

**Ωmega Pruner v11.1 — Forged Anew**  
babyblueviper & the swarm • January 2026  

**Prune smarter. Win forever. • Ω**
