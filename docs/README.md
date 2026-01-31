# Ωmega Pruner v11.1 — Forged Anew

**Reclaim Sovereignty**

A precision **UTXO consolidation analysis** tool.  
Stripped of pretense, refined to essence, honest to the core.

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** built for  
**fee-aware UTXO consolidation and long-term coin control**  
under real, observable network conditions.

> **Terminology note:**  
> “Pruning” in Ωmega Pruner refers to *wallet-side UTXO consolidation*,  
> **not** Bitcoin Core’s node-level UTXO set pruning.

> **Design note:**  
> A technical overview of the fee model, scope constraints, and CIOH tradeoffs  
> is available in **[`docs/design.md`](docs/design.md)**.

**LIVE:** https://omega-pruner.onrender.com  
**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

---

## What’s New in v11.1

- **Network Conditions Badge — LIVE**
  - Immediate fee-context snapshot before any data is entered
- Current economy fee vs dynamic mined medians:
  - 1-day
  - 1-week **(primary benchmark)**
  - 1-month
- Clear vertical comparison: **Current → VS → Medians**
- Live BTC price, block height, and network hashrate
- Next difficulty adjustment + halving countdown
- Powered by **mempool.space** mining statistics
- **Instant insight:** assess whether conditions favor consolidation *before* loading UTXOs

---

## Supported Address Types

Optimized for modern Bitcoin script types:

- **Native SegWit** (`bc1q…`) — P2WPKH  
- **Taproot** (`bc1p…`) — P2TR  

These provide superior fee efficiency, cleaner accounting, and forward compatibility.

**Legacy (`1…`) and Nested SegWit (`3…`)** inputs are shown for transparency only and  
**cannot be consolidated** (faded, disabled).

Spend or migrate them separately before consolidation.

---

## Scope & Safety Model (Read This)

Ωmega Pruner enforces a strict **single-scope safety model**:

- One address **or** one xpub per run
- **No** cross-wallet or multi-wallet mixing
- **No** hidden aggregation — ever
- Deterministic selection → predictable signing → reduced CIOH risk
- Wallet-side analysis only — no node state, no broadcast role

These constraints are deliberate and foundational to the tool’s guarantees.

---

## Hardware Wallet & Taproot Notes

Ωmega Pruner always produces valid PSBTs — even without derivation metadata.

However:

- Some hardware wallets require explicit Taproot derivation paths to sign
- If Taproot inputs are detected and no derivation path is provided:
  - A **non-blocking warning** is shown
  - PSBT construction proceeds normally
  - Signing may fail on certain devices
- No automatic re-generation with corrected paths is currently supported

**Workaround:**  
Import the PSBT into a wallet that already knows the account  
(e.g., Sparrow), or recreate the transaction there.

---

## Core Features

- Table-first interface — data loads instantly; decide before reading
- Explicit labels — no ambiguity between pre- and post-consolidation state
- CIOH recovery guidance — warnings translated into concrete next steps
- **True air-gapped / offline mode** 🔒 — paste raw UTXOs, zero API calls
- Pure dark high-contrast mode — clarity over comfort
- Deterministic selection export — JSON + cryptographic fingerprint
- Live mempool fee oracle — Economy / 1h / 30m / Fastest presets
- **Privacy Score (0–100)** — linkage, merge exposure, CIOH visibility
- Tiered CIOH warnings — color-coded and unavoidable
- “Consolidate now vs later” fee delta — quantify future regret in sats
- Per-input weight (wu) — SegWit vs Taproot vs dust clearly marked
- Live wallet footprint comparison — before / after consolidation
- **Irreversible Consolidation Warning** (formerly “NUCLEAR”)
- Full preview → PSBT fidelity guarantee
- Zero custody • Full coin control • RBF • Taproot • Dust-resistant

---

## Under the Hood — Canonical State Model

| Principle              | Implementation           | Why It Matters             |
|------------------------|--------------------------|----------------------------|
| Single source of truth | Immutable enriched state | No stale or desynced UI    |
| Derived economics      | Live computation         | Internal coherence         |
| Intent fingerprint    | Deterministic hash       | Provable user intent       |

**Audit-friendly. Deterministic. Explicit.**

---

## Diagram — Fee-Aware Consolidation Flow

```text
┌──────────────────────────┐
│        User Input        │
│  (Single Address / xpub) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     UTXO Enumeration     │
│  (No clustering, no mix) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Immutable Enriched     │
│        State             │
│  (value, script type,    │
│   weight, age, dust)     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Fee Context Layer    │
│  Current fee vs medians  │
│  (1d / 1w / 1m)          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Deterministic Selection  │
│   & Consolidation Policy │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   CIOH Risk Evaluation   │
│  Linkage & merge signals │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   PSBT Construction     │
│  (unsigned, reproducible)│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Deterministic Export   │
│      & Review            │
│ (JSON + fingerprint)    │
└──────────────────────────┘
````

### Diagram Notes

* **Single-entry point:** one address or xpub per run
* **No hidden inference:** no clustering or attribution
* **Immutable state:** all downstream logic derives from a fixed snapshot
* **Fee-aware decision layer:** time-context is explicit
* **Deterministic output:** identical inputs → identical PSBTs
* **Human-in-the-loop:** no signing or broadcasting

---

## Threat Model & Explicit Non-Goals

Ωmega Pruner is deliberately **not** a wallet, coordinator, or inference engine.

### Explicit Non-Goals

* Wallet clustering or attribution
* Cross-wallet or multi-account inference
* Heuristic enrichment beyond visible CIOH signals
* Silent optimization or auto-selection
* Transaction signing or broadcasting

These are excluded to prevent **false certainty**, **hidden linkage**, and **irreversible privacy errors**.

### Security Posture

* **Local-first:** no custody, no signing, no broadcast
* **Deterministic:** same inputs produce the same outputs
* **Explainable:** every warning and decision is visible
* **Interruptible:** abort at any stage with no side effects

### Design Rationale

Consolidation is irreversible once spent.

Ωmega Pruner therefore optimizes for **constraint, visibility, and reversible intent**,
not automation.

Reducing scope is treated as a **security feature**, not a limitation.

> *The safest consolidation decision is one whose risks are visible before the transaction exists.*

---

## Philosophy

Most consolidators compress tradeoffs into automation.

Ωmega Pruner refuses.

**No keys. No signing. No silent assumptions. No fake privacy.**

---

## Ωmega Pruner — Custom Builds

Your treasury. Your rules.

* Custom integrations
* Air-gapped / on-prem deployments
* Branded dashboards
* Dedicated support

**By quote only**
📧 [babyblueviperbusiness@gmail.com](mailto:babyblueviperbusiness@gmail.com)

🎙 **Baby Blue Viper** — [https://babyblueviper.com](https://babyblueviper.com)

---

**Ωmega Pruner v11.1 — Forged Anew**
babyblueviper & the swarm • January 2026

**Consolidate smarter. Win forever. • Ω**
