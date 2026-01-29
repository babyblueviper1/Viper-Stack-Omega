# Ωmega Pruner v11.1 — Forged Anew

**Reclaim Sovereignty**

A precision UTXO consolidation analysis tool.  
Stripped of pretense, refined to essence, honest to the core.

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** designed for  
**fee-aware UTXO consolidation analysis and long-term coin control**  
under real network conditions.

> **Terminology note:**  
> “Pruning” in Ωmega Pruner refers to *wallet-side UTXO consolidation*,  
> **not** Bitcoin Core’s node-level UTXO set pruning.

> **Design note:**  
> A technical overview of the fee model, scope, and CIOH tradeoffs is available in  
> **[`docs/design.md`](docs/design.md)**

**LIVE:** https://omega-pruner.onrender.com  
**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

---

## What’s New in v11.1

- **Pruning Conditions Badge — LIVE**  
  Fee-context snapshot of current network conditions
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
- Wallet-side analysis only — no node state, mempool authority, or broadcast role

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
- “Consolidate now vs later” fee delta — see future regret in sats
- Per-input weight (wu) — SegWit vs Taproot vs dust clearly marked
- Live wallet footprint comparison — before / after cleanup
- **NUCLEAR WALLET CLEANUP** confirmation step
- 100% preview → PSBT fidelity
- Zero custody • Full coin control • RBF • Taproot • Dust-resistant

---

## Under the Hood — Canonical State Model

| Principle              | Implementation           | Why It Matters             |
|------------------------|--------------------------|----------------------------|
| Single source of truth | Immutable enriched state | No stale or desynced UI    |
| Derived economics      | Live computation         | Perfect internal coherence |
| Selection fingerprint  | Deterministic hash       | Provable user intent       |

**Audit-friendly. Deterministic. Explicit.**

---

## Diagram — Fee-Aware Pruning Flow

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
│  (values, script type,   │
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
│   Deterministic Pruning  │
│        Strategy          │
│  (user-selected policy) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   CIOH Risk Evaluation   │
│  Linkage & merge checks  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   PSBT Construction      │
│  (unsigned, reproducible)│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Deterministic Export &  │
│        Review            │
│  (JSON + fingerprint)   │
└──────────────────────────┘
````

### Diagram Notes

* **Single-entry point:** Only one address or xpub is processed per run
* **No hidden inference:** No clustering, labeling, or wallet-level heuristics
* **Immutable state:** All downstream computation derives from a fixed snapshot
* **Fee-aware decision layer:** Pruning decisions are evaluated against time-based fee context
* **Deterministic output:** Identical inputs and fee context produce identical PSBTs
* **Human-in-the-loop:** No automatic broadcasting or signing

### Why This Matters

Most pruning tools conflate selection, economics, and privacy into a single opaque step.

Ωmega Pruner separates these layers explicitly, allowing users to reason about:

* **When** to prune (fee timing)
* **What** to prune (UTXO selection)
* **How much** risk is introduced (CIOH visibility)

Before any transaction is signed.

This layered approach mirrors protocol design: constrain scope, surface tradeoffs, and preserve determinism.

---

## Diagram — Threat Model & Explicit Non-Goals

```text
┌────────────────────────────────────────────┐
│               NOT IN SCOPE                 │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ Wallet        │   │ Address          │  │
│  │ Clustering    │   │ Attribution      │  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ Cross-Wallet  │   │ Multi-Account   │  │
│  │ Aggregation   │   │ Inference       │  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ Heuristic     │   │ Silent           │  │
│  │ Enrichment    │   │ Auto-Selection   │  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
└────────────────────────────────────────────┘
                ▲
                │  Explicit boundary
                │
┌────────────────────────────────────────────┐
│                 IN SCOPE                   │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ Single        │   │ Deterministic    │  │
│  │ Address/xpub  │   │ UTXO Selection  │  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ Fee & Time    │   │ CIOH Visibility  │  │
│  │ Context       │   │ (No suppression)│  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
│  ┌───────────────┐   ┌─────────────────┐  │
│  │ PSBT          │   │ Human-in-the-    │  │
│  │ Construction  │   │ Loop Review     │  │
│  └───────────────┘   └─────────────────┘  │
│                                            │
└────────────────────────────────────────────┘
```

### Threat Model Notes

Ωmega Pruner is deliberately **not** a wallet, coordinator, or inference engine.

#### Explicit Non-Goals

* Wallet clustering or address attribution
* Cross-wallet or multi-account inference
* Heuristic enrichment beyond visible CIOH signals
* Automatic selection or silent optimization
* Transaction signing or broadcasting

These are excluded to avoid **false certainty**, **hidden linkage**, and **irreversible privacy mistakes**.

### Security Posture

* **Local-first:** No custody, no signing, no broadcast
* **Deterministic:** Identical inputs yield identical outputs
* **Explainable:** Every selection and warning is visible to the user
* **Interruptible:** Users may abort at any stage without side effects

### Design Rationale

Pruning is irreversible once spent.

Ωmega Pruner therefore optimizes for **constraint, visibility, and reversibility of intent**, not automation.

Reducing scope is treated as a **security feature**, not a limitation.

> *The safest pruning decision is one whose risks are visible before the transaction exists.*

---

## Philosophy

Most consolidators hide complexity or compress tradeoffs into automation.
Ωmega Pruner does neither.

**No keys. No signing. No silent failures. No fake privacy.**

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

**Prune smarter. Win forever. • Ω**
