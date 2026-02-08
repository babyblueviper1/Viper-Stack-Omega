# Ωmega Pruner v11.1

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** built for  
**fee-aware UTXO consolidation and long-term coin control**  
under real, observable network conditions.

It is no longer just a standalone tool — it has evolved into an **infrastructure layer**  
that empowers Bitcoin wallets, services, and platforms to optimize UTXO structure intelligently,  
without ever compromising custody, control, or privacy.

- **Non-custodial by design** — no keys, no signing, no broadcast role  
- **Analysis-first** — intent is evaluated before any transaction exists  
- **Deterministic & reproducible** — identical inputs yield identical outputs  
- **Privacy tradeoffs surfaced upfront** — CIOH linkage, merge risk, fee regret

> **Terminology note:**  
> “Pruning” here refers to *wallet-side UTXO consolidation*,  
> **not** Bitcoin Core’s node-level UTXO set pruning.

> **Design note:**  
> A technical overview of the fee model, scope constraints, and CIOH tradeoffs  
> is available in **[`design.md`](design.md)**.

**LIVE:** https://omega-pruner.onrender.com  
**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

## What’s New in v11.1

- **Network Conditions Badge — LIVE**  
  Immediate fee-context snapshot before any data is entered
- Current economy fee vs dynamic mined medians:
  - 1-day
  - 1-week **(primary benchmark)**
  - 1-month
- Clear vertical comparison: **Current → VS → Medians** (1-day → 1-week → 1-month)
- Live BTC price, block height, and network hashrate
- Next difficulty adjustment + halving countdown
- Powered by **mempool.space** mining statistics
- **Instant insight:** assess whether conditions favor consolidation *before* loading UTXOs

## Supported Address Types

Optimized for modern Bitcoin script types:

- **Native SegWit** (`bc1q…`) — P2WPKH  
- **Taproot** (`bc1p…`) — P2TR  

These provide superior fee efficiency, cleaner accounting, and forward compatibility.

**Legacy (`1…`) and Nested SegWit (`3…`)** inputs are displayed for transparency only and  
**cannot be consolidated** (faded rows, disabled checkboxes).

Spend or migrate them separately before consolidation.

## UTXO Consolidation as an Infrastructure Layer

Ωmega Pruner is designed not just as a tool, but as an **infrastructure layer** that empowers Bitcoin wallets and services to scale more effectively.

As the backend layer for fee-aware consolidation, it enables immediate, on-demand UTXO analysis and optimization of your wallet's structure, making it the next-gen tool for privacy-conscious and fee-sensitive Bitcoin users and services.

Ωmega Pruner integrates seamlessly into existing Bitcoin wallets and platforms, enhancing user experience without sacrificing security or control.  
It’s built to be part of an overarching Bitcoin infrastructure, enabling smarter transactions without compromising sovereignty.

## Scope & Safety Model (Read This)

Ωmega Pruner enforces a strict **single-scope safety model**:

- One address per run
- **No** cross-wallet or multi-wallet mixing
- **No** hidden aggregation — ever
- Deterministic selection → predictable signing → reduced CIOH risk
- Wallet-side analysis only — no node state, no signing, no broadcast role

These constraints are deliberate and foundational to the tool’s guarantees.

## On Offline vs Online Operation

True offline workflows are harder than they appear — and partial implementations often introduce  
more ambiguity than safety.

For now, Ωmega Pruner does not attempt to simulate or approximate “offline mode” inside a browser  
environment. We prefer no half-measures.

If a genuinely sound, inspectable, and user-verifiable offline architecture can be achieved in the  
future, it may be incorporated. Until then, the project remains explicit about what it does and does  
not guarantee.

More broadly, the tradeoffs between offline and online operation — when done correctly — are often  
misunderstood. Offline is not automatically safer, just as online is not inherently surveillance.  
Both can fail. Both can be done well.

Ωmega Pruner is designed around clarity of intent, observable behavior, and minimized trust — not  
ideology.

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
Import the PSBT into a wallet that already knows the account or recreate the transaction there.

## Core Features

- Table-first interface — data loads instantly; decide before reading  
- Analysis-first flow — intent is evaluated before any commitment  
- Unambiguous labeling — no confusion between pre- and post-consolidation states  
- CIOH recovery guidance — warnings translated into concrete next steps  
- Explicit online execution model — no simulated or partial “offline mode”  
- Pure dark mode — full contrast, no haze  
- Deterministic selection export — JSON + cryptographic fingerprint  
- Live mempool fee oracle — Economy / 1h / 30m / Fastest presets  
- **Privacy Score (0–100)** — linkage, merge exposure, CIOH risk  
- Tiered CIOH warnings — color-coded and impossible to miss  
- “Consolidate now vs later” fee delta — see future regret in sats  
- Per-input weight (wu) — SegWit vs Taproot vs dust clearly marked  
- **Per-input age display** — shows confirmation age ("<1 day", "12 days", "~3 months", "~4 years") with color-coding:  
  red = very recent (high linkage risk), orange = months-old, green = years-old (ideal for cleanup)  
  **Age is secondary context only** — primary recommendations/pre-checks are still driven by value + weight/script type; age helps manual prioritization for lowest CIOH risk
- Live wallet footprint comparison — before / after cleanup  
- **One-Time Structural Consolidation Warning**  
- 100% preview → PSBT fidelity  
- Zero custody • Full coin control • RBF • Taproot • Dust-resistant

## Under the Hood — Canonical State Model

| Principle              | Implementation           | Why It Matters          |
|------------------------|--------------------------|-------------------------|
| Single source of truth | Immutable enriched state | No stale or desynced UI |
| Derived economics      | Live computation         | Internal coherence      |
| Intent fingerprint    | Deterministic hash       | Provable user intent    |

**Audit-friendly. Deterministic. Explicit.**

## Diagram — Fee-Aware Consolidation Flow

```text
┌──────────────────────────┐
│        User Input        │
│     (Single Address)     │
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
│   PSBT Construction      │
│ (unsigned, reproducible) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Deterministic Export   │
│        & Review          │
│  (JSON + fingerprint)   │
└──────────────────────────┘
```

### Diagram Notes

- **Single-entry point:** one address per run
- **No hidden inference:** no clustering or attribution
- **Immutable state:** all downstream logic derives from a fixed snapshot
- **Fee-aware decision layer:** time context is explicit
- **Deterministic output:** identical inputs → identical PSBTs
- **Human-in-the-loop:** no signing or broadcasting

## Threat Model & Explicit Non-Goals

Ωmega Pruner is deliberately **not** a wallet, coordinator, or inference engine.

### Explicit Non-Goals

- Wallet clustering or attribution
- Cross-wallet or multi-account inference
- Heuristic enrichment beyond visible CIOH signals
- Silent optimization or auto-selection
- Transaction signing or broadcasting

These are excluded to prevent **false certainty**, **hidden linkage**, and **irreversible privacy errors**.

### Security Posture

- **Local-first:** no custody, no signing, no broadcast
- **Deterministic:** identical inputs yield identical outputs
- **Explainable:** every warning and decision is visible
- **Interruptible:** abort at any stage with no side effects

### Design Rationale

Consolidation is irreversible once spent.

Ωmega Pruner therefore optimizes for **constraint, visibility, and reversible intent**,  
not automation.

Reducing scope is treated as a **security feature**, not a limitation.

> *The safest consolidation decision is one whose risks are visible before the transaction exists.*

## Philosophy

Most consolidators compress tradeoffs into automation.

Ωmega Pruner refuses.

**No keys. No signing. No silent assumptions. No fake privacy.**

## Ωmega Pruner — Custom Builds

Your treasury. Your rules.

- Custom integrations
- Air-gapped / on-prem deployments
- Branded dashboards
- Dedicated support

**By quote only**  
📧 [babyblueviperbusiness@gmail.com](mailto:babyblueviperbusiness@gmail.com)

🎙 **Baby Blue Viper** — [https://babyblueviper.com](https://babyblueviper.com)

---

**Ωmega Pruner v11.1**  
babyblueviper & the swarm • January 2026

**Consolidate smarter. Win forever. • Ω**
