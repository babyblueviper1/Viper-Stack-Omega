# Ωmega Pruner — Fee-Aware UTXO Consolidation  
## Design Overview

Ωmega Pruner is an experimental tool for **fee-aware and time-aware UTXO consolidation**, with explicit, user-visible exposure to **Common Input Ownership Heuristic (CIOH)** tradeoffs.

Rather than treating consolidation as a one-time wallet cleanup, the project frames it as an **economic decision made under changing network conditions**, where timing, fee environment, and linkage risk matter as much as transaction construction.

> **Visual reference:**  
> System flow and threat-model diagrams are included in the full README:  
> https://github.com/babyblueviper1/Viper-Stack-Omega/blob/main/docs/README.md

> **Terminology note:**  
> Ωmega Pruner uses *“pruning”* internally to describe **intentional UTXO reduction via consolidation**.  
> This is distinct from Bitcoin Core’s node-level UTXO set pruning and carries no consensus meaning.

---

## Core Question

Can UTXO consolidation decisions be modeled as **economic and time-dependent choices**—rather than a static cleanup operation—**without increasing address linkage or CIOH exposure**?

---

## Design Goals

- Treat consolidation as **fee- and time-dependent**, not opportunistic
- Surface **privacy and CIOH tradeoffs before** transaction construction
- Preserve **determinism and signability** across software and hardware wallets
- Avoid hidden aggregation, inference, or wallet-level heuristics
- Ensure all risk signals are **visible, explicit, and user-controlled**

---

## Scope & Threat Model

Ωmega Pruner operates under a deliberately **constrained and explicit scope**:

- **Single address per run**
- No cross-wallet or multi-account aggregation
- No clustering, labeling, or heuristic inference
- Deterministic input selection → predictable signing behavior
- Outputs are reproducible given identical inputs and fee context

This constraint is intentional.  
Reducing scope is treated as a **security property**, not a limitation.

By refusing to infer wallet structure or user intent, the tool minimizes CIOH amplification and avoids false certainty derived from assumed relationships between UTXOs.

---

## Fee Context & Timing Model

Ωmega Pruner evaluates consolidation decisions against **observed network conditions**, not static fee presets.

The current **economy fee** is compared against mined historical medians:

- **1-day median**
- **1-week median** *(primary benchmark)*
- **1-month median*

(Fee data sourced from **mempool.space** mining statistics.)

This comparison enables users to reason explicitly about **“consolidate now vs later” regret**, expressed directly in sats, *before* any PSBT is constructed.

The fee context layer is informational and advisory; it does not auto-select or override user intent.

---

## Input Support & Constraints

### Consolidatable Inputs

- **Native SegWit** (`bc1q…`)
- **Taproot** (`bc1p…`)

These inputs provide predictable weights, modern script semantics, and safer consolidation properties.

### Display-Only Inputs

- **Legacy** (`1…`)
- **Nested SegWit** (`3…`)

Display-only inputs are intentionally excluded from PSBT construction to prevent unsafe consolidation, unexpected fee inflation, and unintended linkage.

---

## Hardware Wallet & Taproot Considerations

- Some hardware wallets require explicit Taproot derivation paths to sign
- PSBTs generated without derivation metadata remain valid
- Signing may fail on certain devices depending on firmware behavior
- No automatic derivation path inference or regeneration is performed

This condition is surfaced clearly in the UI and does **not** block transaction construction.

---

## Safety Properties

Ωmega Pruner is designed to be interruptible, inspectable, and non-authoritative:

- No custody of keys or funds
- No transaction signing or broadcasting
- No background state mutation
- Offline / air-gapped operation supported via raw UTXO input
- Deterministic export (JSON + cryptographic fingerprint)
- All computation is local, reviewable, and user-initiated

---

## Explicit Non-Goals

Ωmega Pruner deliberately does **not** attempt to provide:

- Wallet clustering or address attribution
- Cross-wallet or cross-account inference
- Heuristic enrichment beyond visible CIOH signals
- Automatic consolidation or silent optimization
- Fee prediction beyond observed historical context

These exclusions prevent silent linkage, hidden assumptions, and irreversible privacy errors.

---

## Current Limitations

- Single-address only
- Legacy and Nested SegWit inputs cannot be consolidated
- No automatic Taproot derivation path recovery
- No batch, multi-wallet, or coordinator functionality

---

## Status

**Active experiment.**

The project prioritizes correctness, determinism, and explicit tradeoffs over convenience.

Critical review and adversarial feedback are encouraged.
