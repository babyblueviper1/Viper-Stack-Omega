# Ωmega Pruner — Fee-Aware UTXO Pruning  
## Design Overview

Ωmega Pruner is an experimental tool exploring **fee-aware and time-aware UTXO pruning**, with explicit visibility into **Common Input Ownership Heuristic (CIOH)** tradeoffs.

Rather than treating pruning as a one-time cleanup operation, the project frames it as an **economic and temporal decision** made under changing network conditions.

> **Visual reference:**  
> System flow and threat-model diagrams are included in the full README:  
> https://github.com/babyblueviper1/Viper-Stack-Omega/blob/main/docs/README.md

> **Terminology note:**  
> Ωmega Pruner uses the term *“pruning”* in the **user-side sense of intentional UTXO reduction via consolidation**, not in the consensus or node-level pruning sense.

---

## Core Question

Can pruning decisions be modeled as **economic and time-dependent choices**—rather than a static cleanup operation—**without increasing linkage risk**?

---

## Design Goals

- Make pruning decisions explicitly **fee- and time-dependent**
- Surface **privacy tradeoffs before** transaction construction
- Preserve **determinism and signability** across hardware wallets
- Avoid hidden aggregation or wallet-level inference

---

## Scope & Threat Model

Ωmega Pruner operates under a deliberately **constrained scope**:

- **Single-address analysis only**
- No cross-wallet or multi-account aggregation
- No background clustering or heuristic inference
- Deterministic input selection → predictable signing behavior
- Results are reproducible given identical inputs and fee context

This constraint is intentional. Reducing scope is treated as a **security feature**, minimizing CIOH amplification and avoiding false certainty derived from inferred or *assumed* wallet structure.

---

## Fee Context & Timing

Ωmega Pruner compares the current mempool **economy fee** against recent historical medians:

- **1-day median**
- **1-week median**
- **1-month median**

(Fee data sourced from **mempool.space** mining statistics.)

This comparison allows users to reason explicitly about **“prune now vs later” regret**, expressed directly in sats, *before* constructing a PSBT.

---

## Input Support & Constraints

### Prunable Inputs

- **Native SegWit** (`bc1q…`)
- **Taproot** (`bc1p…`)

### Non-Prunable (Display-Only)

- **Legacy** (`1…`)
- **Nested SegWit** (`3…`)

Non-prunable inputs are intentionally excluded from PSBT construction to avoid unsafe consolidation and unintended linkage.

---

## Hardware Wallet & Taproot Notes

- Taproot inputs may require explicit derivation paths on some hardware wallets
- PSBTs remain valid without a derivation path, but signing may fail depending on device firmware
- No automatic derivation path inference is performed

This condition is surfaced early in the UI and does **not** block transaction construction.

---

## Safety Properties

- No custody
- No transaction broadcasting
- No state mutation
- Offline / air-gapped operation supported via raw UTXO input
- Deterministic export (JSON + cryptographic fingerprint)
- Source-available, locally executed, and reviewable prior to use

---

## Current Limitations

- Single-address scope only
- Legacy and Nested SegWit inputs cannot be pruned
- No automatic Taproot derivation path recovery
- No batch or multi-wallet analysis

---

## Status

**Active experiment.**  
Feedback and critical review are welcome.
