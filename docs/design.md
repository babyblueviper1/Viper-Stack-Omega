# Ωmega Pruner — Fee-Aware UTXO Consolidation

## Design Overview

Ωmega Pruner has evolved from an experimental standalone tool into an **infrastructure layer** for Bitcoin wallets and services.

It provides **fee-aware, time-aware UTXO consolidation analysis** with deliberate, user-visible exposure to **Common Input Ownership Heuristic (CIOH)** tradeoffs — enabling smarter structural decisions before any transaction is constructed.

Rather than treating consolidation as a one-time wallet cleanup, the project frames it as an **economic decision made under observable network conditions**, where timing, fee environment, and linkage risk are as consequential as transaction construction itself.

As an infrastructure layer, Ωmega Pruner integrates seamlessly into existing wallets and platforms, enhancing their UTXO management capabilities without compromising custody, control, or privacy.

> **Visual reference:**  
> System flow and threat-model diagrams are included in the full README:  
> [https://github.com/babyblueviper1/Viper-Stack-Omega/blob/main/docs/README.md](https://github.com/babyblueviper1/Viper-Stack-Omega/blob/main/docs/README.md)

> **Terminology note:**  
> Ωmega Pruner uses *“pruning”* internally to describe **intentional UTXO reduction via consolidation**.  
> This is distinct from Bitcoin Core’s node-level UTXO set pruning and carries no consensus meaning.

---

## Core Question

Can UTXO consolidation be modeled as a **fee- and time-dependent economic choice** — rather than a static cleanup operation — **without increasing address linkage or CIOH exposure**, and can this model be provided as a reusable **infrastructure layer** for wallets and services?

---

## Design Goals

- Treat consolidation as **time- and fee-contextual**, not opportunistic
- Surface **privacy and CIOH tradeoffs before** any transaction is constructed
- Preserve **determinism and signability** across software and hardware wallets
- Avoid hidden aggregation, inference, or wallet-level assumptions
- Ensure all risk signals are **user-visible and decision-relevant**
- Provide a **reusable, non-custodial infrastructure layer** that wallets and services can integrate to offer intelligent UTXO optimization

---

## Scope & Threat Model

Ωmega Pruner operates under a deliberately **constrained scope**:

- **Single address per run**
- No cross-wallet or multi-account aggregation
- No clustering, labeling, or heuristic inference
- Deterministic input selection → predictable signing behavior
- Outputs are reproducible given identical inputs and fee context

This constraint is intentional.

Reducing scope is treated as a **security property**, not a limitation.  
By refusing to infer wallet structure or user intent, the tool minimizes CIOH amplification and avoids false certainty derived from assumed UTXO relationships.

As an infrastructure layer, these constraints ensure safe, predictable integration into higher-level systems without introducing hidden trust or privacy leakage.

---

## On Offline vs Online Operation

True offline workflows are harder than they appear — and partial implementations often introduce more ambiguity than safety.

Ωmega Pruner does not attempt to simulate or approximate an “offline mode” inside a browser environment. No half-measures are taken.

While PSBTs produced by the tool can be signed in fully offline or air-gapped environments, the analysis phase itself assumes an online context with observable network data.

If a genuinely sound, inspectable, and user-verifiable offline architecture can be achieved in the future, it may be incorporated. Until then, the project remains explicit about what it does and does not guarantee.

More broadly, offline is not automatically safer, just as online is not inherently surveillance. Both can fail. Both can be done well.

Ωmega Pruner is designed around clarity of intent, observable behavior, and minimized trust — not ideology.

---

## Fee Context & Timing Model

Ωmega Pruner evaluates consolidation decisions against **observed network conditions**, not fixed fee presets.

The current **economy fee** is compared against mined historical medians:

- **1-day median**
- **1-week median** *(primary benchmark)*
- **1-month median**

(Fee data sourced from **mempool.space** mining statistics.)

This comparison allows users to reason explicitly about **“consolidate now vs later” regret**, expressed directly in sats, *before* any PSBT is constructed.

The fee context layer is **advisory only**.  
It informs decision-making but does not auto-select inputs or override user intent.

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

Ωmega Pruner is designed to be **inspectable, interruptible, and non-authoritative**:

- No custody of keys or funds
- No transaction signing or broadcasting
- No hidden background state or side effects
- Deterministic export (JSON + cryptographic fingerprint)
- All computation is explicit, reviewable, and user-initiated
- PSBTs may be signed in online or fully offline / air-gapped environments

Safety is derived from **scope reduction, determinism, and visibility** — not from attempting to approximate offline guarantees in unsuitable environments.

As an infrastructure layer, these properties ensure safe, predictable integration into higher-level systems without introducing hidden trust or privacy leakage.

---

## Assumptions & Failure Modes

Ωmega Pruner is intentionally narrow. Its guarantees hold **only** if the following assumptions are understood and respected.

### Core Assumptions

- **User intent is deliberate**  
  The user understands that UTXO consolidation is irreversible once spent and reviews all warnings before proceeding.

- **Input data is accurate**  
  UTXOs provided by the user correctly represent the address being analyzed.  
  The tool does not verify ownership, provenance, or wallet context.

- **Observed fees are representative, not predictive**  
  Fee medians reflect recent mined conditions, not future certainty.  
  The tool assumes past conditions are a useful *context*, not a guarantee.

- **Single-address scope is sufficient**  
  The user does not expect cross-address, cross-wallet, or account-level inference.

### Known Failure Modes

- **Fee regime shifts**  
  Sudden mempool shocks (spam waves, market events, miner behavior changes) may invalidate short-term fee comparisons after analysis.

- **Hardware wallet signing refusal**  
  Taproot PSBTs without explicit derivation paths may fail to sign on some hardware devices, despite being structurally valid.

- **False sense of optimality**  
  A favorable fee comparison does **not** imply a consolidation is privacy-optimal, future-proof, or globally optimal — only that it is economically favorable *relative to recent history*.

- **Unmodeled linkage outside scope**  
  CIOH exposure is evaluated **within the provided input set only**.  
  Linkage arising from prior transactions, external surveillance, or wallet behavior is out of scope.

- **User misinterpretation**  
  The tool surfaces tradeoffs; it does not make decisions.  
  Acting without understanding the warnings negates the tool’s safety model.

### Non-Failures (By Design)

The following are **not** considered failures, even if they surprise users:

- The tool refusing to consolidate Legacy or Nested SegWit inputs
- The absence of automatic selection or optimization
- The inability to infer wallet structure or ownership
- The requirement for manual review before PSBT export

These behaviors are deliberate and preserve the tool’s security and privacy guarantees.

---

## Design Position

Ωmega Pruner does not attempt to eliminate risk.

It attempts to **make risk legible before it becomes irreversible**.

Any consolidation decision made with incomplete information is the user’s responsibility —  
the tool’s responsibility is to ensure that **no risk is obscured by design**.

As an infrastructure layer, it provides wallets and services with the primitives to surface these tradeoffs explicitly, enabling more informed structural decisions across the ecosystem.

---

## Explicit Non-Goals

Ωmega Pruner deliberately does **not** attempt to provide:

- Wallet clustering or address attribution
- Cross-wallet or multi-account inference
- Heuristic enrichment beyond visible CIOH signals
- Automatic consolidation or silent optimization
- Fee prediction beyond observed historical context

These exclusions prevent silent linkage, hidden assumptions, and irreversible privacy errors.

---

## Current Limitations

- Single-address scope only
- Legacy and Nested SegWit inputs cannot be consolidated
- No automatic Taproot derivation path recovery
- No batch, multi-wallet, or coordinator functionality

---

## Status

**Active experiment.**

The project prioritizes correctness, determinism, and visible tradeoffs over convenience.

Critical review and adversarial feedback are encouraged.
