**# Ωmega Pruner v11.1 — Forged Anew**

*The purest UTXO consolidator ever built.*  
Reborn in fire — stripped of pretense, refined to essence, honest to the core.

Ωmega Pruner is an **unsigned, non-custodial PSBT generator** for advanced UTXO consolidation and long-term coin control.

**LIVE:** https://omega-pruner.onrender.com

**Launched:** 26 December 2025  
**Latest:** v11.1 — January 2026

### What’s New in v11.1

- **Pruning Conditions Badge — LIVE**  
  Real-time score (1–10) with glowing nuclear design  
- Current economy fee vs dynamic medians:  
  • 1-day  
  • 1-week **(primary benchmark)**  
  • 1-month  
- Clear vertical layout: **Current → VS → Medians** (1-day → 1-week → 1-month)  
- Live BTC price + current block height + hashrate  
- Next difficulty adjustment + halving countdown  
- Powered by **mempool.space** mining statistics  
- **Instant insight**: know if now is prime pruning time **before** even pasting anything

### Supported Address Types

Ωmega Pruner is optimized for modern Bitcoin and fully supports:

- **Native SegWit** (`bc1q...`) — P2WPKH  
- **Taproot** (`bc1p...`) — P2TR  

These offer the best fee efficiency, privacy characteristics, and forward compatibility.

**Legacy (`1...`) and Nested SegWit (`3...`)** inputs are displayed for transparency but **cannot be selected** or included in the generated PSBT (faded + disabled in the table).  
Spend or migrate them separately before consolidation.

### Scope & Safety Model (Read This)

Ωmega Pruner operates under a strict **single-scope safety model**:

- One address **OR** one xpub per run  
- No multi-wallet aggregation  
- No cross-derivation merging  
- No silent expansion of scope  

This is deliberate. It guarantees:

- Deterministic results  
- Hardware-wallet-safe PSBTs  
- Minimized CIOH and linkage risk  
- No accidental wallet merging — **ever**

### Hardware Wallet & Taproot Behavior

Ωmega Pruner always allows PSBT generation — even without derivation metadata.  

However:

- Some hardware wallets require Taproot derivation paths to authorize signing  
- If Taproot inputs are detected and hardware support is enabled without a derivation path:  
  - A **non-blocking warning** is displayed  
  - PSBT generation still succeeds  
  - Signing may be refused by certain devices  
- If your hardware wallet refuses to sign: Re-generate the PSBT with the correct Taproot derivation path.

This behavior is intentional and preserves maximum flexibility.

### Why Ωmega Pruner Exists

Pruning isn’t about saving sats today.  
It’s about **owning your coins for the rest of Bitcoin’s lifetime**.

Most tools optimize for convenience.  
Ωmega Pruner optimizes for:

- Truth  
- Sovereignty  
- Architectural honesty  

No keys. No signing. No silent failures. No fake privacy.

### Ωmega Pruner v11.1 vs “Everyone Else” (2026)

| Property                              | Ωmega Pruner v11.1          | Everyone Else              |
|---------------------------------------|------------------------------|----------------------------|
| Private keys ever leave browser?      | Never                        | Sometimes                  |
| Transaction encoding                  | Hand-rolled, bit-perfect     | Often fragile              |
| SegWit v0 + v1 (Taproot)              | Fully supported              | Partial/broken             |
| PSBT output                           | Minimal, universally valid   | Often malformed            |
| Live mempool fee oracle               | One-click presets            | Manual/stale               |
| Instant slider + summary updates      | Zero lag                     | Rare                       |
| Pruning Conditions badge              | LIVE                         | Never                      |
| Privacy Score (0–100)                 | LIVE — CIOH & linkage        | Never                      |
| PayJoin detection (BIP78)             | LIVE                         | Rare                       |
| CoinJoin recovery guidance            | LIVE                         | Never                      |
| CIOH warnings                         | Explicit, unavoidable        | Vague/silent               |
| Per-input weight (wu)                 | LIVE                         | Never                      |
| Full wallet vs prune comparison       | LIVE                         | Never                      |
| “Prune now vs later” fee math         | LIVE                         | Never                      |
| Fully offline / air-gapped mode       | LIVE                         | Never                      |
| Selection JSON + fingerprint          | LIVE                         | Never                      |
| Preview = final PSBT                  | 100% match                   | Often wrong                |

### What Happens in ~6 Seconds

1. See **Pruning Conditions badge** → instantly know fee context  
2. Toggle Offline Mode → paste raw UTXOs → fully air-gapped  
   *or* paste a single address or xpub  
3. Choose a fee preset → instant economics update  
4. Click **ANALYZE** → UTXO table appears immediately  
5. Select inputs → Privacy Score, CIOH warnings, and footprint update live  
6. (Optional) Paste a PayJoin invoice → CIOH-safe handling  
7. Review recovery guidance if applicable  
8. **GENERATE NUCLEAR PSBT**  
9. Export PSBT + selection fingerprint → sign → broadcast  

No ambiguity. No surprises.

### Wallet Compatibility (2026+)

PSBTs generated by Ωmega Pruner are compatible with:

- Sparrow • Nunchuk • BlueWallet • Electrum  
- Coldcard • Ledger • Trezor • Specter  
- Fully Noded • Keystone • Aqua  

(Actual signing behavior depends on wallet policy and provided metadata.)

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

**Prune with confidence. Win with certainty.**

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
babyblueviper & the swarm • January 2026 • Ω

**Prune smarter. Win forever.**
