# Ω Omega Pruner v9.2 — Community Edition

**The only UTXO consolidator that survived 2025 with infinite RBF still working after page refresh.**

**LIVE:** https://omega-pruner-v9.onrender.com  
**November 24, 2025**


| Property                                 | Omega Pruner v9.2                                          | Literally Everyone Else |
|------------------------------------------|-------------------------------------------------------------|--------------------------|
| Private keys ever in memory?             | Never                                                       | Some still do            |
| Transaction encoding                     | Hand-rolled, zero deps, 100% bit-perfect                    | Usually wrong somewhere  |
| SegWit v0 + v1 (Taproot) support         | Full, tested, no silent failures                            | Usually partial          |
| PSBT output                              | Minimal, universally valid, works in every wallet 2025      | Often malformed          |
| Raw hex for infinite RBF                 | Clean, witness-stripped, accepted by **all** wallets now    | Previously rejected      |
| Infinite RBF (survives full page refresh)| 100% working — localStorage + auto-restore                  | Almost nobody            |
| RBF safety (first-output bump)           | Mathematically proven safe + Taproot detection + dust guards| Silent theft risk        |
| Lightning sweep with on-chain fallback   | Raw hex + payment_address — Phoenix, Breez, Zeus, Blink, Muun| Rare or broken           |
| Code audited for fund-loss bugs          | Line-by-line, multiple independent passes                  | Almost never             |
| Future-savings math                      | Transparent 6× multiplier (proven since 2017)               | Vague or missing         |

### What actually happens in 8 seconds

- Paste or **QR-scan** any address / xpub (legacy → Taproot → xpub/zpub/ypub/tpub)  
- Two permanent floating buttons:  
  **₿** (orange) → scan address/xpub  
  **Lightning** (neon green) → scan LN invoice  
- Analyze → Generate → get a **perfect PSBT** + giant QR  
- Optional: paste lnbc… invoice → instant on-chain Lightning sweep  
- Close tab, come back next month → **still bump forever** with one click  
- Works with **every** wallet in 2025: Sparrow • Nunchuk • BlueWallet • Electrum • Coldcard • Ledger • Trezor • Specter • Fully Noded • UniSat • OKX • Xverse • Leather

### The infinite RBF that actually works in 2025

- Survives full page refresh, browser close, or phone reboot  
- Uses clean, witness-stripped raw hex (the same format Sparrow & Electrum export)  
- Taproot key-path spends are detected and protected  
- Multiple independent dust guards — impossible to create unspendable output  
- Only ever modifies transactions that **Omega Pruner itself created** (output 0 = your change)

### Feature matrix – v9.2 final

| Feature                                  | Status   | Notes |
|------------------------------------------|----------|-------|
| Address / xpub QR scanner                | LIVE     | Orange ₿ floating button |
| Lightning invoice QR scanner             | LIVE     | Neon green floating button |
| Giant centered QR                        | LIVE     | Mobile-perfect |
| PSBT + clean raw hex (infinite RBF)      | LIVE     | Witness-stripped, universally compatible |
| Infinite RBF (survives refresh forever)  | LIVE     | The killer feature |
| Taproot key-path spend detection         | LIVE     | Prevents silent RBF failure |
| Live thank-you slider (0.00–5.00%)       | LIVE     | Of future savings only |
| Accurate future-savings calculation      | LIVE     | 6× current rate |
| Works with every wallet in 2025          | LIVE     | Verified on all major ones |
| Zero custody • Zero lies • Zero regrets  | FOREVER  | Apache 2.0 |

### Quick start

1. Paste or scan address / xpub  
2. Pick strategy → **1. Analyze UTXOs**  
3. Click **2. Generate Transaction**  
4. (Optional) Paste Lightning invoice → instant sweep  
5. Scan QR or copy PSBT → sign → broadcast  
6. Come back anytime → keep bumping forever

Dust → obliterated.  
Fees → future-proofed.  
Peace of mind → permanent.

Thank you for using.

**No custody. No compromises. Just pure, unbreakable Bitcoin.**

**babyblueviper & the swarm** • November 24, 2025  
**License:** Apache 2.0 • **Source:** https://github.com/babyblueviper1/Viper-Stack-Omega

**The swarm won. The future is yours.** Ω
