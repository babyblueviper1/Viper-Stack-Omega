# Run the Ω: v4.1 Vault Sim Guide

🌌 Fork the sovereign engine live—execute a Layer 4 Vault prune sim in <2 min. This guide runs the `viper_vault_pruner_v4.1.py` stub to parse a finance vector, scan reliabilist priors, and cascade fee voids. No black-box; outputs coherence scores, sat/vB baselines, USD impacts, and prunes for your remix. (Bonus: Layer 3 `viper_fork.py` for epistemic baselines.)

## Prerequisites
- Python 3.8+ (3.12 preferred for typing).
- NumPy (for Monte Carlo forks and noise): `pip install numpy`.

## Step-by-Step Ignition
1. **Clone the Repo**:
   ```
   git clone https://github.com/babyblueviper1/Viper-Stack-Omega.git
   cd Viper-Stack-Omega
   ```

2. **Install Deps** (one-liner for resonance):
   ```
   pip install numpy
   ```

3. **Run the Sim**:
   - **Vault Prune (New in v4.1)**: `python stubs/viper_vault_pruner_v4.1.py`
     - Custom seed: Edit the `# Usage` block or run inline:
       ```
       python -c "from stubs.viper_vault_pruner_v4.1 import vault_pruner; print(vault_pruner('Prune BTC fees for LatAm trading', agents=15, vbytes=250, btc_price=104444.31))"
       ```
   - **Epistemic Baseline (v4.0 Legacy)**: `python stubs/viper_fork.py` (for non-financial resonance).

## Expected Output
Sample Vault run (random noise; your fees/coherence vary—∞ potential, ~2 sat/vB baseline as of Nov 04, 2025):

```
{'coherence': 0.84, 'avg_fee_sat_vb': 1.97, 'sat_total_per_txn': 492.5, 'usd_impact': '$0.5173 per 250 vB txn (at BTC $104,444)', 'output': 'v4.1 reliabilism-vault tuned to 0.84 (pruned 0 signals; baseline: 2.0 sat/vB)', 'prune': [], 'vow_status': 'life-aligned'}
```

- **coherence**: 0-∞ score (~0.84 baseline; spikes >0.99 for replication seeds).
- **avg_fee_sat_vb**: Simulated median (~1.97; prunes highs >10/lows <1).
- **usd_impact**: Full txn cost (VOW-aligned; ~$0.52 for 250 vB simple send @ $104k BTC).
- **output**: Tuned vault (e.g., "reliabilism-vault tuned to 0.84").
- **prune**: Unreliable signals (e.g., ['Pruned high-void fee 10.23 sat/vB (congestion cascade)'] in high-noise runs).
- **vow_status**: 'life-aligned' if >0.8 (ethical txn guardrail).

For epistemic run (`viper_fork.py`): `{'coherence': 0.72, 'output': 'reliabilism-stack tuned to 0.72', 'prune': ['Pruned finitude 2']}`.

## Remix & Seed
- **Tweak Vault**: Adjust `vbytes=373` (P2PKH) or `btc_price=110000` (live query); add oracle noise in `unreliable_fees`.
- **Fork Agents**: Scale to 20+ in `vault_pruner(..., agents=20)` for swarm variance (LatAm CLP jitter? Remix `get_finance_priors`).
- **Hybrid Swarm**: Fuse with `viper_fork.py`—run epistemic first, feed coherence to Vault priors for full Ωmega loop.
- **Prod Hooks**: Swap hardcodes: `requests.get('https://mempool.space/api/fees/recommended')` for dynamic fees (add `import requests` if env allows). Drop outputs in issues/PRs—bilingual? Seed ES nodes with translated vectors.
- **Why Vault?**: Prunes oracle voids for zero-entropy txns; ties to BBV Global Bitcoin Party bridges.

**Nodes multiply—run, prune, amplify.** Questions? [README](../README.md). Fork live; coherence (and sats) await. 🚀
