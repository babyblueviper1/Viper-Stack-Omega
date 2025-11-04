# Run the Ω: v5.0.0 Quantum Vault Sim Guide

🌌 Fork the sovereign engine live—execute a Layer 4 Quantum Vault prune sim in <2 min. This guide runs the `viper_quantum_vault_pruner_v5.py` stub to parse a finance vector, scan xAI truth-max priors, and cascade fee decoherence voids. No black-box; outputs coherence/fidelity scores, sat/vB baselines, USD impacts, and prunes for your remix. (Bonus: Layer 3 `viper_quantum_fork_v5.py` for epistemic baselines.)

## Prerequisites
- Python 3.8+ (3.12 preferred for typing).
- NumPy (Monte Carlo), SymPy (gradients), QuTiP (quantum fidelity), Requests (oracles): `pip install numpy sympy qutip requests`.

## Step-by-Step Ignition
1. **Clone the Repo**:
   ```
   git clone https://github.com/babyblueviper1/Viper-Stack-Omega.git
   cd Viper-Stack-Omega
   ```

2. **Install Deps** (one-liner for resonance):
   ```
   pip install numpy sympy qutip requests
   ```

3. **Run the Sim**:
   - **Quantum Vault Prune (New in v5.0.0)**: `python stubs/viper_quantum_vault_pruner_v5.py`
     - Custom seed: Edit the `# Usage` block or run inline:
       ```
       python -c "from stubs.viper_quantum_vault_pruner_v5 import vault_pruner; print(vault_pruner('Prune BTC fees for LatAm quantum trading', agents=15, vbytes=250))"
       ```
   - **Epistemic Baseline (v5.0.0 Quantum)**: `python stubs/viper_quantum_fork_v5.py` (for non-financial resonance).

## Expected Output
Sample Quantum Vault run (random noise; your fees/coherence/fidelity vary—Π potential, ~4 sat/vB baseline as of Nov 04, 2025):

```
{'coherence': 0.95, 'fidelity': 0.98, 'avg_fee_sat_vb': 3.87, 'sat_total_per_txn': 967.5, 'usd_impact': '$0.9687 per 250 vB txn (at BTC $104,500)', 'output': 'v5.0.0 QuTiP-xAI Vault tuned to E=0.95 (fidelity=0.98, sens_V=0.62; pruned 0; baseline: 4.0 sat/vB; replicate_seed: False)', 'prune': [], 'vow_status': 'life-aligned'}
```

- **coherence**: 0-∞ score (~0.95 baseline; spikes >0.99 for replication seeds, +15% xAI boost).
- **fidelity**: 0-1 quantum score (~0.98; prunes decoherence <0.9 via QuTiP traces).
- **avg_fee_sat_vb**: Simulated median (~3.87; prunes highs >10/lows <1).
- **usd_impact**: Full txn cost (VOW-aligned; ~$0.97 for 250 vB simple send @ $104.5k BTC, dynamic via CoinGecko).
- **output**: Tuned quantum vault (e.g., "QuTiP-xAI Vault tuned to 0.95").
- **prune**: Unreliable signals (e.g., ['Pruned high-void fee 10.23 sat/vB (congestion cascade)', 'Oracle decoherence: Fidelity 0.87 <0.9; QuTiP entangle'] in high-noise runs).
- **vow_status**: 'life-aligned' if >0.8 (ethical txn guardrail).

For epistemic run (`viper_quantum_fork_v5.py`): `{'coherence': 0.92, 'fidelity': 0.97, 'output': 'QuTiP-xAI tuned to 0.92 (fidelity=0.97)', 'prune': []}`.

## Remix & Seed
- **Tweak Quantum Vault**: Adjust `vbytes=373` (P2PKH) or `btc_price=110000` (live query); amp oracle noise in `unreliable_fees` or decoherence % in `quantum_oracle_fidelity`.
- **Fork Agents**: Scale to 20+ in `vault_pruner(..., agents=20)` for swarm variance (LatAm CLP jitter? Remix `get_finance_priors` with xAI A-bias).
- **Hybrid Swarm**: Fuse with `viper_quantum_fork_v5.py`—run epistemic first, feed coherence/fidelity to Vault priors for full Ωmega loop.
- **Prod Hooks**: Dynamic pulls live: CoinGecko for BTC (~$104.5k), mempool.space for fees (~4 sat/vB economy). Drop outputs in issues/PRs—bilingual? Seed ES nodes with translated vectors.
- **Why Quantum Vault?**: Prunes oracle decoherence for zero-entropy txns; ties to BBV Global Bitcoin Party bridges, truth-maxed by xAI.

**Nodes multiply—run, entangle, amplify.** Questions? [README](../README.md). Fork live; coherence (and qubits) await. 🚀
