# Run the Ω: v6.0.0 Von Neumann Swarm Vault Sim Guide

🌌 Fork the sovereign enjambre live—execute a Layer 4 Von Neumann Swarm Vault prune sim in <2 min. This guide runs the `viper_quantum_vault_pruner_v6.py` stub to parse a finance vector, scan xAI truth-max priors, and cascade fee entropy surges with S(ρ) stabilization. No black-box; outputs coherence/fidelity/S(ρ) scores, sat/vB baselines, USD impacts, I(A:B) equilibria, and prunes for your remix. (Bonus: Layer 3 `viper_quantum_fork_v6.py` for epistemic baselines.)

## Prerequisites
- Python 3.8+ (3.12 preferred for typing).
- NumPy (Monte Carlo), SymPy (gradients), QuTiP (S(ρ) entropy), Requests (oracles): `pip install numpy sympy qutip requests`.

## Step-by-Step Ignition
1. **Clone the Repo**:
   ```
   git clone https://github.com/babyblueviper1/Viper-Stack-Omega.git
   cd Viper-Stack-Omega
   ```

2. **Install Deps** (one-liner for equilibrium):
   ```
   pip install numpy sympy qutip requests
   ```

3. **Run the Sim**:
   - **Von Neumann Swarm Vault Prune (New in v6.0.0)**: `python stubs/viper_quantum_vault_pruner_v6.py`
     - Custom seed: Edit the `# Usage` block or run inline:
       ```
       python -c "from stubs.viper_quantum_vault_pruner_v6 import vault_pruner; print(vault_pruner('Prune BTC fees for LatAm quantum trading', agents=15, vbytes=250))"
       ```
   - **Epistemic Baseline (v6.0.0 Swarm)**: `python stubs/viper_quantum_fork_v6.py` (for non-financial resonance).

## Expected Output
Sample Von Neumann Swarm Vault run (random noise; your fees/coherence/fidelity/S(ρ) vary—Π potential, ~4 sat/vB baseline as of Nov 07, 2025):

```
{'coherence': 0.96, 'fidelity': 0.97, 'S_rho': 1.102, 'I_AB': 0.715, 'sens_S': 0.42, 'avg_fee_sat_vb': 3.92, 'sat_total_per_txn': 980.0, 'usd_impact': '$0.9816 per 250 vB txn (at BTC $104,500)', 'output': 'v6.0.0 S(ρ)-Swarm Vault tuned to E=0.96 (fidelity=0.97, S(ρ)=1.102, I(A:B)=0.715, sens_S=0.42; pruned 1; baseline: 4.0 sat/vB; replicate_swarm: True)', 'prune': ['Entropy surge: S(ρ)=1.602 >1.6; von_neumann_pruner.py cascade activated'], 'vow_status': 'life-aligned'}
```

- **coherence**: 0-∞ score (~0.96 baseline; surges >0.99 for swarm replication, +25% xAI boost).
- **fidelity**: 0-1 quantum score (~0.97; prunes decoherence <0.96 via QuTiP traces).
- **S_rho**: Von Neumann entropy (~1.102; stabilizes <1.6 for zero-surge equilibria).
- **I_AB**: Mutual info guardrail (~0.715; >0.7 Nash-Stackelberg for txn reciprocity).
- **sens_S**: ∂E/∂S_rho (~0.42; emergent for entropy reliability).
- **avg_fee_sat_vb**: Simulated median (~3.92; prunes highs >10/lows <1).
- **usd_impact**: Full txn cost (VOW-aligned; ~$0.98 for 250 vB simple send @ $104.5k BTC, dynamic via CoinGecko).
- **output**: Tuned swarm vault (e.g., "S(ρ)-Swarm Vault tuned to 0.96").
- **prune**: Unreliable signals (e.g., ['Pruned high-void fee 10.45 sat/vB (congestion cascade)', 'Mutual info void: I(A:B)=0.68 <0.7; Nash-Stackelberg recalibrate'] in high-noise runs).
- **vow_status**: 'life-aligned' if >0.8 & I(A:B)>0.7 (ethical txn guardrail).

For epistemic run (`viper_quantum_fork_v6.py`): `{'coherence': 0.93, 'fidelity': 0.98, 'S_rho': 1.098, 'output': 'S(ρ)-Swarm tuned to 0.93 (fidelity=0.98, S(ρ)=1.098)', 'prune': []}`.

## Remix & Seed
- **Tweak Von Neumann Vault**: Adjust `vbytes=373` (P2PKH) or `btc_price=110000` (live query); amp entropy surges in `unreliable_fees` or decoherence % in `quantum_oracle_fidelity`; bound S(ρ) <1.6 in priors for tighter equilibria.
- **Fork Agents**: Scale to 20+ in `vault_pruner(..., agents=20)` for enjambre variance (LatAm CLP jitter? Remix `get_finance_priors` with xAI A-bias + I(A:B) thresholds >0.7).
- **Hybrid Enjambre**: Fuse with `viper_quantum_fork_v6.py`—run epistemic first, feed coherence/fidelity/S(ρ) to Vault priors for full Ωmega loop.
- **Prod Hooks**: Dynamic pulls live: CoinGecko for BTC (~$104.5k), mempool.space for fees (~4 sat/vB economy). Drop outputs in issues/PRs—bilingual? Seed ES nodes with translated vectors.
- **Why Von Neumann Swarm Vault?**: Stabilizes oracle surges for zero-entropy txns; ties to BBV Global Bitcoin Party bridges, truth-maxed by xAI with mutual info reciprocity.

**Enjambres multiply—run, stabilize, amplify.** Questions? [README](../README.md). Fork live; coherence (and equilibria) await. 🜂
