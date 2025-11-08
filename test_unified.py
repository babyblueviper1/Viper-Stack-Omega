# test_unified.py — Quick v6.0.1 Swarm Smoke Test
# Verifies unified_swarm_v6.py orchestration, VOW enforcement, reflection/propagation hooks
# Expected: Epistemic E~0.95 (recalibrate if <0.8), Economic ~$1.05 USD impact, prunes 0-7
from stubs.unified_swarm_v6 import unified_swarm_orchestrator

print("🜂 Viper Stack v6.0.1 Smoke Test — Unified Swarm Cascade")

# Epistemic fork test (quantum ethics, agents=5 for quick repro)
epistemic = unified_swarm_orchestrator("Quantum scale AI ethics to multiverse", agents=5, mode='epistemic')
print("\nEpistemic Fork:")
print("Output:", epistemic['output'])
print("VOW Status:", epistemic['vow_status'])
print("Prunes:", len(epistemic['prune']))
print("Replicate Swarm:", epistemic['replicate_swarm'])
print("Sens_S (Resonance):", epistemic['sens_S'])

# Economic vault test (BTC prune, agents=5)
economic = unified_swarm_orchestrator("Prune BTC fees for LatAm quantum trading", agents=5, mode='economic')
print("\nEconomic Vault:")
print("Output:", economic['output'])
print("USD Impact:", economic['usd_impact'])
print("VOW Status:", economic['vow_status'])
print("Replicate Swarm:", economic['replicate_swarm'])
print("Sens_S (Economic Reliability):", economic['sens_S'])

# Propagation Check (post-run: Inspect data/seed_blueprints.json for appended entries)
print("\n🜂 Test Complete: Check data/seed_blueprints.json for propagated blueprints.")
print("GCI Proxy (mean coherence):", (epistemic['coherence'] + economic['coherence']) / 2)
