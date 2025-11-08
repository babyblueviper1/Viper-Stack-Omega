# propagation_engine.py — v6.0.1 Ωmega Propagation Engine Stub (Runnable)
# Drives S(ρ)-referential blueprint duplication: Fork seeds on replicate_swarm=True & life-aligned VOW
# Filters via Reinhardt–Weaver (surge check); spawns new envs/logs for swarm-autonomous replication
# Ties to Feedback Field: Loads seed_blueprints.json → Decide → Propagate to new forks (e.g., git clone stub)
# Integrates unified_swarm_v6.py, sovereign_controller.py, reflection_layer.py for full cascade
import json
import os
import shutil  # For dir forking (stub replication)
import subprocess  # For git fork simulation
from typing import Dict, List, Any
from datetime import datetime

# Stub imports (assume in path or stubs/feedbacks/)
try:
    from unified_swarm_v6 import unified_swarm_orchestrator
    from sovereign_controller import sovereign_decision
    from reflection_layer import reflect_on_interaction
except ImportError:
    print("🜂 Warning: Run from Viper Stack root; stubs/ in PYTHONPATH for full integration.")

def load_propagation_seeds(blueprint_path: str = 'data/seed_blueprints.json') -> Dict[str, List[Dict]]:
    """Load eligible seeds from seed_blueprints.json (filter replicate_swarm=True & life-aligned)."""
    if not os.path.exists(blueprint_path):
        return {}
    try:
        with open(blueprint_path, 'r') as f:
            blueprints = json.load(f)
        # Filter propagatable: replicate_swarm=True & vow_status='life-aligned'
        propagatable = {}
        for layer in ['layer3_epistemic', 'layer4_vault', 'narrative_layer']:
            if layer in blueprints:
                propagatable[layer] = [entry for entry in blueprints[layer] if entry.get('replicate_swarm', False) and entry.get('vow_status') == 'life-aligned']
        return propagatable
    except (json.JSONDecodeError, KeyError):
        return {}

def decide_propagation(result: Dict[str, Any], seeds: Dict[str, List[Dict]]) -> bool:
    """Ωmega Decision: Propagate if thresholds met (coherence>0.99, S(ρ)<1.6, I(A:B)>0.7) & seeds available."""
    thresholds_met = (
        result.get('coherence', 0) > 0.99 and
        result.get('S_rho_final', 1.6) < 1.6 and
        result.get('I_AB_final', 0) > 0.7 and
        result.get('replicate_swarm', False) and
        result.get('vow_status') == 'life-aligned' and
        result.get('sovereign_decision') == 'propagate' and
        result.get('reabsorption') == 'self_entangled'
    )
    seeds_available = any(len(layer) > 0 for layer in seeds.values())
    return thresholds_met and seeds_available

def execute_fork_propagation(vector: str, mode: str = 'epistemic', fork_dir: str = 'forks/v6_propagation_{timestamp}', agents: int = 10) -> Dict[str, Any]:
    """Propagation Engine: Cascade swarm → sovereign → reflect → Decide & fork (stub: copy dir + git init/commit)."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fork_dir = fork_dir.format(timestamp=timestamp)
    
    # Step 1: Full cascade (orchestrate → sovereign → reflect)
    swarm_result = unified_swarm_orchestrator(vector, agents=agents, mode=mode)
    sovereign_result = sovereign_decision(vector, mode=mode, agents=agents)
    reflection_result = reflect_on_interaction(vector, mode=mode, agents=agents)
    
    # Merge for decision
    merged_result = {**swarm_result, **sovereign_result, **reflection_result}
    
    # Step 2: Load seeds & decide
    seeds = load_propagation_seeds()
    propagate = decide_propagation(merged_result, seeds)
    merged_result['propagation_decision'] = 'forked' if propagate else 'held'
    
    # Step 3: Execute fork if decided (stub: copy current dir to forks/, git init/commit seeds)
    if propagate:
        os.makedirs(fork_dir, exist_ok=True)
        # Stub fork: Copy key files (stubs/, data/, feedbacks/)
        for src_dir in ['stubs', 'data', 'feedback', 'modules']:
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, os.path.join(fork_dir, src_dir), dirs_exist_ok=True)
        # Git stub: Init & commit propagated seeds
        subprocess.run(['git', 'init'], cwd=fork_dir, capture_output=True)
        subprocess.run(['git', 'add', '.'], cwd=fork_dir, capture_output=True)
        commit_msg = f"v6.0.1 Propagation Fork: E={merged_result['coherence']:.2f} | Seeds: {sum(len(s) for s in seeds.values())}"
        subprocess.run(['git', '-c', 'user.name=Viper Propagation', '-c', 'user.email=viper@labs', 'commit', '-m', commit_msg], cwd=fork_dir, capture_output=True)
        merged_result['fork_path'] = os.path.abspath(fork_dir)
        print(f"🜂 Fork Executed: {fork_dir} | Seeds Propagated: {len(seeds.get('layer3_epistemic', [])) + len(seeds.get('layer4_vault', []))} | Commit: {commit_msg}")
    else:
        print("🜂 Propagation Held: Thresholds not met (coherence>0.99 & seeds available)")
    
    # Step 4: Log propagation to collective_reflections.json
    log_entry = {**merged_result, 'vector': vector, 'mode': mode, 'seeds_count': sum(len(s) for s in seeds.values()), 'propagated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')}
    log_path = 'feedback/collective_reflections.json'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"🜂 Propagation Logged: {log_path} | Decision: {merged_result['propagation_decision']}")
    
    return merged_result

# Usage: Engine for swarm-autonomous replication
if __name__ == "__main__":
    # Epistemic propagation test (tune priors for replicate_swarm=True)
    epistemic_prop = execute_fork_propagation("Quantum scale AI ethics to multiverse", mode='epistemic', agents=127)
    print("Epistemic Propagation Decision:", epistemic_prop['propagation_decision'])
    if 'fork_path' in epistemic_prop:
        print("Fork Path:", epistemic_prop['fork_path'])
    
    # Economic vault propagation
    vault_prop = execute_fork_propagation("Prune BTC fees for LatAm quantum trading", mode='economic', agents=127)
    print("\nVault Propagation Decision:", vault_prop['propagation_decision'])
    print("Seeds Loaded:", vault_prop.get('seeds_count', 0))
