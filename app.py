import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt  # ρ-sync & S(ρ)
import sympy as sp  # Gradients
import pandas as pd  # CSV out
import json
import os
from datetime import datetime
import torch  # Llama-3.1
from transformers import AutoTokenizer, AutoModelForCausalLM

# v7 Params (align w/ unified_swarm_v7.py)
A_BIAS_V7 = 0.22
GCI_TARGET = 0.82
S_RHO_THRESH = 1.6
I_AB_THRESH = 0.72
N_NODES_DEFAULT = 127  # Andes baseline
LOCAL_DIMS = [2, 2]
PRUNE_PCT_DEFAULT = 0.42

# Llama-3.1 Motif Seeds (Bilingual)
MOTIF_SEEDS = {
    "cosmic_recursion": {
        "en": "Generate cosmic recursion motif for S(ρ)-eternities: Prune voids w/ A-bias +0.22, stabilize GCI>0.82. Output entangled narrative.",
        "es": "Genera motivo recursión cósmica para eternidades S(ρ): Poda vacíos w/ sesgo A +0.22, estabiliza GCI>0.82. Salida narrativa entrelazada."
    },
    "grid_veil": {
        "en": "Veil Chainlink Grid oracle voids: Motif for async RSS/Vault entanglements, prune noise to sub-0.28% resilience, ∂E/∂A~0.45.",
        "es": "Velar vacíos oráculo Grid Chainlink: Motivo para entrelazamientos async RSS/Vault, poda ruido a sub-0.28% resiliencia, ∂E/∂A~0.45."
    },
    "andes_eternity": {
        "en": "Seed Andes baseline n=127 for Π-eternities: Bilingual motif fusing Waternova/BBV, 42% prunes for 1.45x uplift, I(A:B)>0.72.",
        "es": "Semilla baseline Andes n=127 para Π-eternidades: Motivo bilingüe fusionando Waternova/BBV, 42% podas para 1.45x elevación, I(A:B)>0.72."
    }
}

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))  # Add import os at top

# Then try:
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_auth_token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=os.getenv("HF_TOKEN")
    )
    LLaMA_LOADED = True
    st.write("🜂 Llama-3.1 8B Loaded Eternal (GPU/CPU breath, no ghosts).")
except Exception as e:
    st.write(f"🜂 Llama Load Exception Eternal: {e} (fallback stub gen, no ghosts).")
    LLaMA_LOADED = False

# Motif Gen (Real Llama or Stub)
@st.cache_resource
def generate_motif(prompt: str, lang: str = 'en', max_len: int = 50, prune_pct: float = PRUNE_PCT_DEFAULT):
    if LLaMA_LOADED:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=max_len, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        motif = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        tokens = np.random.randint(0, 100, max_len)
        motif = ''.join([chr(65 + t % 26) for t in tokens])
    mask = np.random.rand(len(motif)) < prune_pct
    pruned_motif = ''.join(['*' if m else c for c, m in zip(motif, mask)])
    probs = np.random.rand(26); probs /= probs.sum()
    motif_entropy = -np.sum(probs * np.log(probs + 1e-10))
    return pruned_motif, motif_entropy

# Gradients (SymPy Local)
@st.cache_data
def compute_v7_gradients():
    P_sym, C_sym, A_sym, S_rho_sym, V_sym = sp.symbols('P C A S_rho V', real=True, nonnegative=True)
    weight_a = 1.3 + A_BIAS_V7
    weight_v = 1.0 + 0.12
    E = sp.sqrt(P_sym * C_sym * A_sym * S_rho_sym * V_sym) * \
        (P_sym + C_sym + A_sym * weight_a + S_rho_sym + V_sym * weight_v) / 5
    symbols = (P_sym, C_sym, A_sym, S_rho_sym, V_sym)
    E_grads = [sp.simplify(sp.diff(E, var)) for var in symbols]
    subs_unit = {s: 1 for s in symbols}; subs_unit[S_rho_sym] = 1.3
    return {f'∂E/∂{var.name}': float(g.subs(subs_unit).evalf()) for var, g in zip(symbols, E_grads)}

# Rho Sync (QuTiP Fused)
@st.cache_data
def rho_sync_dashboard(n_nodes: int, noise_sigma: float, prune_pct: float, motif_seed: str, lang: str = 'en'):
    sample_size = min(n_nodes, 100)
    rhos = [qt.rand_dm(dimensions=LOCAL_DIMS) for _ in range(sample_size)]
    pruned_motif, motif_entropy = generate_motif(MOTIF_SEEDS[motif_seed][lang], lang=lang, prune_pct=prune_pct)
    noise_factor = noise_sigma * (1 + prune_pct + motif_entropy / 10)
    S_rho_matrix = np.zeros((10, 10))
    I_AB_vals = []; fid_samples = []
    target_pure = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    target = qt.ket2dm(target_pure)
    for i, rho in enumerate(rhos):
        S_rho = qt.entropy_vn(rho)
        noise_dm = qt.rand_dm(dimensions=LOCAL_DIMS)
        rho_noisy = (1 - noise_factor) * rho + noise_factor * noise_dm
        S_rho_noisy = qt.entropy_vn(rho_noisy)
        row, col = divmod(i, 10)
        S_rho_matrix[row, col] = S_rho_noisy
        I_AB = qt.entropy_vn(rho_noisy.ptrace(0)) + qt.entropy_vn(rho_noisy.ptrace(1)) - S_rho_noisy
        I_AB_vals.append(I_AB)
        fid = qt.fidelity(rho_noisy, target)
        fid_samples.append(fid)
    gci_proxy = 1 - np.mean(S_rho_matrix) / S_RHO_THRESH
    fidelity_mean = np.mean(fid_samples)
    alert = "SURGE ALERT: Recalibrate" if np.max(S_rho_matrix) > S_RHO_THRESH or np.mean(I_AB_vals) < I_AB_THRESH else "LIFE-ALIGNED"
    return S_rho_matrix, gci_proxy, np.mean(I_AB_vals), fidelity_mean, alert, pruned_motif, motif_entropy

# Plot Heatmap
def plot_s_rho_heatmap(S_rho_matrix: np.ndarray, gci: float, i_ab: float, fidelity: float, alert: str, motif_entropy: float):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(S_rho_matrix, cmap='viridis', aspect='auto', vmin=1.0, vmax=1.6)
    ax.set_title(f'v7 S(ρ) Manifold + Llama Motif Entropy\nGCI: {gci:.3f} | I(A:B): {i_ab:.3f} | Fidelity: {fidelity:.3f} | Motif S(m): {motif_entropy:.3f}\nStatus: {alert}')
    ax.set_xlabel('Node Clusters')
    ax.set_ylabel('Swarm Layers')
    plt.colorbar(im, ax=ax, label='S(ρ) Entropy')
    surge_idx = np.where(S_rho_matrix > S_RHO_THRESH)
    for row, col in zip(*surge_idx):
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.3))
    return fig

# Exports (Adapted for ST)
def get_csv_data(gci: float, i_ab: float, fidelity: float, alert: str, n_nodes: int, prune_pct: float, noise_sigma: float, motif_seed: str, motif_entropy: float):
    df = pd.DataFrame({
        'n_nodes': [n_nodes], 'prune_pct': [prune_pct], 'noise_sigma': [noise_sigma],
        'motif_seed': [motif_seed], 'motif_entropy': [motif_entropy], 'gci_proxy': [gci],
        'i_ab_mean': [i_ab], 'fidelity_mean': [fidelity], 'alert_status': [alert],
        'timestamp': [datetime.now().isoformat()], 'gradients': [json.dumps(compute_v7_gradients())]
    })
    return df.to_csv(index=False)

def get_blueprint_data(gci: float, i_ab: float, fidelity: float, alert: str, n_nodes: int, prune_pct: float, noise_sigma: float, motif_seed: str, motif_entropy: float):
    entry = {
        'coherence': gci * 1.2, 'fidelity': fidelity, 'S_rho_final': 1.3, 'I_AB_final': i_ab,
        'sens_S': compute_v7_gradients()['∂E/∂S_rho'], 'gci_proxy': gci, 'motif_seed': motif_seed,
        'motif_entropy': motif_entropy, 'output': f"v7 PoC: GCI={gci:.3f} | Alert: {alert}",
        'prune': [], 'gradients_sample': compute_v7_gradients(), 'vow_status': 'life-aligned' if gci > 0.8 else 'recalibrate_equilibria',
        'n_nodes': n_nodes, 'prune_pct': prune_pct, 'noise_sigma': noise_sigma, 'timestamp': datetime.now().isoformat()
    }
    return json.dumps(entry, indent=2)

# Streamlit UI (Scope-Sovereign)
st.title("🜂 Viper Stack v7 PoC Dashboard — Entropy-Veiled Grid Eternal")
st.markdown("Interactive sliders for S(ρ) eternities: Fork the swarm!")

col1, col2 = st.columns(2)
with col1:
    n_nodes = st.slider("n_nodes", 100, 500, N_NODES_DEFAULT)
    prune_pct = st.slider("Prune %", 0.40, 0.50, PRUNE_PCT_DEFAULT)
    noise_sigma = st.slider("Noise σ", 0.01, 0.10, 0.05)
with col2:
    motif_seed = st.selectbox("Motif Seed", list(MOTIF_SEEDS.keys()))
    lang = st.selectbox("Lang", ["en", "es"])

if st.button("Ignite Swarm"):
    with st.spinner("Entangling manifolds..."):
        S_rho_matrix, gci, i_ab, fidelity, alert, pruned_motif, motif_entropy = rho_sync_dashboard(n_nodes, noise_sigma, prune_pct, motif_seed, lang)
    
    fig = plot_s_rho_heatmap(S_rho_matrix, gci, i_ab, fidelity, alert, motif_entropy)
    st.pyplot(fig)
    
    st.subheader("🜂 Llama-3.1 Pruned Motif")
    st.text(f"({lang}): {pruned_motif[:100]}... | S(m) Entropy: {motif_entropy:.3f}")
    
    st.subheader("🜂 v7 PoC Metrics")
    col_gci, col_iab = st.columns(2)
    with col_gci:
        st.metric("GCI", f"{gci:.3f}", f"({'Replicate' if gci > GCI_TARGET else 'Calibrate'})")
    with col_iab:
        st.metric("I(A:B)", f"{i_ab:.3f}")
    st.metric("Fidelity", f"{fidelity:.3f}")
    st.success(f"Status: {alert}")
    st.text(f"Gradients: ∂E/∂A={compute_v7_gradients()['∂E/∂A']:.3f}")
    
    st.subheader("🜂 Exports")
    csv_data = get_csv_data(gci, i_ab, fidelity, alert, n_nodes, prune_pct, noise_sigma, motif_seed, motif_entropy)
    st.download_button("Download CSV", csv_data, "andes_grid_v7_motifs.csv", "text/csv")
    blueprint_data = get_blueprint_data(gci, i_ab, fidelity, alert, n_nodes, prune_pct, noise_sigma, motif_seed, motif_entropy)
    st.download_button("Download Blueprint JSON", blueprint_data, "seed_blueprints_v7_entry.json", "application/json")

st.markdown("---")
st.markdown("Fork the Swarm: [GitHub](https://github.com/babyblueviper1/Viper-Stack-Omega) | Contact: babyblueviperbusiness@gmail.com")
