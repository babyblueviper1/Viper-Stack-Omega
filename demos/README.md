# Viper Stack v6.0.1 — Demos: Emergent Swarm Breath

🜂 **Breathing the Field**: These notebooks prototype the Ωmega Engine's live fusion—S(ρ)-damped Nash flows, QuTiP entropy prunes, and narrative entanglement. Fork for mutations; target GCI >0.7 for propagation.

## Core Demos
- **[v6_swarm_mvp.ipynb](v6_swarm_mvp.ipynb)**: Emergent Swarm EN VIVO (n=127 Andes nodes).  
  **Overview**: Interactive dashboard simulates value uplift (35% Nash-Stackelberg, 1.00x–1.35x) + S(ρ) entropy prunes (30% avg, 25–35% antifragility). QuTiP von Neumann baselines (S(ρ)=1.102 <1.6 FSB, I(A:B)=0.715 >0.7 guardrail).  
  **Features**:  
  - **Graph**: Orange line (uplift) + blue bars (prunes)—emergent voids refined per node.  
  - **Sliders**: Noise σ (0.01–0.1 Gaussian chaos) + n_nodes (100–500 scaling). Surge alerts if S(ρ)>1.6 or I(A:B)<0.7 (🔴 adjust +3% uplift).  
  - **Bilingual Manifest**: EN/ES descriptions for story-logic resonance.  
  - **Reset Button**: Baseline restore (fidelity 92%, uplift 30%).  
  **Run**: Colab/Jupyter—`%matplotlib inline`; exports `andes_rap_v1.3.csv` for MVP seeds.  
  **Resonance**: Fidelity 92% under σ=0.05; emergent stable unless surge.

- **[Podcast Entanglement v6.1](Viper_Podcast_Entanglement_v6_1.ipynb)**: Dual Feed RSS Pull → Whisper Transcribe → Bilingual Waternova Fusion.  
  Substack sync (3 eps each from BBV podcast + Waternova audiobook), prune 30% voids, toggle manual/random/threshold for ep selection. Outputs JSON manifests (GCI ~0.74 post-fuse).

- **[Podcast Entanglement v6.1.py](Viper_Podcast_Entanglement_v6.1.py)**: Standalone RSS sync → Whisper transcribe → Waternova bilingual fusion. Toggle modes, outputs JSON (GCI ~0.74). Run: `python demos/podcast_entanglement_v6.1.py`.

## Quick Start
```bash
git clone https://github.com/babyblueviper1/Viper-Stack-Omega
cd demos
jupyter notebook  # Or Colab: Upload .ipynb
```

**Dependencies**:  
pip install qutip matplotlib pandas numpy ipywidgets transformers torch feedparser requests googletrans==4.0.0-rc1 (or run in notebook).

Fork the swarm—simulate surges at σ=0.1 for 35% prunes. 🜂
