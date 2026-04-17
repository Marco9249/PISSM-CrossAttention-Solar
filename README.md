<div align="center">

<img src="https://img.shields.io/badge/%F0%9F%94%AD-Cross--Attention%20AI-8B5CF6?style=for-the-badge&labelColor=1a1a2e" alt="Cross-Attention AI"/>

# Physics-Informed Cross-Attention Networks (PISSM-CA)

### 🔬 *Bridging Atmospheric Physics and Deep Learning via Dual-Attention Solar Forecasting* 🔬

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-005288?style=for-the-badge&logo=nasa&logoColor=white)](https://power.larc.nasa.gov/)
[![Author](https://img.shields.io/badge/Author-Mohammed%20E.%20B.%20Abdullah-blue?style=for-the-badge)](https://github.com/Marco9249)

<br/>

<img src="https://img.shields.io/badge/Author-Mohammed%20Ezzeldin%20Babiker%20Abdullah-4A90D9?style=flat-square&logo=google-scholar&logoColor=white" alt="Author"/>

---

*"What if the model could learn WHERE to look in time — and WHAT physics to trust?"*

</div>

---

## 🎯 The Core Innovation

> Standard attention treats all features equally. **PISSM-CA** introduces a **Dual-Attention pipeline** — first learning *temporal* relevance via Self-Attention, then injecting *physical constraints* via Cross-Attention with Solar Zenith Angle (SZA) and Clearness Index (KT).

This creates a model that doesn't just learn patterns — it **learns physics-aware patterns**.

### 🏆 Key Contributions

| Contribution | Description |
|:------------:|:------------|
| 🧠 **Self → Cross Attention** | Two-stage attention: temporal focus first, then physics injection |
| 🌡️ **Physics as Keys/Values** | SZA and KT form the K/V pairs — forcing the model to attend through physics |
| 📐 **Hankel State-Space Embedding** | Koopman-linearized input via overlapping sub-windows |
| 🌙 **ReLU Safety Filter** | Structural non-negativity guarantee (no phantom nocturnal output) |
| ⚡ **Ultra-Lightweight** | < 35K parameters — edge-deployable on microcontrollers |

---

## 🏗️ Architecture (5-Layer Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ☀️  NASA POWER Input (15 Features, 24h Window)            │
│       9 Physical + 6 Cyclical (sin/cos encoded)            │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  PRE-LAYER: Hankel Embedding                  │          │
│  │  unfold(sub_window=5) → (batch, 20, 75)       │  Koopman │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LAYER 1: 1D-CNN (75→64 filters)             │          │
│  │  Conv1D + GELU + Dropout(0.2)                 │  Feature │
│  │  Output: (batch, 20, 64)                      │  Extrac. │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LAYER 2: Temporal Self-Attention 🎯          │          │
│  │  MultiheadAttention(embed=64, heads=4)        │  Q=K=V   │
│  │  Q = K = V = CNN output                       │  (learn  │
│  │  "Which time steps matter most?"              │  when)   │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LAYER 3: Physical Cross-Attention 🌡️         │          │
│  │  Q = Self-Attention output                    │          │
│  │  K = V = Linear(concat[SZA, KT]) → 64-dim    │  Physics │
│  │  "What does the physics say at each step?"    │  Inject. │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LAYER 4: FC Integrator                       │          │
│  │  h_last → Dense(64→128) → GELU → Drop → (1)  │  Predict │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LAYER 5: ReLU Safety Filter                  │  Night=0 │
│  │  max(0, prediction) — non-negativity          │  Always  │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│              📊 GHI Prediction (Wh/m²)                      │
│              Physically bounded, always                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🔬 Self-Attention vs Cross-Attention — Design Rationale

| Attention Type | Role | Q | K | V |
|:--------------:|:-----|:-:|:-:|:-:|
| **Self-Attention** (Layer 2) | Learn *temporal* importance | CNN out | CNN out | CNN out |
| **Cross-Attention** (Layer 3) | Inject *physics* constraints | Self-Attn out | SZA+KT proj. | SZA+KT proj. |

> 💡 **Why this order matters:** Self-Attention first learns *when* to focus. Cross-Attention then gates *what physics* to trust at each focused timestep. Reversing the order would dilute physics signals before temporal focusing.

---

## 📂 Repository Structure

```
📦 PISSM-CrossAttention-Solar/
│
├── 📁 training_code/
│   └── 🧠 pissm_cross_attention.py       # Full 5-layer architecture + training
│
├── 📁 evaluation_code/
│   └── 📊 pissm_multi_year_test.py        # Multi-year evaluation pipeline
│
├── 📁 training_data/
│   ├── 📊 Hourly_2010_2015.csv            # NASA POWER hourly (training)
│   └── 📊 Hourly_2020_2025.csv            # NASA POWER hourly (testing)
│
├── 📄 PISSM_CrossAttention_Paper.docx     # Research paper
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone https://github.com/Marco9249/PISSM-CrossAttention-Solar.git
cd PISSM-CrossAttention-Solar
pip install -r requirements.txt

# Train the model (uses Hourly_2010_2015.csv)
python training_code/pissm_cross_attention.py

# Multi-year evaluation (2020-2025)
python evaluation_code/pissm_multi_year_test.py

# Outputs:
#   → pissm_saved_weights.pth (model checkpoint)
#   → pissm_forecast_results.png
#   → pissm_training_curves.png
```

---

## 🧪 Technical Specifications

| Parameter | Value |
|:---------:|:-----:|
| 🪟 **Input Window** | 24 hours |
| 📐 **Hankel Sub-Window** | 5 steps → 20 windows |
| 🧮 **Total Parameters** | < 35,000 |
| 🎯 **Self-Attention** | 4 heads × 64 dim |
| 🌡️ **Cross-Attention** | 4 heads × 64 dim (SZA + KT projected) |
| 📏 **FC Hidden** | 128 units (GELU) |
| 🔄 **Optimizer** | Adam (lr=1e-3, weight_decay=1e-5) |
| 📉 **Loss** | MSE + ReduceLROnPlateau |
| 🌍 **Data** | NASA POWER Hourly, Khartoum, Sudan |

---

## 📚 Related Research Papers

<div align="center">

| # | Paper | Repository | arXiv |
|:-:|:------|:----------:|:-----:|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) | [![arXiv](https://img.shields.io/badge/-2604.13455-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13455) |
| 2 | Physics-Informed State Space Model (PISSM) | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PISSM-Solar-Forecasting) | [![arXiv](https://img.shields.io/badge/-2604.11807-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11807) |
| **3** | **PISSM Cross-Attention** *(this repo)* 🌟 | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PISSM-CrossAttention-Solar) | *Coming Soon* |
| 4 | Thermodynamic Liquid Manifold Networks | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) | [![arXiv](https://img.shields.io/badge/-2604.11909-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.11909) |
| 5 | Asymmetric-Loss Industrial RUL Prediction | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) | [![arXiv](https://img.shields.io/badge/-2604.13459-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.13459) |
| 🎮 | Interactive 3D Architecture Visualization | [![Repo](https://img.shields.io/badge/-Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-Hybrid-3D-Viz) | — |

</div>

---

## 📖 Citation

```bibtex
@misc{abdullah2026pissmca,
  title   = {Physics-Guided Cross-Attention Networks for Reliable Solar
             Irradiance Forecasting in Off-Grid Systems},
  author  = {Mohammed Ezzeldin Babiker Abdullah},
  year    = {2026}
}
```

> **APA 7th Edition:**
> Abdullah, M. E. B. (2026). *Physics-Guided Cross-Attention Networks for Reliable Solar Irradiance Forecasting in Off-Grid Systems*. Manuscript in preparation.

---

<div align="center">

### 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**
*Researcher in Physics-Informed Deep Learning & Renewable Energy Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-181717?style=for-the-badge&logo=github)](https://github.com/Marco9249)

---

© 2026 Mohammed Ezzeldin Babiker Abdullah — All rights reserved.

</div>
