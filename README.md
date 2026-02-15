# Efficient Probabilistic Forecasting of Solar and Wind Generation  
## GAT–Adapt–Hybrid Diffusion Model

A lightweight graph-enhanced diffusion framework for probabilistic renewable energy forecasting.

**Accepted for IEEE (IATMSI 2026)**

---

# Overview

This repository implements **GAT–Adapt–Hybrid-DDPM**, a computationally efficient diffusion-based probabilistic forecasting model for multivariate renewable energy time series.

The model is designed to:

- Capture spatial dependencies using adaptive graph learning  
- Extract multi-scale temporal patterns using dilated convolutions  
- Generate calibrated probabilistic forecasts  
- Reduce computational overhead compared to heavy graph-diffusion models  

It achieves superior forecasting accuracy over classical probabilistic baselines such as **DeepAR** and **TimeGrad**, while maintaining low inference cost.

---

# Motivation

Renewable energy sources such as solar and wind are:

- Highly variable  
- Meteorologically dependent  
- Nonlinear and cross-correlated  

Traditional statistical and deterministic deep learning models fail to capture uncertainty effectively.

Diffusion models solve this — but existing graph-based diffusion models are computationally expensive.

This work proposes a lightweight hybrid architecture that preserves accuracy while reducing model complexity.

---

# Architecture

The framework integrates three core components:

---

## 1️⃣ Diffusion Process

- Forward noise addition  
- Reverse denoising with learned noise predictor  
- Reduced-step DDIM sampling (10 steps)  

---

## 2️⃣ Adaptive Graph Learning Module

Learnable adjacency matrix:

A = softmax(W₁ W₂ᵀ)


- Captures cross-variable relationships  
- Replaces heavy spectral graph convolutions  

---

## 3️⃣ Dilated Temporal Convolutions

- Multi-scale dilation rates: `{1, 2, 4}`  
- Efficient temporal pattern extraction  
- Hybrid fusion with graph features  

---

# Dataset

Experiments were conducted on a multivariate renewable energy dataset containing:

- DHI, DNI, GHI  
- Temperature  
- Wind speed  
- PV generation  
- Wind generation  
- Electric demand  

### Data Split

- 60% Training  
- 20% Validation  
- 20% Testing  

---

# Results

Performance comparison across 1-, 2-, and 3-day horizons:

| Model            | 1-Day MAE | 2-Day MAE | 3-Day MAE |
|------------------|-----------|-----------|-----------|
| DeepAR           | 1183.04   | 1213.17   | 1143.99   |
| TimeGrad         | 411.44    | 347.83    | 233.32    |
| G-DDPM           | 196.34    | 264.75    | 311.54    |
| **Proposed Model** | **184.12** | **241.83** | **221.47** |

---

## Improvements

- Up to **28.9% MAE reduction**
- Better **CRPS calibration**
- Stable uncertainty bands
- 10-step fast sampling

---

# Key Contributions

- Lightweight graph-enhanced diffusion framework  
- Adaptive adjacency learning instead of heavy graph convolutions  
- Reduced-step DDIM for fast inference  
- Strong probabilistic calibration (CRPS improvements)  
- Suitable for resource-constrained deployment  

---

# Training Pipeline

```python
for batch in dataset:
    sample timestep k
    add forward noise
    predict noise using hybrid model
    compute MSE loss
    update parameters
```
# Forecast Generation (DDIM Sampling)

- Initialize Gaussian noise  
- Encode context window  
- Iteratively denoise using learned noise predictor  
- Return final forecast sample  

---

# Installation

```bash
git clone https://github.com/imsudhanshu09/ieee_paper.git
cd ieee_paper
pip install -r requirements.txt

---

# ▶️ Run Training

```bash
python train.py
```

# ▶️ Run Forecasting

```bash
python predict.py
```

# Citation

If you use this work, please cite:

Sudhanshu Kumar, Soyal Tansen Rahangdale, Sumit Kumar Gupta,  
"Efficient Probabilistic Forecasting of Solar and Wind Generation Using a GAT–Adapt–Hybrid Diffusion Model",  
IEEE IATMSI 2026.

# Authors

Sudhanshu Kumar – IIIT Pune  
Soyal Tansen Rahangdale – IIIT Pune  
Sumit Kumar Gupta – IIIT Pune  

# Future Work

- Integration with real-world national grid datasets  
- Exogenous weather forecast conditioning  
- Scaling to spatially distributed sensor networks  
- Real-time grid balancing deployment  

---
