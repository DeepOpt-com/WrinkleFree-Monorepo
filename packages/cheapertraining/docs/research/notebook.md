## 12/19/2025

### GUM (GaLore Unbiased Muon) Optimizer

- [ ] __TODO__: Integrate GUM optimizer when official code is released, benchmark against Apollo

**Paper**: [arXiv:2510.17802](https://arxiv.org/abs/2510.17802) - "Unbiased Gradient Low-Rank Projection"
**Authors**: Rui Pan, Yang Luo, Yuxing Liu, Yang You, Tong Zhang (Oct 2025)

**Why it matters**: Apollo is currently our best optimizer (4x improvement over baseline in 1.58Quant benchmarks). GUM claims to match Muon's convergence while maintaining GaLore's memory efficiency, and even outperforms full-parameter training.

**Key innovation**: Layerwise sampling for debiasing low-rank gradient projection
- Combines GaLore (low-rank gradient projection) + Muon (Newton-Schulz iterations)
- Two update types: low-rank (scaled 1/(1-q)) and full-rank compensated (scaled 1/q)
- Probability q = gamma/N_L where gamma is number of full-rank update layers
- Theoretically unbiased: the biased low-rank term cancels out in expectation

**Algorithm summary**:
1. Compute SVD of gradient, take top-r left singular vectors as projection P
2. Sample layers for either low-rank or full-rank update
3. Low-rank: R = beta*R + (1/(1-q)) * P^T * G, then W += lr * P * NewtonSchulz(R)
4. Full-rank: R = beta*R + (1/q) * (G - P*P^T*G), then W += lr * NewtonSchulz(R)

**Hyperparameters** (from paper):
- Projection rank r: 128 (vs GaLore's 512)
- Momentum beta: 0.95
- Full-rank layers gamma: 2

**Status**: No official code repository yet (paper from Oct 2025). Watch:
- Author Rui Pan's GitHub: https://github.com/research4pan
- GaLore repo for updates: https://github.com/jiaweizzhao/GaLore

**Benchmark plan**: When code is available, run quick comparison on runpod (root@69.30.85.193:22066) against Apollo using same setup as 1.58Quant benchmarks.

---

## 12/18/2025
- [ ] __Idea__: right now we only calculate influence based on embedding and final layer but this seems rather not good for many cases where the "reasoning" is some highly structured internal representation. So, what if we just sample a random internal layer at each step