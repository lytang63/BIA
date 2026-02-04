
<!-- =========================
README.md (Modern OSS Style)
Project: Bures-Isotropy Alignment (BIA)
ICLR 2026 (Camera Ready)
========================= -->

<div align="center">

# BIA: Bures-Isotropy Alignment for Generalized Category Discovery

**Restore token geometry, prevent dimensional collapse, boost GCD baselines — plug-and-play in a few lines.**

<!-- Badges (replace links once you have them) -->
[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-PDF-green)](#)
[![Code](https://img.shields.io/badge/Code-PyTorch-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-informational)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#)

<!-- ====== Put an eye-catching teaser figure/gif here ====== -->
<!-- ✅ 放图：Paper Figure 1（动机对比：Conventional GCD collapse vs BIA geometry restore） -->
<p align="center">
  <img src="assets/fig1_motivation.png" width="88%" alt="BIA Motivation (Figure 1)"/>
</p>

[📄 Paper](#) · [🧪 Results (Known-K)](#known-k-results) · [⚙️ Installation](#installation) · [🚀 Quickstart](#quickstart) · [🧩 Integrations](#integrate-bia-into-baselines) · [📌 BibTeX](#citation)

</div>

---

## 🔥 News
- **[2026]** BIA accepted to **ICLR 2026**.
- Code release & reproducibility scripts are under active maintenance.

---

## What is BIA?

Generalized Category Discovery (GCD) clusters unlabeled data containing both **known (old)** and **novel (new)** classes. Many methods focus on **compactness** and may over-compress representations, leading to **anisotropic, low-rank token manifolds** and unstable discovery.

**BIA** restores the geometry by aligning the **class-token covariance** toward an **isotropic prior** via the **Bures distance**, which (under mild constraints) admits a simple surrogate:
> maximize the **nuclear norm** of stacked class tokens (i.e., encourage full-rank, spectrum-uniform representations).

### ✅ Key properties
- **Plug-and-play:** add one loss term, no architecture change.
- **General:** works across contrastive / prototype / prompt-based GCD pipelines.
- **Lightweight:** batch-level Gram matrix + SVD/eig on a B×B matrix.

---

## Method Overview

<!-- ✅ 放图：Paper Figure 2（整体框架：BIA 插到任意 GCD framework） -->
<p align="center">
  <img src="assets/fig2_overview.png" width="92%" alt="BIA Overview (Figure 2)"/>
</p>

### Objective (high-level)

Let `Z ∈ R^{B×d}` be stacked **[cls] tokens** of an unlabeled mini-batch.

**Training loss:**
- `L = L_GCD + λ · L_BIA`

**BIA option A (metric form):** Bures distance to identity  
**BIA option B (surrogate):** `L_BIA = - ||Z||_*` (nuclear norm)

> In practice, the surrogate is simple, stable, and only a few lines.

---

## Why it works (intuitions you can cite in talks)

- **Spectrum homogenization:** pushes eigenvalues toward a more uniform distribution.
- **Higher manifold capacity:** reduces dimensional collapse.
- **More reliable discovery:** better separation for novel classes while keeping known classes compact.

<!-- ✅ 放图：Paper Figure 3（VNE / rank proxy 对比） -->
<!-- ✅ 放图：Paper Figure 6（eigenvalue distribution & accuracy correlation / dimensional collapse mitigation） -->
<p align="center">
  <img src="assets/fig3_vne_rank.png" width="49%" alt="VNE vs Rank (Figure 3)"/>
  <img src="assets/fig6_dimcollapse.png" width="49%" alt="Dimensional Collapse (Figure 6)"/>
</p>

---

## Installation

```bash
git clone https://github.com/lytang63/BIA.git
cd BIA

conda create -n bia python=3.10 -y
conda activate bia

pip install -r requirements.txt
````

> If you use your own baseline repos (CMS / SimGCD / SelEx / SPTNet), you can also install as a minimal module and drop-in the loss.

---

## Quickstart

### 1) Train (example)

```bash
# Example flags (replace with your actual scripts/args)
python train.py \
  --dataset CUB \
  --baseline simgcd \
  --known_k 1 \
  --bia 1 \
  --bia_lambda 0.004
```

### 2) Evaluate (Known-K setting)

```bash
python eval.py \
  --dataset CUB \
  --baseline simgcd \
  --eval_known_k 1
```

---

## Integrate BIA into baselines

BIA is designed to be **architecture-agnostic**. You only need batch cls tokens `Z`.

```python
# Pseudocode (minimal)
# Z: (B, d) normalized class-token features from unlabeled batch
# bia_lambda: scalar
loss = loss_gcd

# nuclear-norm surrogate
loss_bia = - torch.linalg.svdvals(Z).sum()
loss = loss + bia_lambda * loss_bia
```

### Supported baselines (highlight)

* **SimGCD**
* **CMS**
* **SelEx**
* **SPTNet**

> We recommend reporting **Known-K** results prominently for fair apples-to-apples comparisons.

---

<a name="known-k-results"></a>

## 🧪 Results (Known-K Setting)

We **mainly present** results under **ground-truth K given** (Known-K) as the primary setting.

### Table 1 — Fine-grained benchmarks (Known-K)

<!-- ✅ 表格只保留 Table 1 + Table 2；这里就是 Table 1 -->

<!-- ✅ 只突出 CMS / SimGCD / SelEx / SPTNet（符合你的“主推 known K + 四个 baseline”要求） -->

| Method           |  CUB All |  CUB Old |  CUB New | Cars All | Cars Old | Cars New | Aircraft All | Aircraft Old | Aircraft New |
| ---------------- | -------: | -------: | -------: | -------: | -------: | -------: | -----------: | -----------: | -----------: |
| SelEx            |     78.7 |     81.3 |     77.5 |     55.9 |     76.9 |     45.8 |         60.8 |         70.3 |         56.2 |
| **SelEx + BIA**  | **80.6** |     81.0 | **80.4** | **57.0** | **77.3** | **47.2** |     **61.8** |         68.2 |     **59.2** |
| SimGCD           |     60.7 |     65.6 |     57.7 |     51.2 |     69.4 |     42.4 |         54.0 |         58.8 |         51.5 |
| **SimGCD + BIA** | **62.1** | **65.8** | **60.3** | **52.3** | **70.0** | **43.7** |     **55.1** |     **58.9** |     **53.1** |
| CMS†             |     67.1 |     74.9 |     63.2 |     56.7 |     76.8 |     37.5 |         53.6 |         60.3 |         47.0 |
| **CMS† + BIA**   | **71.1** |     74.1 | **66.9** | **57.4** | **79.4** |     36.2 |     **55.7** |     **63.7** |     **47.9** |
| SPTNet           |     62.0 |     69.2 |     56.0 |     56.2 |     70.3 |     46.6 |         51.6 |         60.7 |         45.9 |
| **SPTNet + BIA** | **63.3** | **70.7** | **59.6** | **58.8** | **75.4** | **50.8** |     **54.7** |     **65.3** |     **48.5** |

> **Tip (presentation):** you can add a small “Δ row” or color highlights in the paper; README usually keeps it clean.

---

### Table 2 — Coarse/fine-grained benchmarks (Known-K)

| Method           | CIFAR100 All | CIFAR100 Old | CIFAR100 New | ImageNet100 All | ImageNet100 Old | ImageNet100 New | Herbarium19 All | Herbarium19 Old | Herbarium19 New |
| ---------------- | -----------: | -----------: | -----------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| SelEx            |         80.0 |         84.8 |         70.4 |            82.3 |            93.9 |            76.5 |            36.2 |            46.0 |            30.9 |
| **SelEx + BIA**  |     **80.7** |         84.3 |     **72.1** |        **82.8** |        **94.1** |        **77.8** |        **36.8** |        **47.5** |        **31.0** |
| SimGCD           |         80.1 |         81.5 |         77.2 |            83.3 |            92.1 |            78.9 |            44.7 |            57.4 |            37.9 |
| **SimGCD + BIA** |     **80.2** |     **81.5** |     **77.5** |        **86.7** |        **93.1** |        **83.6** |        **45.6** |        **57.8** |        **39.0** |
| CMS†             |         79.5 |         85.4 |         67.7 |            83.0 |            95.6 |            76.6 |            36.5 |            55.4 |            26.4 |
| **CMS† + BIA**   |         79.0 |     **85.5** |         66.1 |        **84.8** |        **95.6** |        **79.5** |            36.3 |        **56.5** |            25.4 |
| SPTNet           |         81.3 |         84.3 |         75.6 |            85.4 |            93.2 |            81.4 |            43.4 |            58.7 |            35.2 |
| **SPTNet + BIA** |     **82.1** |     **84.8** |     **76.2** |        **85.4** |        **93.4** |            81.3 |        **44.2** |        **58.9** |        **36.3** |

---

## Reproducibility

* We recommend providing:

  * a single script per dataset (`scripts/cub_knownk.sh`, `scripts/cifar100_knownk.sh`, …)
  * configs for each baseline (`configs/simgcd/*.yaml`, `configs/cms/*.yaml`, …)
  * fixed seeds & log exports

---

## FAQ

**Q: Does BIA require extra labels?**
A: No — it operates on unlabeled batches (class tokens).

**Q: Does it change model architecture?**
A: No — only adds a batch-level regularization term.

---

## Citation

If you find this repo useful, please cite:

```bibtex
@inproceedings{bia_iclr2026,
  title     = {Bures-Isotropy Alignment: Manifold Learning in Generalized Category Discovery},
  author    = {***},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## Acknowledgements

This repo builds upon strong GCD baselines including **SimGCD**, **CMS**, **SelEx**, and **SPTNet**.

---

## Contact

* Issues and PRs are welcome!
* For questions: open a GitHub issue with logs + config.
 的小徽章）。

