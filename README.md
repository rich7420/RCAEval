# GNN\_KAN for Root Cause Analysis in Microservices

This branch (`gnn-kan`) contains the code, experiments, and paper sources for
**"GNN\_KAN: Diagnosing Non-Linear Fault Propagation in Scale-Free Microservices via
Kolmogorov-Arnold Networks"**, built on top of the
[RCAEval](https://github.com/phamquiluan/RCAEval) benchmark.

## Research Overview

- **Problem**: Modern cloud-native systems decompose applications into many microservices. Failures propagate along complex, often scale-free service graphs, making it hard to quickly localize the true root cause (e.g., the specific service and resource that first failed).
- **Limitation of existing methods**:
  - Statistical and random-walk based RCA methods (e.g., PageRank-style or change-point detection) assume mostly linear fault propagation and struggle with non-linear resource saturation (such as memory leaks or queue buildup).
  - Standard GNNs and GATs use scalar attention and MLPs; on scale-free graphs their attention tends to collapse onto hub nodes (like API gateways), and ReLU-based MLPs do not naturally encode soft resource limits or saturation behaviors.
- **Key idea of GNN\_KAN**:
  - Model RCA as a learning-to-rank problem on dynamic service graphs, where each node's score reflects how likely it is to be the root cause.
  - Replace scalar attention and MLP transformations with **Kolmogorov–Arnold Networks (KANs)** on graph edges and nodes. KANs decompose multivariate mappings into interpretable one-dimensional basis functions that can explicitly encode **resource constraints and saturation effects**.
  - Introduce a **Rotation-based Periodic Kernel (RPK)** as a bounded, tensor-product basis tailored to capture non-linear resource coupling (e.g., high memory + high latency) and cyclic contention patterns, while keeping computation linear in the number of edges O(|E|).
- **Dual-path, dual-basis design**:
  - Two complementary graphs — a **propagation graph** (service dependencies) and a **similarity graph** (similar anomaly patterns) — coupled layer-wise in a dual-path GNN.
  - **RPK** and Chebyshev-based KANs are combined in an orthogonal frequency-filtering ensemble: RPK focuses on low-frequency, cumulative faults (like memory leaks), while Chebyshev captures high-frequency, transient faults (like network spikes). A max operator over the two scores per node acts as a non-parametric selector.

## What Is in This Branch

| Path | Purpose |
|---|---|
| `RCAEval/gnn_kan_module/` | The GNN\_KAN model: config, KAN components (`kan_components/`), encoders, training loop, feature processing. Also contains GAT / GATv2 / GraphTransformer baselines. |
| `RCAEval/e2e/gnnkan.py` | End-to-end `gnn_kan_rca` entry point used by the benchmark harness. |
| `main.py` | CLI to run any RCA method (including `gnn_kan`) on a dataset. |
| `experiment.py`, `experiment_table2.py`, `run_experiments.py` | Batch experiment drivers for the paper tables. |
| `output/`, `目前結果.csv` | Experiment results (per-case JSONs, aggregated CSVs, logs). |
| `ch5/` | Systems evaluation of the SparseFuse fused Triton B-spline kernel inside GNN\_KAN (see below). |
| `paper_draft_twsc.tex`, `PAPER/` | Paper sources and drafts. |
| `DOWNLOAD_DATASETS.md`, `download_all_datasets.py` | Dataset download guide and scripts. |

## Installation

Python 3.10 is recommended (`.tool-versions` pins 3.10.13).

```bash
python -m venv .venv310
source .venv310/bin/activate
pip install -e ".[default]"        # installs requirements.txt
```

## Datasets

Datasets (Online Boutique, Sock Shop 1/2, Train Ticket, RE1–RE3) are downloaded
automatically on first run, or in bulk via:

```bash
python download_all_datasets.py
```

See `DOWNLOAD_DATASETS.md` for details. Dataset ids accepted by `main.py`:
`ob`, `ss1`, `ss2`, `tt`, `re1`, `re2-ob`, `re2-tt`, `re3`, plus multi-source
variants `mm-ob`, `mm-ss1`, `mm-ss2`, `mm-tt`.

## Quick Start

Run GNN\_KAN on Online Boutique:

```bash
python main.py --method gnn_kan --dataset ob
```

Useful flags: `--test` (small run), `--fast_mode` (faster GNN-KAN settings),
`--iter_num N`, `--length N`. Run `python main.py --method x --dataset ob` to
print the full list of available methods (BARO, CIRCA, RCD, PC/GES/LiNGAM +
PageRank, GAT, GATv2, GNN, GraphTransformer, ...).

Reproduce the paper tables:

```bash
python experiment_table2.py          # Table 2 — results land in output/
./run_gnn_kan_comparison.sh          # GNN_KAN vs baselines comparison
```

## Ch5: SparseFuse Fused KAN Kernel (Systems Evaluation)

`ch5/` benchmarks the *same* B-spline GNN\_KAN with its KAN evaluators computed by a
naive kernel vs the **SparseFuse** fused Triton kernel (latency, peak VRAM, OOM
frontier, and accuracy parity). The real model can opt in via
`config.kernel = 'sparsefuse'` (default `'naive'`); only the cubic-B-spline spline-term
compute is swapped, everything else is unchanged. Benchmarks run on Modal H100s.
See `ch5/README.md` for the run matrix and results (`ch5/results/`).

## Main Findings (Qualitative)

- On complex, deep topologies (e.g., the Train-Ticket benchmark with 50+ services), GNN\_KAN improves Precision@1 and Top-5 recall over BARO, GAT, and MLP-based GNNs, narrowing the operator's search space from all services to a small candidate set.
- Learned KAN edge functions exhibit clear **saturation zones** and **zero-gradient regions** that align with physical resource limits (e.g., turning "on" only when memory exceeds ~85%), providing interpretable evidence for RCA instead of opaque scalar weights.
- Attention collapse in GAT is mitigated because GNN\_KAN learns **functional shapes on edges**, decoupling topological hubness from causal importance.

For implementation details, experimental setups, and full results, see
`paper_draft_twsc.tex` and `PAPER/`.

## Acknowledgments

This work builds on [RCAEval](https://github.com/phamquiluan/RCAEval)
(MIT License — see `LICENSE` and `LICENSES/`).
