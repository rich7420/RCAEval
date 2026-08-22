# Ch5 — GNN_KAN with SparseFuse kernel (systems evaluation)

**Claim under test (systems, not accuracy):** the *same* B-spline GNN_KAN, with its KAN
evaluators (KAN_edge / KAN_node / KAN_message) computed by a **naive** kernel vs the
**SparseFuse** fused kernel. Everything else (dual-graph, attention, residual, fusion,
ranking) is unchanged. SparseFuse only applies to the **cubic B-spline** config (local
support, active-K=4); the Chebyshev/RPK headline config cannot be swapped.

## Backends (`kan_backends.py`)
| backend | role | notes |
|---|---|---|
| `sparsefuse` | fused kernel (experiment) | `MatrixFusedKANLinear`, Triton fused basis+GEMM |
| `ekan` | naive control | `efficient_kan.KANLinear` — identical B-spline definition to SparseFuse (drop-in), the cleanest "same B-spline, naive vs fused" pair |
| `rca_bspline` | naive control | vendored Cox-de Boor from RCAEval `basis_functions.py` — ties to Ch4's own naive numbers |

All use cubic order (p=3), L = G+p basis functions, base SiLU-linear branch, LayerNorm before the KAN edge, no spline scaler (`enable_standalone_scale_spline=False`, matching SparseFuse §5.3 Table 7).

## Pieces
- `synth_graph.py` — Barabási–Albert directed DAG, |E|≈3|V| (avg out-deg ~3, aligns Ch4 3.2). |V|∈{100,250,500,1000} + larger.
- `gnn_kan_bench.py` — clean 3-layer width-64 GNN+KAN mirroring `OptimizedGNNKANEncoder` (LayerNorm→KAN, sym-norm adj message passing→KAN msg proc→0.3 residual), link-prediction loss (negative sampling, the RCA objective). No exception-swallowing, so OOM/latency/NaN are observable.
- `instrument.py` — CUDA-event latency, peak VRAM, OOM/error classification, git provenance.
- `run_bench.py` — one measured (backend, G, |V|, seed) cell + sweep + CLI; emits CSV.
- `modal_ch5.py` — Modal H100 dispatcher (torch 2.10 + triton 3.6 + sparsefuse + efficient-kan).
- `real_model_adapter.py` — swaps the **real** GNN_KAN's B-spline spline path to SparseFuse, for parity (H1) on real datasets.

## Running
```bash
# from repo root, using the modal venv:
.venv-modal/bin/modal run ch5/modal_ch5.py --task list
.venv-modal/bin/modal run ch5/modal_ch5.py --task sanity       # 3 backends, G=5, |V|=100
.venv-modal/bin/modal run ch5/modal_ch5.py --task efficiency    # H2: latency/VRAM vs G
.venv-modal/bin/modal run ch5/modal_ch5.py --task feasibility   # H3: OOM frontier (G × |V|)
# results download to ch5/results/*.csv
```
Local (CPU, naive only — torch 1.12 venv):
```bash
.venv310/bin/python -m ch5.run_bench --backends rca_bspline --grids 5 --nodes 100 --seeds 0 --device cpu --out /tmp/x.csv
```

## Run sets (≈60 runs)
- **Run set 2 — Efficiency (H2):** G∈{5,15,50,100} × |V|=500 × {ekan, sparsefuse} × 3 seeds. Metric: step latency + peak VRAM; expect SF/naive ratio to widen with G.
- **Run set 3 — Feasibility (H3):** G∈{5,15,50,100} × |V|∈{100,250,500,1000} × 2 backends. Metric: OOM frontier (largest (G,|V|) that trains). Expect SF frontier several cells larger.
- **Run set 1 — Parity (H1):** real datasets via `real_model_adapter.py` (naive vs swapped), G∈{5,15,50,100}, ≥3 seeds. Confirms the swap does not change accuracy. **Needs the parity image (below).**

## Open decisions (from the planning A-table)
- **A1.3** G=100 in the graph setting: covered — feasibility sweep includes G=100; sanity validates the path.
- **A1.4** torch.compile/checkpoint: harness runs eager; a compile arm can be added to `run_bench` (SparseFuse supports `torch.compile`).
- **A2.1/A2.2** |V| start ≈100, |E|≈3|V|: generator matches; confirm Ch4 final "64 services vs ~100 nodes" so both chapters agree.
- **A3.1** grid {5,15,50,100}: yes, the four points.
- **A3.3** parity seeds: ≥3 per cell (default seeds 0,1,2).
- **A3.4** parity naive baseline accuracy (B-spline on Train-Ticket ~0.016 in Ch4): parity is a *systems-equivalence* check, not an accuracy bar — acceptable, but flag.

## Parity image (real model, H1) — TODO
`real_model_adapter.py` imports the real `RCAEval.gnn_kan_module` model, which pulls
`torch_geometric` (pinned to a torch-1.12-era 2.0.4). The Modal H100 image uses torch 2.10,
so the parity task needs a `torch_geometric` build matching torch 2.10 and the RCA code
verified to import/run under torch 2.x. Not yet wired into `modal_ch5.py`.
