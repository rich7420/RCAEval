"""
Definitive check that backend='sparsefuse' actually executes the Triton FUSED kernel on
this GPU, and never silently falls back to the pure-PyTorch reference path.

SparseFuse's MatrixFusedKANFunction.forward uses Triton only when x.is_cuda and
B*I >= 32, and a try/except silently falls back to forward_reference() on any Triton
error. So "backend=sparsefuse" alone does not prove the fused path ran. This script:

  1. counts calls to the Triton basis kernel (triton_compute_basis_sparse) vs the
     reference evaluator (BSplineEvaluator.forward_reference) during a real GNN+KAN
     forward+backward, and asserts triton>0 and reference==0;
  2. reads sparsefuse.function._last_forward_triton (must be True);
  3. checks fused output ~= reference output (numerical equivalence), proving the fused
     kernel is both ACTIVE and CORRECT.

Run inside the Modal H100 container (PYTHONPATH=/workspace).
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    assert torch.cuda.is_available(), "no CUDA device"
    import triton  # noqa: F401
    import sparsefuse.function as sf_fn
    import sparsefuse.kernels.bspline.basis_deboor_sparse as bd
    from sparsefuse.basis.base import BasisRegistry

    from ch5.gnn_kan_bench import (BenchConfig, GNNKANBench, build_norm_adj,
                                   link_pred_loss)
    from ch5.synth_graph import make_synth_graph

    print(f"torch={torch.__version__}  triton={triton.__version__}  "
          f"gpu={torch.cuda.get_device_name(0)}")

    # --- instrument: count Triton-basis vs reference-evaluator calls ---
    counters = {"triton_basis": 0, "reference": 0}

    orig_triton = bd.triton_compute_basis_sparse
    def counted_triton(*a, **k):
        counters["triton_basis"] += 1
        return orig_triton(*a, **k)
    bd.triton_compute_basis_sparse = counted_triton  # picked up by in-forward import

    BSplineEval = BasisRegistry.get("bspline")
    orig_ref = BSplineEval.forward_reference
    def counted_ref(self, *a, **k):
        counters["reference"] += 1
        return orig_ref(self, *a, **k)
    BSplineEval.forward_reference = counted_ref

    sf_fn._TRACK_TRITON_PATH = True

    dev = torch.device("cuda")
    g = make_synth_graph(2000, feat_dim=64, m=3, seed=0, device=dev)
    cfg = BenchConfig(backend="sparsefuse", feat_dim=64, hidden=512,
                      num_layers=3, grid_size=50)
    model = GNNKANBench(cfg).to(dev)
    adj = build_norm_adj(g.edge_index, g.num_nodes, dev)

    # count KAN edges that should each hit the Triton basis kernel
    from sparsefuse.layers import MatrixFusedKANLinear
    n_kan = sum(isinstance(m, MatrixFusedKANLinear) for m in model.modules())

    loss = link_pred_loss(model(g.x, adj), g.edge_index)
    loss.backward()
    torch.cuda.synchronize()

    print(f"\nKAN edges in model              : {n_kan}")
    print(f"Triton basis-kernel calls (fwd) : {counters['triton_basis']}")
    print(f"reference-path calls (fwd)      : {counters['reference']}")
    print(f"_last_forward_triton            : {sf_fn._last_forward_triton}")

    ok_triton = counters["triton_basis"] >= n_kan and counters["reference"] == 0
    ok_flag = sf_fn._last_forward_triton is True

    # --- numerical equivalence: fused vs forced reference, identical weights ---
    bd.triton_compute_basis_sparse = orig_triton  # restore real triton
    torch.manual_seed(0)
    layer = MatrixFusedKANLinear(256, 256, grid_size=50, spline_order=3,
                                 enable_standalone_scale_spline=False,
                                 include_base_branch=False).to(dev)
    x = torch.randn(1024, 256, device=dev)
    y_fused = layer(x)
    # force reference by temporarily breaking the triton import path
    import sparsefuse.kernels.bspline.basis_deboor_sparse as bd2
    def boom(*a, **k):
        raise RuntimeError("force-reference")
    bd2.triton_compute_basis_sparse = boom
    layer._w_cache = None  # invalidate cache so forward recomputes
    y_ref = layer(x)
    bd2.triton_compute_basis_sparse = orig_triton
    max_abs = (y_fused - y_ref).abs().max().item()
    rel = max_abs / (y_ref.abs().max().item() + 1e-9)
    print(f"\nfused vs reference  max|Δ|={max_abs:.3e}  rel={rel:.3e}")
    ok_num = rel < 1e-3

    print("\n=== VERDICT ===")
    print(f"  fused Triton path active   : {'PASS' if ok_triton else 'FAIL'}")
    print(f"  _last_forward_triton==True : {'PASS' if ok_flag else 'FAIL'}")
    print(f"  fused == reference (num)   : {'PASS' if ok_num else 'FAIL'}")
    allok = ok_triton and ok_flag and ok_num
    print(f"  OVERALL                    : {'PASS — real fused kernel' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
