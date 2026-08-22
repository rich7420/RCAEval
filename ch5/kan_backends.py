"""
B0.1 + B0.2 — Switchable KAN-linear backends for the Ch5 GNN+KAN harness.

A single primitive `make_kan_linear(backend, in_features, out_features, grid_size, ...)`
returns an nn.Module mapping [*, in_features] -> [*, out_features]. Three backends, all
B-spline KAN edges of order 3 with L = grid_size + spline_order basis functions and a
base (SiLU-linear) branch, so they are parameter-comparable for a fair parity test:

  - "sparsefuse"  : sparsefuse.layers.MatrixFusedKANLinear   (fused basis+matmul kernel)
  - "ekan"        : efficient_kan.KANLinear                  (canonical naive baseline)
  - "rca_bspline" : vendored Cox-de Boor basis + einsum      (the thesis's own naive path,
                    faithful to RCAEval/gnn_kan_module/kan_components/basis_functions.py,
                    for tying Ch5 numbers back to Ch4)

The fusion benefit lives at the *layer* level (basis eval + coefficient matmul), so we
swap whole layers, not just the basis tensor — swapping only the basis would not exercise
SparseFuse's fused GEMM.

All backends expect to receive LayerNorm'd input (the GNN harness applies LayerNorm before
each KAN edge); MatrixFusedKANLinear documents this requirement and we apply it uniformly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKENDS = ("sparsefuse", "ekan", "rca_bspline")


# --------------------------------------------------------------------------------------
# rca_bspline: vendored faithful copy of RCAEval's Cox-de Boor B-spline basis.
# Source: RCAEval/gnn_kan_module/kan_components/basis_functions.py::BSplineBasis
# Difference: here num_basis = grid_size + spline_order (use ALL basis funcs, so G actually
# scales representational resolution), and we add a base SiLU-linear branch to match the
# sparsefuse/ekan layer structure.
# --------------------------------------------------------------------------------------
class _CoxDeBoorBasis(nn.Module):
    def __init__(self, num_basis: int, spline_order: int = 3, grid_size: int = 8):
        super().__init__()
        self.num_basis = num_basis
        self.spline_order = spline_order
        self.grid_size = grid_size
        knots = torch.linspace(-1.0, 1.0, grid_size + 2 * spline_order + 1)
        self.register_buffer("knot_vector", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] (already LayerNorm'd); map into [-1, 1] like the RCAEval source.
        x_normalized = torch.clamp(x, min=-2.0, max=2.0) / 2.0
        knots = self.knot_vector
        M = knots.shape[0]
        p = self.spline_order
        x_exp = x_normalized.unsqueeze(-1)
        t_i = knots[:-1].view(1, 1, -1)
        t_ip1 = knots[1:].view(1, 1, -1)
        B_prev = ((x_exp >= t_i) & (x_exp < t_ip1)).to(x.dtype)
        for deg in range(1, p + 1):
            denom1 = (knots[deg:M - 1] - knots[:M - 1 - deg]).view(1, 1, -1)
            denom2 = (knots[deg + 1:M] - knots[1:M - deg]).view(1, 1, -1)
            left = B_prev[..., :B_prev.shape[-1] - 1]
            right = B_prev[..., 1:]
            x_left = x_exp[..., :left.shape[-1]]
            x_right = x_exp[..., :right.shape[-1]]
            num1 = x_left - knots[:M - 1 - deg].view(1, 1, -1)
            a = torch.where(denom1.abs() > 1e-8, num1 / denom1, torch.zeros_like(num1))
            num2 = knots[deg + 1:M].view(1, 1, -1) - x_right
            b = torch.where(denom2.abs() > 1e-8, num2 / denom2, torch.zeros_like(num2))
            B_prev = a * left + b * right
        L = B_prev.shape[-1]
        if self.num_basis <= L:
            return B_prev[..., :self.num_basis]
        pad = torch.zeros(x.shape[0], x.shape[1], self.num_basis - L,
                          dtype=B_prev.dtype, device=B_prev.device)
        return torch.cat([B_prev, pad], dim=-1)


class RCABSplineKANLinear(nn.Module):
    """Naive (unfused) B-spline KAN edge: base SiLU-linear branch + Cox-de Boor spline."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 spline_order: int = 3, include_base_branch: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        L = grid_size + spline_order
        self.L = L
        self.basis = _CoxDeBoorBasis(num_basis=L, spline_order=spline_order, grid_size=grid_size)
        # spline_weight layout [in, L, out] matches sparsefuse for easy reasoning
        self.spline_weight = nn.Parameter(torch.empty(in_features, L, out_features))
        self.include_base_branch = include_base_branch
        if include_base_branch:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.base_activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        if self.include_base_branch:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)
        basis = self.basis(x).to(x.dtype)                       # [B, in, L]
        spline_out = torch.einsum("bil,ilo->bo", basis, self.spline_weight)
        if self.include_base_branch:
            spline_out = spline_out + F.linear(self.base_activation(x), self.base_weight)
        if len(orig_shape) > 2:
            spline_out = spline_out.reshape(*orig_shape[:-1], self.out_features)
        return spline_out


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def make_kan_linear(
    backend: str,
    in_features: int,
    out_features: int,
    grid_size: int = 5,
    spline_order: int = 3,
    enable_standalone_scale_spline: bool = False,
    include_base_branch: bool = True,
) -> nn.Module:
    """Return a KAN-linear module for the requested backend.

    `enable_standalone_scale_spline=False` matches the SparseFuse paper's canonical
    no-scaler config (Table 7 / §5.3) for clean cross-backend comparison.
    """
    backend = backend.lower()
    if backend == "sparsefuse":
        from sparsefuse.layers import MatrixFusedKANLinear
        return MatrixFusedKANLinear(
            in_features, out_features,
            grid_size=grid_size, spline_order=spline_order,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            include_base_branch=include_base_branch,
            basis_type="bspline",
        )
    if backend == "ekan":
        try:
            from efficient_kan import KANLinear
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "efficient_kan not installed. pip install "
                "git+https://github.com/Blealtan/efficient-kan.git"
            ) from e
        return KANLinear(
            in_features, out_features,
            grid_size=grid_size, spline_order=spline_order,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
        )
    if backend == "rca_bspline":
        return RCABSplineKANLinear(
            in_features, out_features,
            grid_size=grid_size, spline_order=spline_order,
            include_base_branch=include_base_branch,
        )
    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")


def basis_count(grid_size: int, spline_order: int = 3) -> int:
    """L = number of B-spline basis functions used per edge (== grid_size + spline_order)."""
    return grid_size + spline_order


if __name__ == "__main__":
    # smoke test the backends importable without GPU (rca_bspline needs no extra deps)
    x = torch.randn(8, 16)
    layer = make_kan_linear("rca_bspline", 16, 32, grid_size=5)
    y = layer(F.layer_norm(x, (16,)))
    print("rca_bspline:", tuple(y.shape), "L =", layer.L)
    for b in ("ekan", "sparsefuse"):
        try:
            l2 = make_kan_linear(b, 16, 32, grid_size=5)
            print(f"{b}: import OK ->", tuple(l2(F.layer_norm(x, (16,))).shape))
        except Exception as e:  # noqa: BLE001
            print(f"{b}: not available locally ({type(e).__name__}: {str(e)[:60]})")
