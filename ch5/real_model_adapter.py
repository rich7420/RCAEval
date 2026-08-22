"""
B0.1 (real model) — swap the B-spline compute path of the *actual* GNN_KAN to SparseFuse,
for parity (H1) on real datasets (Train-Ticket / RE1 / RE2).

Scope, per the Ch5 definition: only the cubic-B-spline spline term inside each KAN
evaluator (KAN_edge / KAN_node / KAN_message) is replaced by a fused SparseFuse layer.
The base SiLU-linear branch, the learnable-activation branch, the 0.3/0.5/0.2 combine
weights, LayerNorm, dual-graph, attention, residual, fusion and ranking are ALL unchanged.

  naive arm  : the model as-is with basis_function='b_spline'  (Cox-de Boor evaluator)
  fused arm  : swap_to_sparsefuse(model, grid_size=G)          (same B-spline, fused kernel)

Only applicable when basis_function == 'b_spline' (SparseFuse accelerates cubic B-spline's
local support active-K=4; Chebyshev / RPK are global-support and cannot be swapped).

Note: importing the real model pulls torch_geometric, which RCAEval pins to a torch-1.12
era version. Running this on Modal H100 (torch 2.10) needs a torch_geometric build matching
torch 2.10 — see ch5/README.md "Parity image".
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _sf_spline_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement SimplifiedKANLayer.forward: identical combine, fused spline term."""
    try:
        x2 = x
        if torch.isnan(x2).any() or torch.isinf(x2).any():
            x2 = torch.nan_to_num(x2, nan=0.0, posinf=1.0, neginf=-1.0)

        # --- swapped kernel: fused B-spline spline term (replaces basis(x) + einsum) ---
        # SparseFuse requires normalized input; the original basis normalized internally.
        poly_output = self._sf_layer(self._sf_norm(x2))

        base_output = self.base_transform(x2)
        activation_output = self.kan_learnable_activation(x2)

        final_output = torch.zeros_like(base_output)
        use_base = torch.all(torch.isfinite(base_output))
        use_poly = torch.all(torch.isfinite(poly_output))
        use_activation = torch.all(torch.isfinite(activation_output))
        if use_base:
            final_output += base_output * 0.3
        if use_poly:
            final_output += poly_output * 0.5
        if use_activation:
            final_output += activation_output * 0.2
        if not (use_base or use_poly or use_activation):
            return torch.zeros_like(base_output)

        kan_output = self.ln(final_output)
        if not torch.all(torch.isfinite(kan_output)):
            return torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
        return kan_output
    except Exception:
        if x.shape[1] == self.output_dim:
            return x
        return (x[:, :self.output_dim] if x.shape[1] > self.output_dim
                else F.pad(x, (0, self.output_dim - x.shape[1])))


def swap_to_sparsefuse(model: torch.nn.Module, grid_size: int, spline_order: int = 3,
                       enable_standalone_scale_spline: bool = False) -> int:
    """In-place swap every B-spline SimplifiedKANLayer's spline kernel to SparseFuse.

    Returns the number of KAN evaluators swapped (edge + node + message processors).
    Raises if the model's KAN layers are not configured for B-spline.
    """
    from sparsefuse.layers import MatrixFusedKANLinear
    # import here to avoid pulling torch_geometric at module import time
    from RCAEval.gnn_kan_module.kan_components.kan_layers import SimplifiedKANLayer

    swapped = 0
    for module in model.modules():
        if isinstance(module, SimplifiedKANLayer):
            if getattr(module, "basis_function", None) != "b_spline":
                raise ValueError(
                    f"KAN layer basis_function is {getattr(module, 'basis_function', None)!r}, "
                    "expected 'b_spline'. SparseFuse only swaps into a B-spline config."
                )
            in_dim, out_dim = module.input_dim, module.output_dim
            sf = MatrixFusedKANLinear(
                in_dim, out_dim, grid_size=grid_size, spline_order=spline_order,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
                include_base_branch=False,  # base branch stays the layer's own base_transform
                basis_type="bspline",
            )
            module.add_module("_sf_layer", sf)
            module.add_module("_sf_norm", torch.nn.LayerNorm(in_dim, eps=1e-4))
            module.forward = types.MethodType(_sf_spline_forward, module)
            swapped += 1
    if swapped == 0:
        raise RuntimeError("No SimplifiedKANLayer found to swap — is this a GNN_KAN model?")
    return swapped


def count_kan_layers(model: torch.nn.Module) -> int:
    from RCAEval.gnn_kan_module.kan_components.kan_layers import SimplifiedKANLayer
    return sum(isinstance(m, SimplifiedKANLayer) for m in model.modules())
