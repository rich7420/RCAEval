"""
Ch5 GNN+KAN benchmark model — a clean, faithful re-implementation of
RCAEval/gnn_kan_module/kan_components/kan_layers.py::OptimizedGNNKANEncoder, with a
switchable KAN backend and no exception-swallowing (so OOM / latency / NaN are observable).

Structure mirrors the real RCA encoder:
  - per-layer: LayerNorm -> KAN node-transform edge (in->out)
  - every other layer: sym-normalized adjacency message passing -> KAN message processor
    -> residual (0.3) [same coefficients as the RCA encoder]
  - self-supervised link-prediction head (dot-product) trained with negative sampling,
    matching the RCA training objective (negative_sampling in training.py).

Used for efficiency (H2) and feasibility (H3) sweeps on synthetic graphs. Parity (H1) on
real datasets goes through the real-model adapter, not this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_backends import make_kan_linear


def build_norm_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Symmetric-normalized sparse adjacency D^-1/2 (A+A^T) D^-1/2, as in the RCA encoder."""
    if edge_index.shape[1] == 0:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.sparse_coo_tensor(idx, torch.empty(0, device=device, dtype=dtype),
                                       (num_nodes, num_nodes)).coalesce()
    row, col = edge_index
    # symmetrize
    row2 = torch.cat([row, col])
    col2 = torch.cat([col, row])
    idx = torch.stack([row2, col2], dim=0)
    vals = torch.ones(idx.shape[1], device=device, dtype=dtype)
    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-6)
    dinv = deg.pow(-0.5)
    r, c = adj.indices()
    norm_vals = dinv[r] * adj.values() * dinv[c]
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, (num_nodes, num_nodes),
                                   device=device).coalesce()


@dataclass
class BenchConfig:
    backend: str = "sparsefuse"
    feat_dim: int = 64
    hidden: int = 64
    num_layers: int = 3
    grid_size: int = 5
    spline_order: int = 3
    enable_standalone_scale_spline: bool = False


class GNNKANBench(nn.Module):
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg
        b = cfg.backend
        G, p = cfg.grid_size, cfg.spline_order
        kw = dict(grid_size=G, spline_order=p,
                  enable_standalone_scale_spline=cfg.enable_standalone_scale_spline)

        dims = [cfg.feat_dim] + [cfg.hidden] * cfg.num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(cfg.num_layers)])
        self.kan_layers = nn.ModuleList([
            make_kan_linear(b, dims[i], dims[i + 1], **kw) for i in range(cfg.num_layers)
        ])
        # message processors for the layers that do message passing (every other layer)
        self.msg_norms = nn.ModuleList()
        self.msg_procs = nn.ModuleList()
        for i in range(cfg.num_layers):
            if i % 2 == 0:
                self.msg_norms.append(nn.LayerNorm(dims[i + 1]))
                self.msg_procs.append(make_kan_linear(b, dims[i + 1], dims[i + 1], **kw))

    def encode(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = x
        mp = 0
        for i, (norm, layer) in enumerate(zip(self.norms, self.kan_layers)):
            h = layer(norm(h))
            if i % 2 == 0 and norm_adj._nnz() > 0:
                message = torch.sparse.mm(norm_adj, h)
                message = self.msg_procs[mp](self.msg_norms[mp](message))
                h = h + 0.3 * message
                mp += 1
        return h

    def forward(self, x, norm_adj):
        return self.encode(x, norm_adj)


def link_pred_loss(z: torch.Tensor, edge_index: torch.Tensor,
                   num_neg: int | None = None) -> torch.Tensor:
    """BCE link-prediction loss with uniform negative sampling (RCA objective)."""
    num_nodes = z.shape[0]
    row, col = edge_index
    pos = (z[row] * z[col]).sum(-1)
    k = num_neg or edge_index.shape[1]
    neg_row = torch.randint(0, num_nodes, (k,), device=z.device)
    neg_col = torch.randint(0, num_nodes, (k,), device=z.device)
    neg = (z[neg_row] * z[neg_col]).sum(-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg_loss = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    return pos_loss + neg_loss


if __name__ == "__main__":
    from .synth_graph import make_synth_graph
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = make_synth_graph(100, feat_dim=64, m=3, seed=0, device=dev)
    cfg = BenchConfig(backend="rca_bspline", grid_size=5)
    model = GNNKANBench(cfg).to(dev)
    adj = build_norm_adj(g.edge_index, g.num_nodes, dev)
    z = model(g.x, adj)
    loss = link_pred_loss(z, g.edge_index)
    loss.backward()
    print(f"backend={cfg.backend} z={tuple(z.shape)} loss={loss.item():.4f} OK (device={dev})")
