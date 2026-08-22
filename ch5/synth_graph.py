"""
B0.3 — Synthetic scale-free directed-graph generator for Ch5 feasibility/efficiency runs.

Targets (aligned with Ch4): |V| in {100, 250, 500, 1000}, |E| ~= 3|V| (avg out-degree ~3,
matching Ch4's "avg degree 3.2, directed"). Topology = Barabasi-Albert preferential
attachment, oriented older->newer to give a causal DAG (typical of microservice
dependency / RCA call graphs).

No networkx dependency: BA is implemented directly so the Modal image stays light and
generation is fully deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SynthGraph:
    num_nodes: int
    edge_index: torch.Tensor   # [2, E] long, directed (row=src, col=dst)
    x: torch.Tensor            # [num_nodes, feat_dim] float32 node features
    seed: int
    m: int                     # BA attachment parameter

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def avg_out_degree(self) -> float:
        return self.num_edges / max(1, self.num_nodes)


def _barabasi_albert_edges(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Undirected BA graph -> array of (u, v) with u < v. |E| = m*(n-m)."""
    if n <= m:
        # fully connect the seed clique
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), np.int64)

    # repeated-nodes list for preferential attachment
    targets = list(range(m))
    repeated: list[int] = []
    edges: list[tuple[int, int]] = []

    for new_node in range(m, n):
        for t in targets:
            edges.append((t, new_node))  # older t < newer new_node
        repeated.extend(targets)
        repeated.extend([new_node] * m)

        # sample m distinct existing nodes weighted by degree
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.integers(0, len(repeated))])
        targets = list(chosen)

    return np.asarray(edges, dtype=np.int64)


def make_synth_graph(
    num_nodes: int,
    feat_dim: int = 64,
    m: int = 3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    correlated_features: bool = True,
) -> SynthGraph:
    """Build a directed scale-free graph with node features.

    Args:
        num_nodes: |V|.
        feat_dim: node feature dimension F.
        m: BA attachment param; avg out-degree ~= m (|E| ~= m*|V| for large |V|).
        seed: RNG seed for reproducibility.
        device: where to place tensors.
        correlated_features: if True, smooth features over the graph (1 step of
            neighbour averaging) so node signals carry mild graph structure, closer
            to real RCA metrics than pure i.i.d. noise. Set False for pure Gaussian.
    """
    rng = np.random.default_rng(seed)
    edges = _barabasi_albert_edges(num_nodes, m, rng)  # [E, 2], src<dst (causal DAG)

    # Build everything on CPU first, then move to `device` at the end (avoids
    # cross-device index_add_ during feature smoothing).
    if edges.shape[0] == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges.T, dtype=torch.long)  # [2, E]

    # node features
    x = torch.from_numpy(rng.standard_normal((num_nodes, feat_dim)).astype(np.float32))
    if correlated_features and edge_index.shape[1] > 0:
        # one symmetric neighbour-averaging smoothing pass
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        agg = x.clone()
        agg.index_add_(0, col, x[row])
        agg.index_add_(0, row, x[col])
        x = 0.5 * x + 0.5 * (agg / deg.unsqueeze(1))

    edge_index = edge_index.to(device)
    x = x.to(device)
    return SynthGraph(num_nodes=num_nodes, edge_index=edge_index, x=x, seed=seed, m=m)


if __name__ == "__main__":
    for n in (100, 250, 500, 1000):
        g = make_synth_graph(n, feat_dim=64, m=3, seed=0)
        print(f"|V|={n:5d}  |E|={g.num_edges:5d}  avg_out_deg={g.avg_out_degree:.2f}  "
              f"x={tuple(g.x.shape)}")
