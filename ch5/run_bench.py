"""
Ch5 benchmark runner — one measured (backend, G, |V|, seed) cell, plus sweep helpers and
a CLI. Emits CSV rows for efficiency (H2: latency/VRAM vs G) and feasibility (H3: OOM
frontier vs (G,|V|)). Parity (H1) on real datasets goes through real_model_adapter.py.

A "cell" = build synthetic graph + GNN+KAN(backend) -> warmup -> N train steps, measuring
per-step latency (CUDA events), peak VRAM, and OOM/error status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

import torch

from .gnn_kan_bench import BenchConfig, GNNKANBench, build_norm_adj, link_pred_loss
from .instrument import (cuda_sync_timer, git_commit, peak_vram_mb,
                         reset_peak_vram, run_guarded)
from .synth_graph import make_synth_graph

CSV_FIELDS = [
    "backend", "grid_size", "num_nodes", "num_edges", "hidden", "num_layers",
    "feat_dim", "seed", "status", "n_steps", "warmup_ms", "mean_step_ms",
    "p50_step_ms", "min_step_ms", "fwd_ms", "peak_alloc_mb", "peak_reserved_mb",
    "final_loss", "error", "commit_sparsefuse", "commit_rcaeval", "torch", "device",
]


def run_cell(
    backend: str,
    grid_size: int,
    num_nodes: int,
    *,
    feat_dim: int = 64,
    hidden: int = 64,
    num_layers: int = 3,
    spline_order: int = 3,
    seed: int = 0,
    n_steps: int = 30,
    warmup_steps: int = 5,
    lr: float = 2e-4,
    device: str | None = None,
    sparsefuse_dir: str = "/workspace/sparsefuse",
    rcaeval_dir: str = "/workspace/RCAEval",
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    base = {f: "" for f in CSV_FIELDS}
    base.update({
        "backend": backend, "grid_size": grid_size, "num_nodes": num_nodes,
        "hidden": hidden, "num_layers": num_layers, "feat_dim": feat_dim, "seed": seed,
        "n_steps": n_steps, "torch": torch.__version__, "device": dev.type,
        "commit_sparsefuse": git_commit(sparsefuse_dir),
        "commit_rcaeval": git_commit(rcaeval_dir),
    })

    g = make_synth_graph(num_nodes, feat_dim=feat_dim, m=3, seed=seed, device=dev)
    base["num_edges"] = g.num_edges

    def _train():
        cfg = BenchConfig(backend=backend, feat_dim=feat_dim, hidden=hidden,
                          num_layers=num_layers, grid_size=grid_size,
                          spline_order=spline_order)
        model = GNNKANBench(cfg).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        adj = build_norm_adj(g.edge_index, g.num_nodes, dev)

        # warmup (compile / autotune / cache)
        with cuda_sync_timer(dev) as wt:
            for _ in range(warmup_steps):
                opt.zero_grad(set_to_none=True)
                loss = link_pred_loss(model(g.x, adj), g.edge_index)
                loss.backward()
                opt.step()
        warmup_ms = wt[0]

        reset_peak_vram(dev)
        step_ms: list[float] = []
        fwd_ms: list[float] = []
        final_loss = float("nan")
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with cuda_sync_timer(dev) as ft:
                z = model(g.x, adj)
            fwd_ms.append(ft[0])
            with cuda_sync_timer(dev) as st:
                loss = link_pred_loss(z, g.edge_index)
                loss.backward()
                opt.step()
            step_ms.append(st[0] + ft[0])
            final_loss = float(loss.detach())
        vram = peak_vram_mb(dev)
        return {
            "warmup_ms": round(warmup_ms, 3),
            "mean_step_ms": round(statistics.mean(step_ms), 4),
            "p50_step_ms": round(statistics.median(step_ms), 4),
            "min_step_ms": round(min(step_ms), 4),
            "fwd_ms": round(statistics.median(fwd_ms), 4),
            "peak_alloc_mb": round(vram["peak_alloc_mb"], 1),
            "peak_reserved_mb": round(vram["peak_reserved_mb"], 1),
            "final_loss": round(final_loss, 5),
        }

    reset_peak_vram(dev)
    outcome = run_guarded(_train, dev)
    base["status"] = outcome.status
    if outcome.status == "ok":
        base.update(outcome.value)
    else:
        base["error"] = outcome.error
    return base


def run_cell_isolated(backend, grid_size, num_nodes, *, seed=0, **cell_kw) -> dict:
    """Run one cell in a fresh subprocess so a CUDA crash (illegal address / context
    poisoning) is contained to that cell instead of killing the whole sweep."""
    cmd = [sys.executable, "-u", "-m", "ch5.run_bench", "--single",
           "--backends", backend, "--grids", str(grid_size), "--nodes", str(num_nodes),
           "--seeds", str(seed)]
    for k, flag in (("feat_dim", "--feat-dim"), ("hidden", "--hidden"),
                    ("num_layers", "--num-layers"), ("n_steps", "--n-steps"),
                    ("warmup_steps", "--warmup-steps")):
        if k in cell_kw and cell_kw[k] is not None:
            cmd += [flag, str(cell_kw[k])]
    if cell_kw.get("device"):
        cmd += ["--device", cell_kw["device"]]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=os.environ.get("PYTHONPATH", "."))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    # subprocess died before emitting a result -> CUDA crash / illegal address
    base = {f: "" for f in CSV_FIELDS}
    base.update({"backend": backend, "grid_size": grid_size, "num_nodes": num_nodes,
                 "seed": seed, "status": "crash",
                 "error": (proc.stderr or proc.stdout)[-400:].replace("\n", " ")})
    return base


def sweep(backends, grids, node_counts, seeds, out_csv: str,
          isolate: bool = False, **cell_kw) -> list[dict]:
    rows: list[dict] = []
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    runner = run_cell_isolated if isolate else run_cell
    for backend in backends:
        for G in grids:
            for n in node_counts:
                for seed in seeds:
                    row = runner(backend, G, n, seed=seed, **cell_kw)
                    rows.append(row)
                    print(f"[{backend:11s} G={G:<3} |V|={n:<5} seed={seed}] "
                          f"status={row['status']:5s} step={row['mean_step_ms']}ms "
                          f"vram={row['peak_alloc_mb']}MB loss={row['final_loss']}",
                          flush=True)
                    _write_csv(out_csv, rows)
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description="Ch5 GNN+KAN efficiency/feasibility sweep")
    p.add_argument("--backends", default="sparsefuse,ekan")
    p.add_argument("--grids", default="5,15,50,100")
    p.add_argument("--nodes", default="100,250,500,1000")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--out", default="ch5/results/ch5_sweep.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--isolate", action="store_true",
                   help="run each cell in a fresh subprocess (contains CUDA crashes)")
    p.add_argument("--single", action="store_true",
                   help="internal: run exactly one cell and print RESULT_JSON")
    args = p.parse_args(argv)

    cell_kw = dict(feat_dim=args.feat_dim, hidden=args.hidden,
                   num_layers=args.num_layers, n_steps=args.n_steps,
                   warmup_steps=args.warmup_steps, device=args.device)

    if args.single:
        row = run_cell(
            args.backends.split(",")[0].strip(),
            _parse_int_list(args.grids)[0],
            _parse_int_list(args.nodes)[0],
            seed=_parse_int_list(args.seeds)[0], **cell_kw)
        print("RESULT_JSON:" + json.dumps(row), flush=True)
        return

    rows = sweep(
        backends=[b.strip() for b in args.backends.split(",") if b.strip()],
        grids=_parse_int_list(args.grids),
        node_counts=_parse_int_list(args.nodes),
        seeds=_parse_int_list(args.seeds),
        out_csv=args.out, isolate=args.isolate, **cell_kw,
    )
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nDone: {n_ok}/{len(rows)} cells ok -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
