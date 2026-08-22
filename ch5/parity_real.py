"""
H1 parity on the REAL GNN_KAN + REAL RCA datasets.

Runs the actual gnn_kan_rca pipeline with basis_function='b_spline' under two kernels:
  --kernel naive       : Cox-de Boor B-spline evaluator (unchanged GNN_KAN)
  --kernel sparsefuse  : same model, B-spline spline-term swapped to the SparseFuse fused
                          Triton kernel (via the gnn_kan_rca kernel hook)

For each case it reads data.csv + inject_time.txt, derives the ground-truth service from
the directory name (<service>_<fault>/<case>/), gets the ranked root-cause list, and scores
AC@k. Parity = naive and sparsefuse produce the same accuracy (Ch5 is a systems claim; this
confirms the kernel swap does not change the RCA result).

Example:
  PYTHONPATH=. .venv310/bin/python -m ch5.parity_real \
      --dataset-dir data/RE1/RE1-OB --dataset-name re1-ob \
      --kernel naive --grid-size 5 --seeds 0,1,2 --max-cases 30 --epochs 30 \
      --out ch5/results/parity_re1ob_naive_g5.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from os.path import basename, dirname, join

import torch

from .instrument import git_commit


def _list_cases(dataset_dir: str):
    paths = sorted(glob.glob(join(dataset_dir, "**", "data.csv"), recursive=True))
    cases = []
    for p in paths:
        try:
            service, metric = basename(dirname(dirname(p))).split("_", 1)
        except ValueError:
            continue
        itf = join(dirname(p), "inject_time.txt")
        if not os.path.exists(itf):
            continue
        cases.append((p, service, metric, itf))
    return cases


def run(dataset_dir, dataset_name, kernel, grid_size, seeds, max_cases, epochs,
        out_csv, sparsefuse_dir="/workspace/sparsefuse", rcaeval_dir="."):
    from RCAEval.utility import read_data
    from RCAEval.e2e.gnnkan import gnn_kan_rca

    cases = _list_cases(dataset_dir)
    if max_cases:
        cases = cases[:max_cases]
    print(f"{dataset_name}: {len(cases)} cases | kernel={kernel} G={grid_size} "
          f"seeds={seeds} epochs={epochs}", flush=True)

    rows = []
    for seed in seeds:
        hit = {1: 0, 3: 0, 5: 0}
        n = 0
        proc_times = []
        for (p, gt_service, metric, itf) in cases:
            with open(itf) as f:
                inject_time = int(f.readlines()[0].strip())
            try:
                data = read_data(p)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                res = gnn_kan_rca(data, inject_time, dataset=dataset_name,
                                  basis_function="b_spline", kernel=kernel,
                                  kan_grid_size=grid_size, num_epochs=epochs)
                ranks = [str(r) for r in res["ranks"]]
                if res.get("processing_time") is not None:
                    proc_times.append(float(res["processing_time"]))
            except Exception as e:  # one bad case shouldn't kill the sweep
                print(f"  [skip] {p}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            n += 1
            for k in (1, 3, 5):
                if any(gt_service in r for r in ranks[:k]):
                    hit[k] += 1
        row = {
            "dataset": dataset_name, "kernel": kernel, "grid_size": grid_size,
            "seed": seed, "n_cases": n,
            "AC@1": round(hit[1] / n, 4) if n else 0.0,
            "AC@3": round(hit[3] / n, 4) if n else 0.0,
            "AC@5": round(hit[5] / n, 4) if n else 0.0,
            "proc_time_s": round(sum(proc_times) / len(proc_times), 4) if proc_times else "",
            "epochs": epochs, "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "commit_sparsefuse": git_commit(sparsefuse_dir),
            "commit_rcaeval": git_commit(rcaeval_dir),
        }
        rows.append(row)
        print(f"  seed={seed} n={n} AC@1={row['AC@1']} AC@3={row['AC@3']} "
              f"AC@5={row['AC@5']}", flush=True)

    if out_csv:
        os.makedirs(dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"-> {out_csv}", flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="H1 parity on real GNN_KAN + RCA data")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--kernel", choices=["naive", "sparsefuse"], default="naive")
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-cases", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--rcaeval-dir", default=".")
    a = p.parse_args(argv)
    run(a.dataset_dir, a.dataset_name, a.kernel, a.grid_size,
        [int(s) for s in a.seeds.split(",") if s.strip()],
        a.max_cases, a.epochs, a.out, rcaeval_dir=a.rcaeval_dir)


if __name__ == "__main__":
    main()
