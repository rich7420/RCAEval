"""
Modal H100 harness for H1 parity: real GNN_KAN + real RCA data, naive vs SparseFuse kernel.

Uploads the RCAEval code package (2.4 MB), ch5/, the sparsefuse package, and the RE1-OB
dataset (168 MB), then runs the real gnn_kan_rca pipeline (basis_function='b_spline') under
both kernels across G and seeds, scoring AC@k. Parity = naive and sparsefuse match.

Run:
  .venv-modal/bin/modal run ch5/modal_parity.py --task smoke
  .venv-modal/bin/modal run ch5/modal_parity.py --task parity
"""

import os
import time

import modal

app = modal.App("ch5-parity-gnnkan")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.10.0+cu126",
        "triton==3.6.0",
        "numpy<2",
        "pandas<2.3",
        "scipy",
        "scikit-learn",
        "scikit-network",
        "requests",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu126",
    )
)

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /Users/rich/RCAEval
_sparsefuse = os.path.join(os.path.dirname(_repo), "sparsefuse")

image = (
    image
    .add_local_dir(os.path.join(_repo, ".git"), remote_path="/workspace/RCAEval/.git", copy=True)
    .add_local_dir(os.path.join(_sparsefuse, ".git"), remote_path="/workspace/sparsefuse/.git", copy=True)
    # code + data mounts (startup, no rebuild)
    .add_local_dir(os.path.join(_repo, "RCAEval"), remote_path="/workspace/RCAEval/RCAEval")
    .add_local_dir(os.path.join(_repo, "ch5"), remote_path="/workspace/RCAEval/ch5")
    .add_local_dir(os.path.join(_sparsefuse, "sparsefuse"), remote_path="/workspace/sparsefuse")
    .add_local_dir(os.path.join(_repo, "data", "RE1", "RE1-OB"),
                   remote_path="/workspace/RCAEval/data/RE1/RE1-OB")
    .add_local_dir(os.path.join(_repo, "data", "RE1", "RE1-SS"),
                   remote_path="/workspace/RCAEval/data/RE1/RE1-SS")
)

# import paths: RCAEval package + ch5 live under /workspace/RCAEval; sparsefuse under /workspace
WORKDIR = "/workspace/RCAEval"
PYPATH = "/workspace:/workspace/RCAEval"


def _parity(kernels, grids, seeds, max_cases, epochs, out_rel, dataset_name="re1-ob",
            dataset_dir="data/RE1/RE1-OB"):
    import csv as _csv
    import sys
    import torch
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, WORKDIR)
    os.chdir(WORKDIR)
    print(f"H100 env — torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    from ch5.parity_real import run

    all_rows = []
    for kernel in kernels:
        for G in grids:
            rows = run(dataset_dir, dataset_name, kernel, G, seeds,
                       max_cases, epochs, out_csv=None, rcaeval_dir=WORKDIR)
            all_rows.extend(rows)

    out_abs = os.path.join(WORKDIR, out_rel)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(out_abs) as f:
        return {"csv_data": f.read(), "csv_filename": out_rel}


@app.function(image=image, gpu="H100", timeout=3600)
def smoke():
    """2 cases, both kernels, G=5 — confirm the real pipeline runs under torch 2.10 + swap."""
    t = time.time()
    res = _parity(["naive", "sparsefuse"], [5], [0], max_cases=2, epochs=8,
                  out_rel="ch5/results/parity_smoke.csv")
    res["time"] = time.time() - t
    return res


@app.function(image=image, gpu="H100", timeout=10800)
def parity():
    """Full H1 parity on RE1-OB: naive vs sparsefuse, G in {5,15,50,100}, 3 seeds."""
    t = time.time()
    res = _parity(["naive", "sparsefuse"], [5, 15, 50, 100], [0, 1, 2],
                  max_cases=40, epochs=30, out_rel="ch5/results/parity_re1ob.csv")
    res["time"] = time.time() - t
    return res


@app.function(image=image, gpu="H100", timeout=10800)
def parity_ss():
    """Full H1 parity on RE1-SS: naive vs sparsefuse, G in {5,15,50,100}, 3 seeds."""
    t = time.time()
    res = _parity(["naive", "sparsefuse"], [5, 15, 50, 100], [0, 1, 2],
                  max_cases=40, epochs=30, out_rel="ch5/results/parity_re1ss.csv",
                  dataset_name="re1-ss", dataset_dir="data/RE1/RE1-SS")
    res["time"] = time.time() - t
    return res


@app.function(image=image, gpu="H100", timeout=7200)
def timing():
    """Real-model end-to-end wall time, naive vs SparseFuse, RE1-OB+RE1-SS, all G, 1 seed.

    Captures gnn_kan_rca processing_time per case (mean). NOTE: on small RCA graphs
    (4-64 services) the kernel is a small fraction of the pipeline (ICA features + graph
    construction dominate), so expect modest differences here; the isolated kernel timing
    is H2 (synthetic, efficiency.csv)."""
    import csv as _csv
    import sys
    import torch
    sys.path.insert(0, "/workspace"); sys.path.insert(0, WORKDIR); os.chdir(WORKDIR)
    print(f"H100 env — torch {torch.__version__}", flush=True)
    from ch5.parity_real import run
    t = time.time()
    rows = []
    for dname, ddir in (("re1-ob", "data/RE1/RE1-OB"), ("re1-ss", "data/RE1/RE1-SS")):
        for kernel in ("naive", "sparsefuse"):
            for G in (5, 15, 50, 100):
                rows += run(ddir, dname, kernel, G, [0], 40, 30, out_csv=None,
                            rcaeval_dir=WORKDIR)
    out_rel = "ch5/results/parity_timing.csv"
    out_abs = os.path.join(WORKDIR, out_rel)
    with open(out_abs, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out_abs) as f:
        return {"csv_data": f.read(), "csv_filename": out_rel, "time": time.time() - t}


_TASKS = {"smoke": smoke, "parity": parity, "parity_ss": parity_ss, "timing": timing}


@app.local_entrypoint()
def main(task: str = "smoke"):
    if task == "list":
        print("tasks:", ", ".join(_TASKS)); return
    for n in ([t.strip() for t in task.split(",")] if task != "all" else list(_TASKS)):
        if n not in _TASKS:
            raise ValueError(f"unknown task {n!r}; choose {list(_TASKS)}")
        print(f"=> [{n}] launching on H100 ...")
        res = _TASKS[n].remote()
        if res.get("csv_data"):
            local = os.path.join(_repo, res["csv_filename"])
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w") as f:
                f.write(res["csv_data"])
            print(f"  -> downloaded {local}")
        print(f"  elapsed {res.get('time', 0):.1f}s")
    print("done.")
