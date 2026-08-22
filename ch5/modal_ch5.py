"""
B0.5 — Modal H100 harness for Ch5 (GNN+KAN: naive kernel vs SparseFuse kernel).

Mirrors sparsefuse/scripts/modal_h100_dispatcher.py: defines an H100 image in code,
uploads the ch5/ harness + the sparsefuse package, runs measured sweeps in a subprocess,
streams logs, and downloads result CSVs back to the local repo.

Run from the modal venv:
    .venv-modal/bin/modal run ch5/modal_ch5.py --task list
    .venv-modal/bin/modal run ch5/modal_ch5.py --task sanity
    .venv-modal/bin/modal run ch5/modal_ch5.py --task efficiency
    .venv-modal/bin/modal run ch5/modal_ch5.py --task feasibility

Image pins torch 2.10 + triton 3.6 (SparseFuse's native stack). The synthetic
efficiency/feasibility sweeps need only torch + sparsefuse + efficient_kan — no
torch_geometric, so the RCA package is not required here. Real-data parity (H1) on the
full GNN_KAN runs through a separate, heavier image (see real_model_adapter.py).
"""

import os
import subprocess
import time

import modal

app = modal.App("ch5-gnnkan-sparsefuse")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.10.0+cu126",
        "triton==3.6.0",
        "numpy",
        "pandas",
        "git+https://github.com/Blealtan/efficient-kan.git",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu126",
    )
)

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # /Users/rich/RCAEval
_sparsefuse = os.path.join(os.path.dirname(_repo), "sparsefuse")          # /Users/rich/sparsefuse

image = (
    image
    # copy=True steps (baked into image) must come BEFORE non-copy add_local_* mounts.
    # carry the two .git dirs so commit hashes are recorded for provenance
    .add_local_dir(os.path.join(_repo, ".git"), remote_path="/workspace/RCAEval/.git", copy=True)
    .add_local_dir(os.path.join(_sparsefuse, ".git"), remote_path="/workspace/sparsefuse/.git", copy=True)
    # code mounts last (added at container startup, no rebuild on edit)
    .add_local_dir(os.path.join(_repo, "ch5"), remote_path="/workspace/ch5")
    .add_local_dir(os.path.join(_sparsefuse, "sparsefuse"), remote_path="/workspace/sparsefuse")
)


def _stream(cmd, env, cwd):
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
    out = []
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        out.append(line)
    proc.wait()
    return "".join(out)


def _run_sweep(out_rel: str, extra_args: list[str]):
    """Run ch5.run_bench inside /workspace and return stdout + the result CSV."""
    import torch
    print(f"H100 env — torch {torch.__version__}, cuda {torch.version.cuda}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    out_abs = f"/workspace/{out_rel}"
    cmd = ["python3", "-u", "-m", "ch5.run_bench", "--out", out_abs] + extra_args
    start = time.time()
    stdout = _stream(cmd, env, cwd="/workspace")
    csv_data = None
    if os.path.exists(out_abs):
        with open(out_abs) as f:
            csv_data = f.read()
    return {"stdout": stdout, "time": time.time() - start,
            "csv_data": csv_data, "csv_filename": out_rel}


@app.function(image=image, gpu="H100", timeout=1800)
def verify():
    """Prove backend='sparsefuse' runs the real Triton FUSED kernel (no silent fallback)."""
    import torch
    print(f"H100 env — torch {torch.__version__}, cuda {torch.version.cuda}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    start = time.time()
    stdout = _stream(["python3", "-u", "-m", "ch5.verify_kernel"], env, cwd="/workspace")
    return {"stdout": stdout, "time": time.time() - start, "csv_data": None,
            "csv_filename": None}


@app.function(image=image, gpu="H100", timeout=1800)
def sanity():
    """Tiny: all 3 backends at G=5, |V|=100, 1 seed — confirms the swap runs on H100."""
    return _run_sweep(
        "ch5/results/sanity.csv",
        ["--backends", "rca_bspline,ekan,sparsefuse", "--grids", "5",
         "--nodes", "100", "--seeds", "0", "--n-steps", "20"],
    )


@app.function(image=image, gpu="H100", timeout=7200)
def efficiency():
    """H2: latency/VRAM ratio vs G. naive vs SparseFuse over G grid at fixed |V|=500."""
    return _run_sweep(
        "ch5/results/efficiency.csv",
        ["--backends", "ekan,sparsefuse", "--grids", "5,15,50,100",
         "--nodes", "500", "--seeds", "0,1,2", "--n-steps", "30"],
    )


@app.function(image=image, gpu="H100", timeout=14400)
def feasibility():
    """H3: OOM frontier over (G, |V|) for both kernels.

    Stress axis = |V| (nodes) up to 100k at width 512, so naive eKAN materializes a dense
    [|V|, width, L] basis that approaches 80GB at high G x large graph, while SparseFuse
    (no materialization) survives. OOM is deterministic -> 1 seed. 5 |V| x 4 G x 2 backends.
    """
    return _run_sweep(
        "ch5/results/feasibility.csv",
        ["--backends", "ekan,sparsefuse", "--grids", "5,15,50,100",
         "--nodes", "1000,5000,20000,50000,100000", "--seeds", "0",
         "--hidden", "512", "--feat-dim", "64", "--n-steps", "12", "--warmup-steps", "3",
         "--isolate"],
    )


_TASKS = {"verify": verify, "sanity": sanity, "efficiency": efficiency,
          "feasibility": feasibility}


def _save(res):
    if res.get("csv_data") and res.get("csv_filename"):
        local = os.path.join(_repo, res["csv_filename"])
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w") as f:
            f.write(res["csv_data"])
        print(f"  -> downloaded {local}")
    print(f"  elapsed {res['time']:.1f}s")


@app.local_entrypoint()
def main(task: str = "sanity"):
    if task == "list":
        print("tasks:", ", ".join(_TASKS))
        return
    names = list(_TASKS) if task == "all" else [t.strip() for t in task.split(",")]
    for n in names:
        if n not in _TASKS:
            raise ValueError(f"unknown task {n!r}; choose from {list(_TASKS)} or 'all'/'list'")
        print(f"=> [{n}] launching on H100 ...")
        _save(_TASKS[n].remote())
    print("done.")
