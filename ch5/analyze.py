"""
Ch5 analysis — turn the efficiency/feasibility CSVs into figures + tables.

Outputs (under ch5/results/):
  figures/efficiency_latency_vs_G.png   H2 latency vs grid size, naive vs SparseFuse
  figures/efficiency_vram_vs_G.png      H2 peak VRAM vs grid size
  figures/feasibility_frontier.png      H3 OOM frontier heatmap (status grid)
  figures/feasibility_vram_ratio.png    H3 VRAM(naive)/VRAM(SF) heatmap
  tables/efficiency_summary.md          per-(backend,G) mean step ms + VRAM + ratio
  tables/feasibility_frontier.md        status grid + VRAM, both backends

Run: .venv310/bin/python -m ch5.analyze
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
NAIVE = "ekan"          # the naive control plotted against sparsefuse
FUSED = "sparsefuse"
COLOR = {NAIVE: "#d62728", FUSED: "#1f77b4"}
LABEL = {NAIVE: "naive (eKAN)", FUSED: "SparseFuse (fused)"}


def _ok(df):
    return df[df["status"] == "ok"].copy()


# ----------------------------------------------------------------------------- efficiency
def efficiency(path: str):
    df = pd.read_csv(path)
    df = _ok(df)
    for c in ("grid_size", "mean_step_ms", "peak_alloc_mb"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = (df.groupby(["backend", "grid_size"])
             .agg(step_ms=("mean_step_ms", "mean"), step_sd=("mean_step_ms", "std"),
                  vram=("peak_alloc_mb", "mean"))
             .reset_index())

    for metric, ylab, fname in [("step_ms", "mean train-step latency (ms)",
                                 "efficiency_latency_vs_G.png"),
                                ("vram", "peak VRAM (MB)", "efficiency_vram_vs_G.png")]:
        plt.figure(figsize=(5.5, 4))
        for be in (NAIVE, FUSED):
            s = agg[agg["backend"] == be].sort_values("grid_size")
            if s.empty:
                continue
            yerr = s["step_sd"] if metric == "step_ms" else None
            plt.errorbar(s["grid_size"], s[metric], yerr=yerr, marker="o",
                         color=COLOR[be], label=LABEL[be], capsize=3)
        plt.xlabel("grid size G"); plt.ylabel(ylab)
        plt.title(f"Ch5 H2 — {ylab} vs G  (|V|=500, 3-layer width-64)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, fname), dpi=140); plt.close()

    # summary table with ratio
    piv = agg.pivot(index="grid_size", columns="backend", values=["step_ms", "vram"])
    lines = ["# H2 Efficiency summary (|V|=500, width 64, mean over seeds)", "",
             "| G | naive ms | SF ms | speedup | naive VRAM MB | SF VRAM MB | VRAM ratio |",
             "|---|---|---|---|---|---|---|"]
    for G in sorted(agg["grid_size"].unique()):
        nm = piv.loc[G, ("step_ms", NAIVE)]; fm = piv.loc[G, ("step_ms", FUSED)]
        nv = piv.loc[G, ("vram", NAIVE)]; fv = piv.loc[G, ("vram", FUSED)]
        lines.append(f"| {G} | {nm:.2f} | {fm:.2f} | {nm/fm:.2f}× | "
                     f"{nv:.0f} | {fv:.0f} | {nv/fv:.2f}× |")
    _write(os.path.join(TAB, "efficiency_summary.md"), "\n".join(lines))
    return agg


# ----------------------------------------------------------------------------- feasibility
def feasibility(path: str):
    df = pd.read_csv(path)
    df["grid_size"] = pd.to_numeric(df["grid_size"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["peak_alloc_mb"] = pd.to_numeric(df["peak_alloc_mb"], errors="coerce")
    Gs = sorted(df["grid_size"].dropna().unique())
    Vs = sorted(df["num_nodes"].dropna().unique())

    # status code grid: 0 ok, 1 oom, 2 err/crash
    code = {"ok": 0, "oom": 1, "error": 2, "crash": 2}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d62728", "#7f7f7f"])
    for ax, be in zip(axes, (NAIVE, FUSED)):
        M = np.full((len(Gs), len(Vs)), np.nan)
        for i, g in enumerate(Gs):
            for j, v in enumerate(Vs):
                r = df[(df["backend"] == be) & (df["grid_size"] == g) & (df["num_nodes"] == v)]
                if not r.empty:
                    M[i, j] = code.get(r.iloc[0]["status"], 2)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(Vs))); ax.set_xticklabels([f"{int(v/1000)}k" for v in Vs])
        ax.set_yticks(range(len(Gs))); ax.set_yticklabels([int(g) for g in Gs])
        ax.set_xlabel("|V| (nodes)"); ax.set_ylabel("grid size G")
        ax.set_title(f"{LABEL[be]}")
        for i, g in enumerate(Gs):
            for j, v in enumerate(Vs):
                r = df[(df["backend"] == be) & (df["grid_size"] == g) & (df["num_nodes"] == v)]
                if not r.empty:
                    st = r.iloc[0]["status"]
                    txt = (f"{float(r.iloc[0]['peak_alloc_mb'])/1000:.0f}G"
                           if st == "ok" and pd.notna(r.iloc[0]["peak_alloc_mb"])
                           else st.upper()[:3])
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white")
    fig.suptitle("Ch5 H3 — OOM frontier (green=trains, red=OOM, grey=kernel limit; width 512)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "feasibility_frontier.png"), dpi=140); plt.close(fig)

    # VRAM ratio heatmap where both ok
    R = np.full((len(Gs), len(Vs)), np.nan)
    for i, g in enumerate(Gs):
        for j, v in enumerate(Vs):
            rn = df[(df["backend"] == NAIVE) & (df["grid_size"] == g) & (df["num_nodes"] == v)]
            rf = df[(df["backend"] == FUSED) & (df["grid_size"] == g) & (df["num_nodes"] == v)]
            if (not rn.empty and not rf.empty and rn.iloc[0]["status"] == "ok"
                    and rf.iloc[0]["status"] == "ok"):
                R[i, j] = rn.iloc[0]["peak_alloc_mb"] / rf.iloc[0]["peak_alloc_mb"]
    plt.figure(figsize=(6, 4))
    im = plt.imshow(R, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="VRAM ratio naive/SF")
    plt.xticks(range(len(Vs)), [f"{int(v/1000)}k" for v in Vs])
    plt.yticks(range(len(Gs)), [int(g) for g in Gs])
    plt.xlabel("|V|"); plt.ylabel("G")
    for i in range(len(Gs)):
        for j in range(len(Vs)):
            if not np.isnan(R[i, j]):
                plt.text(j, i, f"{R[i,j]:.1f}×", ha="center", va="center",
                         color="white", fontsize=9)
    plt.title("Ch5 H3 — peak VRAM reduction (naive/SF, both train)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "feasibility_vram_ratio.png"), dpi=140); plt.close()

    # frontier table
    lines = ["# H3 Feasibility frontier (width 512; cell = peak VRAM MB or status)", ""]
    for be in (NAIVE, FUSED):
        lines += [f"## {LABEL[be]}", "",
                  "| G\\|V| | " + " | ".join(f"{int(v/1000)}k" for v in Vs) + " |",
                  "|" + "---|" * (len(Vs) + 1)]
        for g in Gs:
            cells = []
            for v in Vs:
                r = df[(df["backend"] == be) & (df["grid_size"] == g) & (df["num_nodes"] == v)]
                if r.empty:
                    cells.append("-")
                elif r.iloc[0]["status"] == "ok":
                    cells.append(f"{float(r.iloc[0]['peak_alloc_mb']):.0f}")
                else:
                    cells.append(r.iloc[0]["status"].upper())
            lines.append(f"| G={int(g)} | " + " | ".join(cells) + " |")
        lines.append("")
    _write(os.path.join(TAB, "feasibility_frontier.md"), "\n".join(lines))


def timing(path: str):
    """Real-model end-to-end mean per-case wall time, naive vs SparseFuse, vs G."""
    df = pd.read_csv(path)
    for c in ("grid_size", "proc_time_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    datasets = sorted(df["dataset"].unique())

    plt.figure(figsize=(6, 4))
    styles = {"re1-ob": "-", "re1-ss": "--"}
    for ds in datasets:
        for be in (NAIVE if NAIVE in df["kernel"].values else "naive", FUSED):
            s = df[(df.dataset == ds) & (df.kernel == be)].sort_values("grid_size")
            if s.empty:
                continue
            plt.plot(s["grid_size"], s["proc_time_s"], styles.get(ds, "-"),
                     marker="o", color=COLOR.get(be, "#333"),
                     label=f"{ds} {LABEL.get(be, be)}")
    plt.xlabel("grid size G"); plt.ylabel("mean per-case wall time (s)")
    plt.title("Ch5 — real GNN_KAN end-to-end time (naive vs SparseFuse)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "parity_timing.png"), dpi=140); plt.close()

    lines = ["# Real-model end-to-end wall time (mean s/case, 1 seed × 40 cases)", "",
             "On small RCA graphs (4-64 services) time is dominated by ICA features + graph",
             "construction; the kernel is a small slice (note: naive time barely grows with G).",
             "Isolated kernel timing is H2.", "",
             "| dataset | G | naive s | SF s | SF/naive |", "|---|---|---|---|---|"]
    for ds in datasets:
        for G in sorted(df[df.dataset == ds]["grid_size"].unique()):
            n = df[(df.dataset == ds) & (df.kernel == "naive") & (df.grid_size == G)]
            f = df[(df.dataset == ds) & (df.kernel == "sparsefuse") & (df.grid_size == G)]
            if n.empty or f.empty:
                continue
            nt = n.iloc[0]["proc_time_s"]; ft = f.iloc[0]["proc_time_s"]
            lines.append(f"| {ds} | {int(G)} | {nt:.3f} | {ft:.3f} | {ft/nt:.2f}× |")
    _write(os.path.join(TAB, "timing_summary.md"), "\n".join(lines))


def parity(paths: list[str]):
    """naive vs sparsefuse AC@k per dataset per G (mean over seeds) + max parity gap."""
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    for c in ("grid_size", "AC@1", "AC@3", "AC@5"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = (df.groupby(["dataset", "kernel", "grid_size"])[["AC@1", "AC@3", "AC@5"]]
             .mean().reset_index())

    lines = ["# H1 Parity — real GNN_KAN, naive kernel vs SparseFuse fused kernel",
             "",
             "Same B-spline GNN_KAN, RCA accuracy (AC@k, mean over 3 seeds, 40 cases/case-set).",
             "Parity = the kernel swap does not change accuracy.", ""]
    max_gap = 0.0
    for ds in sorted(agg["dataset"].unique()):
        lines += [f"## {ds}", "",
                  "| G | naive AC@1/@3/@5 | SF AC@1/@3/@5 | Δ AC@5 |",
                  "|---|---|---|---|"]
        for G in sorted(agg[agg["dataset"] == ds]["grid_size"].unique()):
            n = agg[(agg.dataset == ds) & (agg.kernel == "naive") & (agg.grid_size == G)]
            f = agg[(agg.dataset == ds) & (agg.kernel == "sparsefuse") & (agg.grid_size == G)]
            if n.empty or f.empty:
                continue
            n = n.iloc[0]; f = f.iloc[0]
            d5 = abs(n["AC@5"] - f["AC@5"]); max_gap = max(max_gap, d5)
            lines.append(
                f"| {int(G)} | {n['AC@1']:.3f}/{n['AC@3']:.3f}/{n['AC@5']:.3f} | "
                f"{f['AC@1']:.3f}/{f['AC@3']:.3f}/{f['AC@5']:.3f} | {d5:.3f} |")
        lines.append("")
    lines.append(f"**Max |Δ AC@5| across all (dataset,G) = {max_gap:.3f}** "
                 f"(0 ⇒ exact parity on the headline metric).")
    _write(os.path.join(TAB, "parity_summary.md"), "\n".join(lines))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  wrote {os.path.relpath(path, HERE)}")


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    eff = os.path.join(RES, "efficiency.csv")
    fea = os.path.join(RES, "feasibility.csv")
    if os.path.exists(eff):
        print("efficiency:"); efficiency(eff)
    if os.path.exists(fea):
        print("feasibility:"); feasibility(fea)
    parity_csvs = [os.path.join(RES, f) for f in
                   ("parity_re1ob.csv", "parity_re1ss.csv")]
    if any(os.path.exists(p) for p in parity_csvs):
        print("parity:"); parity(parity_csvs)
    tim = os.path.join(RES, "parity_timing.csv")
    if os.path.exists(tim):
        print("timing:"); timing(tim)
    print("done.")


if __name__ == "__main__":
    main()
