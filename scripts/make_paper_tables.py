"""Generate the result tables and figures of the paper from the run records.

Reads the records the campaign appends to runs/campaign_results.jsonl (the last record with
an average precision per run) and writes into results/:

  map_per_run.csv            the mAP of every reported run, the format of release_data/
  tables/fig_budget_curves.tikz    Figure 6, E.I over the label budget
  tables/fig_pretrain_curves.tikz  Figure 7, E.II, the four pretraining schemes
  tables/tab_benchmark.tex   Table 3, E.III at full supervision
  tables/tab_sweeps.tex      Table 4, PointPillars and VoxelNeXt over the label budget
  tables/tab_speed.tex       Table 5, inference cost, from results/speed.jsonl
  tables/tab_control.tex     Table C1, pretraining without the SeePerSea validation frames

Repeated cells are shown as the mean of their runs, with the range of the runs as error
bars in the figures and as a second line in the tables. Nothing here is typed by hand, so a
rerun of a cell cannot leave a table disagreeing with the records.

Usage:
  python make_paper_tables.py                          # from runs/campaign_results.jsonl
  python make_paper_tables.py --per-run release_data/results/map_per_run.csv
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "runs" / "campaign_results.jsonl"
OUT = ROOT / "results"
BUDGETS = (5, 10, 20, 50, 80, 100)
METRICS = ("bev@0.5", "3d@0.5", "bev@0.7", "3d@0.7")
CLASSES = ("small_boat", "medium_boat", "large_boat")
DETECTORS = {"mstnet": "MSTNet", "centerpoint": "CenterPoint", "pointpillar": "PointPillars",
             "voxelnext": "VoxelNeXt", "second": "SECOND", "voxel_rcnn": "Voxel R-CNN",
             "pv_rcnn": "PV-RCNN", "transfusion": "TransFusion", "dsvt": "DSVT, pillar",
             "dsvt_voxel": "DSVT, voxel", "ted_s": "TED-S", "ted_s_perclass": "TED-S",
             "lion_mamba": "LION-Mamba"}
BENCHMARK = [("PointPillars", "lang_pointpillars_2019"), ("SECOND", "yan_second_2018"),
             ("CenterPoint", "yin_center-based_2021"), ("Voxel R-CNN", "deng_voxel_2021"),
             ("PV-RCNN", "shipv-rcnn_2020"), ("VoxelNeXt", "VoxelNeXt_paper"),
             ("TransFusion", "bai_transfusion_2022"), ("DSVT, pillar", "dsvt_paper"),
             ("DSVT, voxel", "dsvt_paper"), ("TED-S", "wu_transformation-equivariant_2023"),
             ("LION-Mamba", "liu2024lion")]
EXPERIMENT = {"E.I": "E.I", "E.II": "E.II", "E.III": "E.III", "appendix-C": "Appendix C"}
DATASET = {"seepersea": "SeePerSea", "london": "Thames"}
RNG = r"\providecommand{\rng}[1]{\textcolor{black!60}{\scriptsize $#1$}}"


def pretraining(ckpt):
    """Pretraining scheme of a run from the name of its pretrained checkpoint."""
    if not ckpt:
        return "none"
    s = Path(ckpt).stem.replace("mstnet_ssl_", "")
    return {"seepersea": "in-domain", "london": "in-domain", "pohang": "cross-domain maritime",
            "kitti": "out-of-domain automotive", "pohang_seepersea": "combined",
            "pohang_london": "combined", "seepersea_noval": "in-domain without validation frames"}[s]


def from_records(path):
    """One row per reported run, from the campaign records; a later record of a run supersedes."""
    last = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("ap") and r.get("family") in EXPERIMENT:
            last[r["name"]] = r
    rows = []
    for r in last.values():
        stem, budget = r["cfg_stem"], 100
        for b in BUDGETS[:-1]:
            if stem.endswith(f"_{b}pct"):
                stem, budget = stem[: -len(f"_{b}pct")], b
        ap = r["ap"]
        m = ap.get("mAP", ap)
        row = dict(dataset=DATASET[r["ds"]], experiment=EXPERIMENT[r["family"]],
                   detector=DETECTORS[stem], pretraining=pretraining(r["ckpt"]),
                   label_budget_pct=budget,
                   run=3 if r["tag"].endswith("_s3") else 2 if r["tag"].endswith("_s2") else 1)
        row.update({f"mAP_{k}": round(m[k], 4) for k in METRICS})
        pc = ap.get("per_class")
        for c in CLASSES:
            for k in METRICS:
                row[f"AP_{c}_{k}"] = round(pc[c][k], 4) if pc else ""
        if budget == 100 and (row["detector"] == "CenterPoint" or
                              (row["detector"] == "MSTNet" and row["pretraining"] in ("none", "in-domain"))):
            row["experiment"] = "E.I, E.III"
        rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["experiment"], r["detector"], r["pretraining"],
                             r["label_budget_pct"], r["run"]))
    return rows


def cell(rows, ds, det, pre, b, k):
    """Values of one metric over the runs of one cell."""
    return [float(r[f"mAP_{k}"]) for r in rows if r["dataset"] == ds and r["detector"] == det
            and r["pretraining"] == pre and int(r["label_budget_pct"]) == b]


def span(v):
    return max(v) - min(v)


def tidy(line):
    """One space between the separators of empty cells, as in the manuscript."""
    while "&  &" in line:
        line = line.replace("&  &", "& &")
    return line


# ---------------------------------------------------------------- figures 6 and 7
AXES = [("ax0", "at={(0,0)},", "Thames, Bird's-eye view", "Thames", "bev@0.5", "mAP at $0.5$"),
        ("ax1", "at={(ax0.outer east)}, anchor=outer west,", "SeePerSea, Bird's-eye view", "SeePerSea", "bev@0.5", "AP at $0.5$"),
        ("ax2", "at={(ax0.outer south west)}, anchor=outer north west,", "Thames, 3D", "Thames", "3d@0.5", "mAP at $0.5$"),
        ("ax3", "at={(ax2.outer east)}, anchor=outer west,", "SeePerSea, 3D", "SeePerSea", "3d@0.5", "AP at $0.5$")]


def curves(rows, series, legend_columns):
    """Four axes (BEV and 3D on both datasets) of the given series over the label budget.

    series: (detector, pretraining, color, line style, mark, error-bar shift, legend label)."""
    fig = [r"\begin{tikzpicture}"]
    for name, pos, title, ds, k, ylabel in AXES:
        fig += [r"\begin{axis}[", f"  name={name},", f"  {pos}", r"  width=0.46\textwidth, height=0.38\textwidth,",
                f"  title={{{title}}}, title style={{font=\\small, yshift=-1mm}},",
                f"  xlabel={{Labels in \\unit{{\\percent}}}}, ylabel={{{ylabel}}},",
                "  xmin=0, xmax=105, ymin=0, ymax=100,", "  xtick={0,20,40,60,80,100},",
                r"  xticklabel style={font=\small}, yticklabel style={font=\small},",
                "  label style={font=\\small}, tick align=outside, grid=none,",
                "  every axis plot/.append style={line width=0.7pt, mark size=1.5pt},", "]"]
        for det, pre, c, ls, mk, shift, _ in series:
            v = {b: cell(rows, ds, det, pre, b, k) for b in BUDGETS}
            fig.append(f"\\addplot[{c}, {ls}, mark={mk}] coordinates {{"
                       + " ".join(f"({b},{st.mean(v[b]):.2f})" for b in BUDGETS) + "};")
            eb = [f"({b + shift:g},{st.mean(v[b]):.2f}) +=(0,{max(v[b]) - st.mean(v[b]):.2f}) "
                  f"-=(0,{st.mean(v[b]) - min(v[b]):.2f})" for b in BUDGETS if len(v[b]) > 1]
            if eb:
                fig.append("\\addplot[forget plot, draw=none, mark=none, error bars/.cd, y dir=both, y explicit, "
                           f"error mark options={{rotate=90, mark size=2pt, line width=0.7pt, {c}}}, "
                           f"error bar style={{solid, line width=0.7pt, {c}}}] coordinates {{" + " ".join(eb) + "};")
        fig.append(r"\end{axis}")
    fig += [r"\begin{axis}[hide axis, scale only axis, width=1mm, height=1mm,",
            "  xmin=0, xmax=1, ymin=0, ymax=1,",
            "  at={(ax2.outer south east)}, anchor=north, yshift=-5mm,",
            f"  legend style={{font=\\small, draw=black, at={{(0.5,1)}}, anchor=north, legend columns={legend_columns}, column sep=6pt, inner sep=3pt}},",
            "  legend cell align=left,]"]
    for _, _, c, ls, mk, _, label in series:
        fig += [f"\\addlegendimage{{{ls}, mark={mk}, line width=0.7pt, mark size=1.5pt, {c}}}",
                f"\\addlegendentry{{{label}}}"]
    fig += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(fig) + "\n"


def budget_curves(rows):
    return curves(rows, [("MSTNet", "in-domain", "red!80!black", "solid", "*", 0.0, "MSTNet, pretrained"),
                         ("MSTNet", "none", "black!60", "dashed", "square*", -1.5, "MSTNet, random"),
                         ("CenterPoint", "none", "blue!65!black", "dotted", "triangle*", 1.5, "CenterPoint")], 3)


def pretrain_curves(rows):
    return curves(rows, [("MSTNet", "in-domain", "red!80!black", "solid", "*", 0.0, "In-domain"),
                         ("MSTNet", "cross-domain maritime", "blue!65!black", "dashed", "square*", 0.0, "Cross-domain maritime"),
                         ("MSTNet", "out-of-domain automotive", "black!60", "dotted", "triangle*", 0.0, "Out-of-domain automotive"),
                         ("MSTNet", "combined", "orange!90!black", "dashdotted", "diamond*", 0.0, "Combined")], 2)


# ---------------------------------------------------------------- table 3
def benchmark(rows):
    """E.III at full supervision, all four metric combinations on both datasets."""
    entries = [(f"{n}~\\cite{{{c}}}", n, "none") for n, c in BENCHMARK]
    entries += [None, ("MSTNet, random", "MSTNet", "none"), ("MSTNet, pretrained", "MSTNet", "in-domain")]
    cols = [(ds, k) for ds in ("Thames", "SeePerSea") for k in METRICS]
    vals = {e[0]: [cell(rows, ds, e[1], e[2], 100, k) for ds, k in cols] for e in entries if e}
    best = [max(st.mean(v[i]) for v in vals.values()) for i in range(len(cols))]
    lines = []
    for e in entries:
        if e is None:
            lines.append(r"\midrule")
            continue
        v = vals[e[0]]
        out = []
        for i, x in enumerate(v):
            s = f"{st.mean(x):.2f}"
            out.append(f"$\\mathbf{{{s}}}$" if abs(st.mean(x) - best[i]) < 1e-9 else f"${s}$")
        lines.append(f"{e[0]} & " + " & ".join(out) + r" \\")
        if any(len(x) > 1 for x in v):
            lines.append(r"\quad\textcolor{black!60}{\scriptsize range} & "
                         + " & ".join(f"\\rng{{{span(x):.2f}}}" if len(x) > 1 else "" for x in v) + r" \\")
    head = r"""\begin{table}
\centering
\caption{Experiment~E.III. Mean average precision (mAP) at $40$ recall positions in BEV and in 3D, at both overlap thresholds and at \qty{100}{\percent} of the labels. \emph{MSTNet, pretrained} is the pipeline of Section~\ref{sec:method:MSTNet} with the in-domain pretraining of Section~\ref{sec:method:ssl}, and the further pretraining schemes at full supervision are given in Figure~\ref{fig:pretrain-curves}. The SeePerSea values of VoxelNeXt and all MSTNet values are means of three runs, and the line below each of these entries gives the range of the three runs; all other values are single runs. The highest value of each column is in bold.}\label{tab:benchmark}
\small
""" + RNG + r"""
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{Thames} & \multicolumn{4}{c}{SeePerSea} \\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
 & \multicolumn{2}{c}{overlap $0.5$} & \multicolumn{2}{c}{overlap $0.7$} & \multicolumn{2}{c}{overlap $0.5$} & \multicolumn{2}{c}{overlap $0.7$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Detector & BEV~$\uparrow$ & 3D~$\uparrow$ & BEV~$\uparrow$ & 3D~$\uparrow$ & BEV~$\uparrow$ & 3D~$\uparrow$ & BEV~$\uparrow$ & 3D~$\uparrow$ \\
\midrule
"""
    return head + "\n".join(tidy(l) for l in lines) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"


# ---------------------------------------------------------------- table 4
def sweeps(rows):
    """PointPillars and VoxelNeXt over the label budget, mAP at an overlap of 0.5."""
    lines = []
    for ds in ("Thames", "SeePerSea"):
        first = True
        for det in ("PointPillars", "VoxelNeXt"):
            for i, (k, label) in enumerate((("bev@0.5", "BEV"), ("3d@0.5", "3D"))):
                v = [cell(rows, ds, det, "none", b, k) for b in BUDGETS]
                lines.append(f"{ds if first else ''} & {det if i == 0 else ''} & {label} & "
                             + " & ".join(f"${st.mean(x):.2f}$" for x in v) + r" \\")
                first = False
                if any(len(x) > 1 for x in v):
                    lines.append(r" & & \textcolor{black!60}{\scriptsize range} & "
                                 + " & ".join(f"\\rng{{{span(x):.2f}}}" if len(x) > 1 else "" for x in v) + r" \\")
        if ds == "Thames":
            lines.append(r"\midrule")
    head = r"""\begin{table}
\centering
\caption{Label budget sweeps of PointPillars and VoxelNeXt in Experiment~E.III. Mean average precision (mAP) at $40$ recall positions and an overlap of $0.5$. The SeePerSea values of VoxelNeXt are means of three runs, and the line below each of these entries gives the range of the three runs; all other values are single runs.}\label{tab:sweeps}
\small
""" + RNG + r"""
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lllrrrrrr@{}}
\toprule
Dataset & Detector & Metric & \qty{5}{\percent} & \qty{10}{\percent} & \qty{20}{\percent} & \qty{50}{\percent} & \qty{80}{\percent} & \qty{100}{\percent} \\
\midrule
"""
    return head + "\n".join(tidy(l) for l in lines) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"


# ---------------------------------------------------------------- table 5
def speed(path):
    """Parameter count, latency and peak memory per entry, from the speed records."""
    rec = {}
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            rec[(r["ds"], DETECTORS[r["model"]])] = r
    lines = []
    for name in [n for n, _ in BENCHMARK] + [None, "MSTNet"]:
        if name is None:
            lines.append(r"\midrule")
            continue
        t, s = rec[("london", name)], rec[("seepersea", name)]
        lines.append(f"{name} & ${t['params_M']:.2f}$ & ${t['median_ms']:.1f}$ & ${t['p90_ms']:.1f}$ & "
                     f"${t['peak_mem_MB']:.0f}$ & ${s['median_ms']:.1f}$ & ${s['p90_ms']:.1f}$ & ${s['peak_mem_MB']:.0f}$ \\\\")
    n = next(iter(rec.values()))["n_frames"]
    head = r"""\begin{table}
\centering
\caption{Inference cost of the entries of Table~\ref{tab:benchmark}. Reported are the parameter count, the peak GPU memory, and the median and 90th percentile of the inference time over $""" + str(n) + r"""$ frames of the evaluation split after $20$ warm-up frames, at a batch size of one on one GeForce RTX 4090. The inference time excludes reading and voxelizing a frame in the data loader. The parameter count refers to the Thames configuration, and both MSTNet initializations share the architecture of the row MSTNet.}\label{tab:speed}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Thames} & \multicolumn{3}{c}{SeePerSea} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Configuration & \makecell{Param.\\/\num{e6}~$\downarrow$} & \makecell{median\\\unit{\milli\second}~$\downarrow$} & \makecell{p90\\\unit{\milli\second}~$\downarrow$} & \makecell{peak\\\unit{\mega\byte}~$\downarrow$} & \makecell{median\\\unit{\milli\second}~$\downarrow$} & \makecell{p90\\\unit{\milli\second}~$\downarrow$} & \makecell{peak\\\unit{\mega\byte}~$\downarrow$} \\
\midrule
"""
    return head + "\n".join(tidy(l) for l in lines) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"


# ---------------------------------------------------------------- table C1
def control(rows):
    """MSTNet on SeePerSea pretrained with and without the validation frames."""
    num = lambda x: f"\\num[retain-explicit-plus]{{{x:+.2f}}}" if x > 0 else f"\\num{{{x:.2f}}}"
    lines = []
    for b in (5, 20, 100):
        o = [round(st.mean(cell(rows, "SeePerSea", "MSTNet", "in-domain", b, k)), 2) for k in ("bev@0.5", "3d@0.5")]
        c = [round(st.mean(cell(rows, "SeePerSea", "MSTNet", "in-domain without validation frames", b, k)), 2)
             for k in ("bev@0.5", "3d@0.5")]
        lines.append(f"\\qty{{{b}}}{{\\percent}} & \\num{{{o[0]:.2f}}} & \\num{{{o[1]:.2f}}} & \\num{{{c[0]:.2f}}} & "
                     f"\\num{{{c[1]:.2f}}} & {num(c[0] - o[0])} & {num(c[1] - o[1])} \\\\")
    head = r"""\begin{table}[!hb]
\centering
\caption{MSTNet on SeePerSea after in-domain pretraining with and without the validation frames, as mean average precision (mAP) at \num{40} recall positions and an overlap of \num{0.5}. The values of the original corpus are means of three runs, and the values without the validation frames are single runs. The difference is the value without the validation frames minus the value of the original corpus.}\label{tab:control}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Original corpus} & \multicolumn{2}{c}{Without validation} & \multicolumn{2}{c}{Difference} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Budget & BEV & 3D & BEV & 3D & BEV & 3D \\
\midrule
"""
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--per-run", type=Path, help="build the tables from an existing per-run CSV")
    ap.add_argument("--speed", type=Path, default=OUT / "speed.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    tables = a.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if a.per_run:
        rows = list(csv.DictReader(open(a.per_run)))
    else:
        rows = from_records(a.records)
        with open(a.out / "map_per_run.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} runs -> {a.out / 'map_per_run.csv'}")
    out = {"fig_budget_curves.tikz": budget_curves(rows), "fig_pretrain_curves.tikz": pretrain_curves(rows),
           "tab_benchmark.tex": benchmark(rows), "tab_sweeps.tex": sweeps(rows), "tab_control.tex": control(rows)}
    if a.speed.exists():
        out["tab_speed.tex"] = speed(a.speed)
    else:
        print(f"no speed records at {a.speed}, Table 5 skipped")
    for name, text in out.items():
        (tables / name).write_text(text)
        print(f"-> {tables / name}")


if __name__ == "__main__":
    main()
