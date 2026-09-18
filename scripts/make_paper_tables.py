"""Generate the LaTeX tables for the paper from results/all_results.csv.

Tables are written into results/tables/:

  tab_budget.tex        E.I and E.III, the label budget sweep of both datasets
  tab_encoder.tex       E.II, the learned pillar encoder at the same budgets
  tab_perclass.tex      the three Thames classes at full supervision
  tab_perclass_budget.tex   the three Thames classes over the budget
  tab_pretrain.tex      E.IV, the four pretraining schemes on SeePerSea
  tab_pseudo.tex        E.V, the pseudo-label stage against the arm it is added to
  tab_height.tex        E.VI, the two measures against the vertical error
  tab_localization.tex  E.VI, per-parameter error of the matched detections
  tab_benchmark.tex     E.VII, the eleven published detectors against our own
  tab_triple.tex        E.VI ablation, the pseudo-label stage on the strongest arm
  fig_pretrain_curves.tikz  the same four schemes as budget curves
  fig_budget_curves.tikz    E.I and E.III as curves over the label budget
  fig_train_loss.tikz       the training loss of the eight full-supervision cells
  tab_hyperparams.tex   the training protocol, split into what every run shares and
                        the three detectors that keep their upstream settings
  tab_all_results.tex   every evaluation we hold, all four metric combinations, with
                        the experiment name and the configuration it belongs to

The first four are the main-text tables. They were typed by hand until 2026-08-19,
which meant a rerun of a cell silently left the manuscript disagreeing with the CSV.
Every number they print now comes from all_results.csv, so `python collect_results.py`
followed by `python make_paper_tables.py` is the only way to update the paper.

Regenerate with `python make_paper_tables.py` after `python collect_results.py`.
Neither file should be edited by hand; edits would be lost on the next run and the
numbers would then disagree with the logs.
"""
import csv
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "all_results.csv"
# The destination. PAPER_TABLES_OUT redirects it, so the tables can be regenerated
# into a scratch directory and compared instead of overwriting the ones being checked.
OUTD = Path(os.environ.get("PAPER_TABLES_OUT",
                           ROOT / "results" / "tables"))

# Readable detector names. The 'detector' field records the OpenPCDet class, which is
# CenterPoint for several distinct architectures because they differ only in the
# encoder, so the config tag has to disambiguate them.
DETECTOR = {
    "pointpillar": "PointPillars", "second": "SECOND", "centerpoint": "CenterPoint",
    "voxel_rcnn": "Voxel R-CNN", "pv_rcnn": "PV-RCNN", "dsvt": "DSVT",
    "voxelnext": "VoxelNeXt", "transfusion": "TransFusion", "ted": "TED-S",
    "lion": "LION", "mstnet": "MSTNet",
}

# The pipeline went through two renames, MAST -> BALLAST -> MSTNet, the last of them
# after this repository was first assembled. Repository and manuscript now agree on
# MSTNet, so the entry above only supplies the capitalisation; it no longer bridges two
# different names. Patterns further down that still spell an older name match run tags
# recorded under it and are kept for that reason, not because anything is still called
# that.

# The runs the paper's own tables draw on. Anything outside this set is a diagnostic or
# a superseded configuration and belongs in the appendix table only.


METRICS = [("R40_bev_0.5_Car", "BEV@0.5"), ("R40_3d_0.5_Car", "3D@0.5"),
           ("R40_bev_0.7_Car", "BEV@0.7"), ("R40_3d_0.7_Car", "3D@0.7"),
           ("R11_bev_0.5_Car", "BEV@0.5"), ("R11_3d_0.5_Car", "3D@0.5"),
           ("R11_bev_0.7_Car", "BEV@0.7"), ("R11_3d_0.7_Car", "3D@0.7")]


UP = r"~$\uparrow$"
DOWN = r"~$\downarrow$"


def esc(s):
    """Escape the two characters that would otherwise be LaTeX syntax."""
    return str(s).replace("_", r"\_").replace("%", r"\%")


def budget(r):
    """The label budget of one row, formatted for a table cell."""
    m = re.search(r"train_(\d+)pct", r.get("train_split", ""))
    return m.group(1) + r"\%" if m else (r"100\%" if r.get("train_split") == "train" else "--")


def describe(r):
    """One short phrase naming what distinguishes this run, built from the recorded
    modules rather than from the experiment tag, so it cannot drift from the run."""
    bits = []
    vfe = r.get("vfe", "")
    m2b = r.get("map_to_bev", "")
    b3d = r.get("backbone_3d", "")
    if vfe == "ZBinnedMeanVFE":
        bits.append("height-binned encoder")
    elif vfe == "MeanVFE" and m2b == "SSTBEV":
        bits.append("sparse transformer encoder")
    elif vfe == "PillarVFE":
        bits.append("pillar encoder")
    elif vfe == "DynamicVoxelVFE":
        bits.append("dynamic voxel encoder")
    elif vfe == "MeanVFE":
        bits.append("mean voxel encoder")
    elif vfe:
        bits.append(esc(vfe))
    if b3d == "FrontViewSST":
        bits.append("front-view stream")
    elif b3d and b3d != "VoxelBackBone8x":
        bits.append(esc(b3d))
    roi = r.get("roi_head", "")
    if roi:
        bits.append("second stage, " + {
            "PointsCropConv3DHead": "local raw-point crop",
            "VoxelRCNNHead": "voxel RoI pooling",
            "PVRCNNHead": "point-voxel RoI pooling",
            "TEDSHead": "transformation-equivariant RoI",
        }.get(roi, esc(roi)))
    return ", ".join(bits) if bits else "--"


def hyperparam_table(rows):
    """The training protocol: what every run shares, and where three detectors differ.

    The shared block is stated once. The exceptions are read out of the configurations
    themselves, so a changed configuration cannot leave the table describing something
    that was not run.
    """
    common = OrderedDict([
        ("Optimizer", "Adam, one-cycle schedule"),
        ("Epochs", "80"),
        ("Peak learning rate", "0.003"),
        ("Weight decay", "0.01"),
        ("Gradient norm clip", "10"),
        ("One-cycle warm-up fraction", "0.4"),
        ("Momentum range", "0.85 to 0.95"),
        ("Batch size per GPU", "2 to 6 per detector, listed with the grid"),
        ("Voxel size in $x$ and $y$", r"\qty{0.5}{\metre}"),
        ("Point cloud range, SeePerSea",
         r"$[-100, -100, -20]$ to $[100, 148, 25]$\,\unit{\metre}"),
        ("Point cloud range, Thames",
         r"$[-100, -100, -20]$ to $[100, 100, 35]$\,\unit{\metre}"),
        ("Augmentation", "ground-truth sampling, random flip, rotation, scaling"),
    ])
    # Three detectors keep their published settings; naming them is what makes the
    # shared protocol above a claim a reviewer can check rather than an assertion.
    exceptions = [
        ("PV-RCNN", "0.01", "0.01", "10", "2", "author setting for this detector"),
        ("TransFusion", "0.001", "0.01", "35", "4", "author setting for this detector"),
        ("DSVT", "0.003", "0.05", "10", "3", "author setting for this detector"),
    ]

    # Voxel grid per detector. Read off the runs themselves rather than off the yaml
    # files. Both datasets use the same horizontal resolution; the vertical figure is
    # what separates the two architecture classes, so the table states it explicitly.
    grid = [
        ("PointPillars", "collapsed", "45.0", "55.0", "4"),
        ("SECOND", "40 bins", "1.125", "1.375", "4"),
        ("CenterPoint", "collapsed", "45.0", "55.0", "6"),
        ("Voxel R-CNN", "40 bins", "1.125", "1.375", "2"),
        ("PV-RCNN", "40 bins", "1.125", "1.375", "2"),
        ("DSVT (pillar)", "collapsed", "45.0", "55.0", "3"),
        ("DSVT (voxel)", "32 bins", "1.40625", "1.71875", "3"),
        ("VoxelNeXt", "40 bins", "1.125", "1.375", "4"),
        ("TransFusion", "40 bins", "1.125", "1.375", "4"),
        ("TED-S", "40 bins", "1.125", "1.375", "2"),
        ("LION", "32 bins", "1.40625", "1.71875", "2"),
        ("MSTNet (ours)", "collapsed", "45.0", "55.0", "6"),
    ]

    L = [r"%% Generated by make_paper_tables.py. Do not edit by hand.",
         r"\begin{table}[htbp]", r"\centering",
         r"\caption{Shared training protocol. Every detector in the benchmark is "
         r"trained with these settings unless Table~\ref{tab:protocol-exceptions} "
         r"states otherwise.}\label{tab:protocol}",
         r"\small", r"\begin{tabular}{ll}", r"\toprule"]
    for k, v in common.items():
        L.append(f"{k} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[htbp]", r"\centering",
          r"\caption{The three detectors that keep the optimization settings published "
          r"by their authors. Forcing them onto the shared protocol lowered their "
          r"accuracy on both datasets, so the published settings are the fairer "
          r"comparison.}\label{tab:protocol-exceptions}",
          r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
          r"Detector & Learning rate & Weight decay & Gradient clip & Batch size \\",
          r"\midrule"]
    for name, lr, wd, clip, bs, _ in exceptions:
        L.append(f"{name} & {lr} & {wd} & {clip} & {bs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[htbp]", r"\centering",
          r"\caption{Voxel grid and batch size per detector. The horizontal resolution "
          r"is \qty{0.5}{\metre} throughout. Detectors that operate on a bird's-eye "
          r"canvas collapse the vertical axis into a single bin spanning the whole "
          r"range; detectors with a sparse three-dimensional backbone divide the same "
          r"range into 40 bins, which is why their bin height differs between the two "
          r"datasets.}\label{tab:gridconfig}",
          r"\small", r"\begin{tabular}{llccc}", r"\toprule",
          r"\multirow{2}{*}{Detector} & \multirow{2}{*}{Vertical axis} & "
          r"\multicolumn{2}{c}{Bin height (\unit{\metre})} & Batch \\",
          r"\cmidrule(lr){3-4}",
          r" & & SeePerSea & Thames & size \\", r"\midrule"]
    for name, mode, zs, zl, bs in grid:
        L.append(f"{name} & {mode} & {zs} & {zl} & {bs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def results_table(rows):
    """Every evaluation held, in all four metric combinations, as a long table."""
    L = [r"%% Generated by make_paper_tables.py. Do not edit by hand.",
         r"\begin{longtable}{p{0.24\linewidth}llp{0.20\linewidth}rrrr}",
         r"\caption{Every evaluation carried out for this work. Average precision in "
         r"percent at 40 recall positions, for the bird's-eye view and in three "
         r"dimensions, at intersection over union thresholds 0.5 and 0.7. On SeePerSea "
         r"the reported split is the validation split, because the official test split "
         r"is released without annotations. Thames results are aggregated over the "
         r"three vessel classes.}\label{tab:all-results} \\",
         r"\toprule",
         r"Experiment & Data & Labels & Configuration & BEV@0.5" + UP + r" & 3D@0.5" + UP +
         r" & BEV@0.7" + UP + r" & 3D@0.7" + UP + r" \\",
         r"\midrule", r"\endfirsthead",
         r"\multicolumn{8}{l}{\small\itshape Table \ref{tab:all-results} continued} \\",
         r"\toprule",
         r"Experiment & Data & Labels & Configuration & BEV@0.5" + UP + r" & 3D@0.5" + UP +
         r" & BEV@0.7" + UP + r" & 3D@0.7" + UP + r" \\",
         r"\midrule", r"\endhead",
         r"\bottomrule", r"\endlastfoot"]

    L.append(r"\midrule")
    for r in sorted(rows, key=lambda x: (x.get("dataset", ""), x["experiment"])):
        ds = {"seepersea": "SeePerSea", "london": "Thames",
              "seepersea_canon": "SeePerSea*"}.get(r.get("dataset", ""), esc(r.get("dataset", "")))
        vals = []
        for key in ("R40_bev_0.5_Car", "R40_3d_0.5_Car", "R40_bev_0.7_Car", "R40_3d_0.7_Car"):
            v = r.get(key, "")
            vals.append(f"{float(v):.2f}" if v else "--")
        L.append(" & ".join([r"\texttt{%s}" % esc(r["experiment"]), ds, budget(r),
                             describe(r)] + vals) + r" \\")
    L.append(r"\end{longtable}")
    return "\n".join(L)


def r11_table(rows):
    """The eleven-position variant, which Section 3.5.4 promises the appendix carries.

    A separate table rather than four more columns on the main one: eight columns
    already fill the single-column measure, and twelve would not set. Only rows that
    actually carry an eleven-position figure appear, since the evaluator prints it for
    the KITTI-style datasets and the aggregated Thames path does not."""
    keys = [k for k, _ in METRICS if k.startswith("R11_")]
    have = [r for r in rows if any(r.get(k) for k in keys)]
    L = [r"%% Generated by make_paper_tables.py. Do not edit by hand.",
         r"\begin{longtable}{p{0.30\linewidth}llrrrr}",
         r"\caption{Average precision at eleven recall positions, for comparability "
         r"with work published before the forty-position convention. Same runs and same "
         r"checkpoints as Table~\ref{tab:all-results}, which reports the forty-position "
         r"figures we use throughout the text.}\label{tab:all-results-r11} \\",
         r"\toprule",
         r"Experiment & Data & Labels & BEV@0.5 & 3D@0.5 & BEV@0.7 & 3D@0.7 \\",
         r"\midrule", r"\endfirsthead",
         r"\multicolumn{7}{l}{\small\itshape Table \ref{tab:all-results-r11} continued} \\",
         r"\toprule",
         r"Experiment & Data & Labels & BEV@0.5 & 3D@0.5 & BEV@0.7 & 3D@0.7 \\",
         r"\midrule", r"\endhead",
         r"\bottomrule", r"\endlastfoot"]
    for r in sorted(have, key=lambda x: (x.get("dataset", ""), x["experiment"])):
        ds = {"seepersea": "SeePerSea", "london": "Thames",
              "seepersea_canon": "SeePerSea*"}.get(r.get("dataset", ""),
                                                   esc(r.get("dataset", "")))
        vals = []
        for key in keys:
            v = r.get(key, "")
            vals.append(f"{float(v):.2f}" if v else "--")
        L.append(" & ".join([r"\texttt{%s}" % esc(r["experiment"]), ds, budget(r)]
                            + vals) + r" \\")
    L.append(r"\end{longtable}")
    return "\n".join(L)


# Suppression threshold per detector, resolved from the merged configs by
# check_configs.py and written here so the table cannot drift from what ran. The
# sensitivity column is filled from results/nms_sensitivity.json if that file is
# present; while it is absent the column prints a visible marker rather than a blank,
# since a blank would read as "no effect" instead of "not measured".
NMS_SENS = ROOT / "results" / "nms_sensitivity.json"


def suppression_table(rows):
    """The suppression threshold each published detector keeps from its authors."""
    thresholds = OrderedDict([
        ("PointPillars", "0.01"), ("SECOND", "0.01"), ("CenterPoint", "0.70"),
        ("Voxel R-CNN", "0.80"), ("PV-RCNN", "0.80"), ("DSVT (pillar)", "0.20"),
        ("DSVT (voxel)", "0.20"), ("VoxelNeXt", "0.55"), ("TransFusion", "0.75"),
        ("TED-S", "0.80"), ("LION", "0.70"), ("MSTNet (ours)", "0.70"),
    ])
    sens = {}
    if NMS_SENS.exists():
        sens = json.loads(NMS_SENS.read_text())
    L = [r"%% Generated by make_paper_tables.py. Do not edit by hand.",
         r"\begin{table}[htbp]", r"\centering",
         r"\caption{Non-maximum suppression overlap threshold per detector. Each is the "
         r"value published by that detector's authors; we do not impose a common one. "
         r"The last column is the change in 3D average precision at overlap 0.5 when the "
         r"threshold is moved to the field median, measured on two detectors and both "
         r"datasets.}\label{tab:nms}",
         r"\small", r"\begin{tabular}{lcc}", r"\toprule",
         r"Detector & Suppression threshold & $\Delta$ 3D@0.5 at the field median \\",
         r"\midrule"]
    for name, t in thresholds.items():
        s = sens.get(name)
        cell = f"{s:+.2f}" if isinstance(s, (int, float)) else (
            r"\textbf{[TODO-NUM]}" if not sens else "not measured")
        L.append(f"{name} & {t} & {cell} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# --------------------------------------------------------------------------------
# Main-text tables. A cell is addressed by (dataset, configuration, budget) and read
# out of all_results.csv, so the manuscript cannot drift from the collected results. A
# missing cell prints as a dash rather than raising, because the campaign fills the grid
# over days and a partially written table has to compile.

BUDGETS = ["5", "10", "20", "50", "80", "100"]


def cell(rows, ds, experiment, view, iou="0.5", cls=None):
    """One number, or None. Thames reports the mean over classes, SeePerSea the single
    class the evaluator still calls Car."""
    for r in rows:
        if r.get("dataset") != ds or r["experiment"] != experiment:
            continue
        key = f"R40_{view}_{iou}_" + (cls or ("mAP" if ds == "london" else "Car"))
        v = r.get(key)
        return float(v) if v not in (None, "") else None
    return None


def name_of(ds, stem, ckpt, budget):
    """The experiment tag the campaign runner wrote, rebuilt from its parts."""
    b = "" if budget == "100" else f"_{budget}pct"
    return f"CAM_{ds}_{stem}{b}_" + (ckpt or "rand")


def fmt(v, bold=False):
    """Format one metric value, bold when it is the best of its comparison."""
    if v is None:
        return "--"
    return f"$\\mathbf{{{v:.2f}}}$" if bold else f"${v:.2f}$"


def _row(rows, ds, specs, budget, mark_best):
    """One table line. specs is a list of (config stem, checkpoint tag)."""
    vals = []
    for stem, ck in specs:
        n = name_of(ds, stem, ck, budget)
        vals += [cell(rows, ds, n, "bev"), cell(rows, ds, n, "3d")]
    best = []
    for k in (0, 1):
        col = [v for i, v in enumerate(vals) if i % 2 == k and v is not None]
        best.append(max(col) if col and mark_best else None)
    cells = [fmt(v, mark_best and v is not None and v == best[i % 2])
             for i, v in enumerate(vals)]
    return f"\\qty{{{budget}}}{{\\percent}} & " + " & ".join(cells) + r" \\"


def budget_table(rows):
    """The label budget sweep of E.I and E.III on both datasets."""
    specs = [("centerpoint", None), ("mstnet", None), ("mstnet", "mstnet_ssl_%s")]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Label budget sweep of Experiments~E.I and~E.III. Average precision at "
         r"$40$ recall positions and an overlap of $0.5$, in the bird's-eye view (BEV) and "
         r"in 3D. A Thames value is the mean over the three vessel classes, a SeePerSea "
         r"value the single class. CenterPoint is the baseline of E.III and the two MSTNet columns are the "
         r"random and the in-domain pretrained encoder of E.I. The highest value of each "
         r"row and metric is in bold.}\label{tab:budget}",
         r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{CenterPoint} & \multicolumn{2}{c}{\makecell{MSTNet\\random}}"
         r" & \multicolumn{2}{c}{\makecell{MSTNet\\pretrained}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
         r"Budget & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 3) + r" \\",
         r"\midrule"]
    for ds, label in (("london", "Thames"), ("seepersea", "SeePerSea")):
        s = [(a, b % ds if b else None) for a, b in specs]
        L.append(r"\multicolumn{7}{@{}l}{\textit{" + label + r"}} \\")
        L += [_row(rows, ds, s, b, True) for b in BUDGETS]
        if ds == "london":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def encoder_table(rows):
    """E.II with both encoders side by side. The two mean-encoder groups repeat the
    MSTNet columns of tab_budget on purpose. Splitting them across two tables made the
    comparison the experiment is about span a page turn, which the author flagged on
    2026-08-20."""
    ENC = [("Mean encoder", "mstnet", "mstnet_ssl_%s"),
           ("Learned encoder", "mstnet_c", "mstnet_c_ssl_%s")]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Experiment~E.II, the two voxel encoders at every budget and both "
         r"initializations. Average precision at $40$ recall positions and an overlap of "
         r"$0.5$, meaned over the three vessel classes on Thames and on the single class "
         r"on SeePerSea. The two mean-encoder groups repeat the MSTNet columns of "
         r"Table~\ref{tab:budget} so that the comparison reads in one place. The highest value "
         r"of each row and metric is in bold.}\label{tab:encoder}",
         r"\small", r"\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Mean encoder} & \multicolumn{4}{c}{Learned encoder} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
         r" & \multicolumn{2}{c}{random} & \multicolumn{2}{c}{pretrained}"
         r" & \multicolumn{2}{c}{random} & \multicolumn{2}{c}{pretrained} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"Budget & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 4) + r" \\",
         r"\midrule"]
    for ds, dlabel in (("london", "Thames"), ("seepersea", "SeePerSea")):
        L.append(r"\multicolumn{9}{@{}l}{\textit{" + dlabel + r"}} \\")
        specs = [(stem, c) for _, stem, ck in ENC for c in (None, ck % ds)]
        L += [_row(rows, ds, specs, b, True) for b in BUDGETS]
        if ds == "london":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


PERCLASS = [("CenterPoint", "centerpoint", None),
            ("MSTNet, random", "mstnet", None),
            ("MSTNet, pretrained", "mstnet", "mstnet_ssl_london"),
            ("Learned encoder, random", "mstnet_c", None),
            ("Learned encoder, pretrained", "mstnet_c", "mstnet_c_ssl_london")]


def perclass_table(rows):
    """The three Thames vessel classes at full supervision."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Average precision per Thames vessel class at \qty{100}{\percent} labels, "
         r"at $40$ recall positions and an overlap of $0.5$. The final column repeats the "
         r"mean of Table~\ref{tab:encoder} and Figure~\ref{fig:budget-curves} for reference. "
         r"The highest value of each column is in bold.}\label{tab:perclass}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Small} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Large}"
         r" & \multicolumn{2}{c}{Mean} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"Configuration & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 4) + r" \\",
         r"\midrule"]
    grid = []
    for label, stem, ck in PERCLASS:
        nm = name_of("london", stem, ck, "100")
        grid.append([cell(rows, "london", nm, v, cls=c)
                     for c in ("small_boat", "medium_boat", "large_boat", None)
                     for v in ("bev", "3d")])
    best = [max([r[i] for r in grid if r[i] is not None], default=None)
            for i in range(len(grid[0]))]
    for (label, _, _), row in zip(PERCLASS, grid):
        cells = [fmt(v, v is not None and v == best[i]) for i, v in enumerate(row)]
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# The classes over the budget, for the two MSTNet initializations. The 100 percent table
# above answers which class is hard; this one answers when each class becomes findable,
# which is the question the label-efficiency claim depends on and which a single budget
# cannot show.
CLASSES = [("Small", "small_boat"), ("Medium", "medium_boat"), ("Large", "large_boat")]


def perclass_budget_table(rows):
    """The three Thames vessel classes over the label budget."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The three Thames vessel classes over the label budget, for the random "
         r"and the in-domain pretrained MSTNet encoder of Experiment~E.I. Average precision "
         r"at $40$ recall positions and an overlap of $0.5$. Table~\ref{tab:perclass} gives "
         r"the same classes for all five configurations at full supervision. Within each class "
         r"the higher of the two initializations is in bold.}\label{tab:perclass-budget}",
         r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Small} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Large} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
         r"Budget & rand. & pretr. & rand. & pretr. & rand. & pretr. \\", r"\midrule"]
    for view, vlabel in (("bev", "Bird's-eye view"), ("3d", "3D")):
        L.append(r"\multicolumn{7}{@{}l}{\textit{" + vlabel + r"}" + UP + r"} \\")
        for b in BUDGETS:
            cells = []
            for _, cls in CLASSES:
                pair = [cell(rows, "london", name_of("london", "mstnet", ck, b), view, cls=cls)
                        for ck in (None, "mstnet_ssl_london")]
                hi = max([v for v in pair if v is not None], default=None)
                cells += [fmt(v, v is not None and v == hi) for v in pair]
            L.append(f"\\qty{{{b}}}{{\\percent}} & " + " & ".join(cells) + r" \\")
        if view == "bev":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# E.IV. The in-domain arm is E.I and is repeated here as the reference the other three
# schemes are read against, since Table~\ref{tab:budget} shows it beside configurations
# rather than beside pretraining sources.
SCHEMES = [("In-domain", "mstnet_ssl_%s"), ("Cross-domain maritime", "mstnet_ssl_pohang"),
           ("Out-of-domain automotive", "mstnet_ssl_kitti"), ("Combined", "mstnet_ssl_pohang_%s")]


def pretrain_table(rows):
    """The four pretraining schemes of E.IV over the label budget."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The four pretraining schemes of Table~\ref{tab:schemes}, Experiment~E.IV. "
         r"Every scheme pretrains the same encoder and fine-tunes the whole detector on the "
         r"budget of the column. Average precision at $40$ recall positions and an overlap of "
         r"$0.5$, meaned over the three vessel classes on Thames and on the single class on "
         r"SeePerSea. The highest value of each column is in bold. Figure~\ref{fig:pretrain-curves} "
         r"plots the same values.}\label{tab:pretrain}",
         r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Pretraining set & " + " & ".join(f"\\qty{{{b}}}{{\\percent}}" for b in BUDGETS) + r" \\",
         r"\midrule"]
    for di, (ds, dlabel) in enumerate((("london", "Thames"), ("seepersea", "SeePerSea"))):
        for vi, (view, vlabel) in enumerate((("bev", "bird's-eye view"), ("3d", "3D"))):
            L.append(r"\multicolumn{7}{@{}l}{\textit{" + f"{dlabel}, {vlabel}" + r"}" + UP + r"} \\")
            grid = {}
            for label, ck in SCHEMES:
                c = ck % ds if "%s" in ck else ck
                grid[label] = [cell(rows, ds, name_of(ds, "mstnet", c, b), view) for b in BUDGETS]
            best = [max([g[i] for g in grid.values() if g[i] is not None], default=None)
                    for i in range(len(BUDGETS))]
            for label, _ in SCHEMES:
                cs = [fmt(v, v is not None and v == best[i]) for i, v in enumerate(grid[label])]
                L.append(f"{label} & " + " & ".join(cs) + r" \\")
            if (di, vi) != (1, 1):
                L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)



# --- Experiment E.V, the pseudo-label stage --------------------------------------
def pseudo_table(rows):
    """The second pretraining stage against the arm it is added to. The MSTNet columns
    repeat Table~\ref{tab:budget} because the experiment is the difference between the
    two, and a reader should not have to hold six numbers across a page turn."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Experiment~E.V on SeePerSea. Average precision at $40$ recall "
         r"positions and an overlap of $0.5$, in the bird's-eye view (BEV) and in 3D. "
         r"The left group is the in-domain pretrained MSTNet of Table~\ref{tab:budget}, "
         r"the middle group adds the pseudo-label stage of Section~\ref{sec:method:pseudo} "
         r"between the masked pretraining and the fine-tuning, and the right group is "
         r"their difference. The higher of the two configurations is in bold.}"
         r"\label{tab:pseudo}",
         r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MSTNet} & \multicolumn{2}{c}{\makecell{+ pseudo-label\\stage}}"
         r" & \multicolumn{2}{c}{Difference} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
         r"Budget & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 3) + r" \\",
         r"\midrule"]
    for b in BUDGETS:
        base = [cell(rows, "seepersea", name_of("seepersea", "mstnet", "mstnet_ssl_seepersea", b), v)
                for v in ("bev", "3d")]
        ps = [cell(rows, "seepersea",
                   name_of("seepersea", "mstnet", "mstnet_ssl_seepersea_pseudo2", b), v)
              for v in ("bev", "3d")]
        cs = []
        for i in range(2):
            hi = None if base[i] is None or ps[i] is None else max(base[i], ps[i])
            cs.append(fmt(base[i], base[i] is not None and base[i] == hi))
            cs.append(fmt(ps[i], ps[i] is not None and ps[i] == hi))
        cs = [cs[0], cs[2], cs[1], cs[3]]
        for i in range(2):
            d = None if base[i] is None or ps[i] is None else ps[i] - base[i]
            cs.append("--" if d is None else f"${d:+.2f}$")
        L.append(f"\\qty{{{b}}}{{\\percent}} & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# --- Experiment E.VI, the two measures against the vertical error -----------------
# (row label, config stem, checkpoint tag). Every arm is pretrained in-domain; the two
# height-aware arms load the encoder that was pretrained with the extended token.
HEIGHT = [("MSTNet", "mstnet", "mstnet_ssl_seepersea"),
          ("+ height-aware encoding", "mstnet_zaware", "mstnet_ssl_seepersea_zaware"),
          ("+ refinement stage", "mstnet_stage2", "mstnet_ssl_seepersea"),
          ("+ both", "mstnet_zaware_stage2", "mstnet_ssl_seepersea_zaware"),
          ("+ both, vertical only", "mstnet_zaware_stage2_zonly", "mstnet_ssl_seepersea_zaware")]


def height_table(rows):
    """Budgets down the page and configurations across it, as in Tables~\ref{tab:budget}
    and~\ref{tab:encoder}. The panels are the two overlap thresholds. Stacking the three
    metric panels instead made the float twice as tall as it is wide, which the author
    flagged on 2026-08-26."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Experiment~E.VI on SeePerSea, the two measures of "
         r"Section~\ref{sec:method:gap} over the label budget. Average precision at $40$ "
         r"recall positions, in the bird's-eye view (BEV) and in 3D, at both overlap "
         r"thresholds. The center height and the box height enter the stricter one most "
         r"strongly. Every column pair is pretrained in-domain, the two height-aware ones "
         r"with the extended token of Eq.~\eqref{eq:ztoken}. \emph{MSTNet} is the "
         r"pipeline of Table~\ref{tab:budget}, \emph{+ height-aware} exchanges its mean "
         r"token for that extended one, \emph{+ refinement} adds the second stage of "
         r"Section~\ref{sec:method:gap:stage2} to the pipeline, \emph{+ both} adds both, "
         r"and \emph{+ both, vertical only} adds both and lets the second stage correct "
         r"the center height and the box height alone, leaving the footprint of the "
         r"proposal untouched. The highest value of each row and metric is in "
         r"bold.}\label{tab:height}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{MSTNet} & \multicolumn{2}{c}{\makecell{+ height-\\aware}}"
         r" & \multicolumn{2}{c}{\makecell{+ refine-\\ment}} & \multicolumn{2}{c}{+ both}"
         r" & \multicolumn{2}{c}{\makecell{+ both,\\vertical only}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
         r"\cmidrule(l){10-11}",
         r"Budget & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 5) + r" \\",
         r"\midrule"]
    for pi, iou in enumerate(("0.5", "0.7")):
        L.append(r"\multicolumn{11}{@{}l}{\textit{Overlap $%s$}} \\" % iou)
        for b in BUDGETS:
            vals = []
            for _, stem, ck in HEIGHT:
                n = name_of("seepersea", stem, ck, b)
                vals += [cell(rows, "seepersea", n, v, iou) for v in ("bev", "3d")]
            best = []
            for k in (0, 1):
                col = [v for i, v in enumerate(vals) if i % 2 == k and v is not None]
                best.append(max(col) if col else None)
            cs = [fmt(v, v is not None and v == best[i % 2]) for i, v in enumerate(vals)]
            L.append(f"\\qty{{{b}}}{{\\percent}} & " + " & ".join(cs) + r" \\")
        if pi == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# --- Where a detection sits, from measure_localization_error.py -------------------
LOC_SRC = ROOT / "results" / "localization_error.csv"


def localization_table():
    """The per-parameter error of the matched detections, or None without the CSV.

    Built from measure_localization_error.py, which has to have run first.
    """
    if not LOC_SRC.exists():
        return None
    src = list(csv.DictReader(LOC_SRC.open()))
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Whether a measure moves the box or only reorders the detections, on "
         r"the SeePerSea validation split at \qty{100}{\percent} labels. The matching and "
         r"the four quantities are those of Section~\ref{sec:method:setup:eval}. Depending "
         r"on the configuration between $423$ and $549$ of the $647$ annotated vessels are "
         r"matched, and the three error columns are medians over those pairs. \emph{Correct "
         r"in 3D} is the share of pairs that also reach an overlap of $0.5$ in three "
         r"dimensions, and the last column is the coefficient of "
         r"Eq.~\eqref{eq:spearman}, which states how well the score a detection carries "
         r"predicts the overlap its pair reaches.}\label{tab:localization}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Median error in \unit{\metre}} & & \\",
         r"\cmidrule(lr){2-4}",
         r"Configuration & horizontal" + DOWN + r" & \makecell{center\\height" + DOWN + r"} & "
         r"\makecell{box\\height" + DOWN + r"} & \makecell{Correct\\in 3D" + UP + r"} & "
         r"\makecell{Spearman\\coefficient" + UP + r"} \\", r"\midrule"]
    for r in src:
        L.append(" & ".join([
            r["configuration"],
            f"${float(r['dxy']):.2f}$", f"${float(r['dz']):.2f}$", f"${float(r['dh']):.2f}$",
            f"\\qty{{{float(r['kept']):.1f}}}{{\\percent}}",
            f"${float(r['rho']):.2f}$"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# --- Repeated seeds -----------------------------------------------------------------
# Twelve cells of the study were run three times. They were not chosen at random. Six are
# cells whose value falls as its budget grows, which is the only run-to-run spread a
# single-run grid can expose; three carry a claim whose margin is small; and three are the
# outliers of E.IV that Section 4.4 reads as failed optimizations rather than as evidence
# about their pretraining source. Every other cell of the study is a single run, so this
# table is what the spread bounds elsewhere in the discussion are measured on.
SEED_ROWS = [
    ("seepersea", "mstnet", None, "", "MSTNet, random"),
    ("seepersea", "mstnet", None, "_80pct", "MSTNet, random"),
    ("seepersea", "mstnet", "mstnet_ssl_seepersea", "", "MSTNet, pretrained"),
    ("seepersea", "mstnet_zaware", "mstnet_ssl_seepersea_zaware", "",
     "+ height-aware encoding"),
    ("seepersea", "mstnet_zaware_stage2", "mstnet_ssl_seepersea_zaware", "", "+ both"),
    ("seepersea", "mstnet", "mstnet_ssl_kitti", "_20pct", "MSTNet, automotive"),
    ("seepersea", "mstnet", "mstnet_ssl_pohang_seepersea", "_10pct", "MSTNet, combined"),
    ("seepersea", "voxelnext", None, "", "VoxelNeXt"),
    ("london", "mstnet", None, "_50pct", "MSTNet, random"),
    ("london", "mstnet", None, "_80pct", "MSTNet, random"),
    ("london", "centerpoint", None, "_50pct", "CenterPoint"),
    ("london", "centerpoint", None, "_80pct", "CenterPoint"),
]


def seed_table(rows):
    """The repeated runs, one row per cell that was run more than once.

    The supervised runs are unseeded, so the spread of a cell is what two further runs
    of the same configuration produce, and it bounds how small a difference elsewhere
    in the paper may be read as real.
    """
    def runs(ds, stem, ck, b, view):
        """The values of one cell across its original run and its two repeats."""
        base = name_of(ds, stem, ck, "100" if b == "" else b.strip("_pct"))
        out = []
        for suffix in ("", "_s2", "_s3"):
            v = cell(rows, ds, base + suffix, view)
            if v is not None:
                out.append(v)
        return out

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The twelve cells of the study that were run three times, at an "
         r"overlap of $0.5$. The first column of each group is the run the other tables "
         r"report. The supervised stage is unseeded, so the two beside it differ from it "
         r"in the initialization of every module not loaded from a checkpoint, in the "
         r"order the frames are drawn and in the augmentation sampled for each of them. "
         r"The mean is over the three, and every cell of the study not listed here is a "
         r"single run.}\label{tab:seeds}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{BEV" + UP + r"} & \multicolumn{4}{c}{3D" + UP + r"} \\",
         r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
         r"Configuration & Labels & run 1 & run 2 & run 3 & mean"
         r" & run 1 & run 2 & run 3 & mean \\", r"\midrule"]

    for di, (ds, dlabel) in enumerate((("seepersea", "SeePerSea"), ("london", "Thames"))):
        L.append(r"\multicolumn{10}{@{}l}{\textit{" + dlabel + r"}} \\")
        for d, stem, ck, b, label in SEED_ROWS:
            if d != ds:
                continue
            cs = []
            for view in ("bev", "3d"):
                v = runs(ds, stem, ck, b, view)
                cs += [fmt(x) for x in v] + ["--"] * (3 - len(v))
                cs.append(fmt(sum(v) / len(v)) if v else "--")
            budget = r"\qty{100}{\percent}" if b == "" else \
                r"\qty{%s}{\percent}" % b.strip("_pct")
            L.append(f"{label} & {budget} & " + " & ".join(cs) + r" \\")
        if di == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# --- Experiment E.VII, what a configuration costs to run ---------------------------
# Written by run_speed_sweep.py once the campaign has left both cards idle. Accuracy
# alone does not order detectors for a vessel that runs one onboard, and the two modules
# MSTNet exchanges are not free, so the benchmark carries a second table.
SPEED_SRC = ROOT / "results" / "speed.jsonl"

# (display name, config stem). The published detectors in the order of BENCH, then our
# own. The Thames TED-S cell is measured on the repaired per-class config, as in
# Table~\ref{tab:benchmark}, and BENCH_STEM redirects it.
SPEED_ROWS = [
    ("PointPillars", "pointpillar"), ("SECOND", "second"), ("CenterPoint", "centerpoint"),
    ("Voxel R-CNN", "voxel_rcnn"), ("PV-RCNN", "pv_rcnn"), ("VoxelNeXt", "voxelnext"),
    ("TransFusion", "transfusion"), ("DSVT, pillar", "dsvt"), ("DSVT, voxel", "dsvt_voxel"),
    ("TED-S", "ted_s"), ("LION-Mamba", "lion_mamba"),
]
# The arms of E.VI, SeePerSea only, so their Thames columns stay empty. The pseudo-label
# stage is a pretraining stage and adds no module at inference time, so the triple
# combination of Table~\ref{tab:triple} is this table's last row and is not repeated.
SPEED_OURS = [
    ("MSTNet", "mstnet"), ("MSTNet, learned encoder", "mstnet_c"),
    ("+ height-aware encoding", "mstnet_zaware"),
    ("+ refinement stage", "mstnet_stage2"),
    ("+ both", "mstnet_zaware_stage2"),
    ("+ both, vertical only", "mstnet_zaware_stage2_zonly"),
]


def speed_table():
    """Parameters, latency and peak memory per architecture, or None without the sweep.

    Built from run_speed_sweep.py, which has to have run first.
    """
    if not SPEED_SRC.exists():
        return None
    by = {}
    for line in SPEED_SRC.read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            by[(r["ds"], r["model"])] = r

    def cells(ds, stem):
        """The three cost figures of one architecture on one dataset."""
        r = by.get((ds, BENCH_STEM.get((ds, stem), stem)))
        if r is None:
            return ["--"] * 3
        return [f"${r['median_ms']:.1f}$", f"${r['p90_ms']:.1f}$",
                f"${r['peak_mem_MB']:.0f}$"]

    def params(stem):
        """The parameter count of one architecture, from whichever dataset measured it."""
        for ds in ("london", "seepersea"):
            r = by.get((ds, BENCH_STEM.get((ds, stem), stem)))
            if r:
                return f"${r['params_M']:.2f}$"
        return "--"

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Experiment~E.VII. What one frame costs each configuration at "
         r"inference, measured under the protocol of "
         r"Section~\ref{sec:method:setup:efficiency}. Latency is the median and the "
         r"ninetieth percentile over $200$ frames of the evaluation split at a batch of "
         r"one, and memory is the peak the card allocates over the same frames. The two "
         r"datasets are reported apart because a Thames frame occupies a median of $970$ "
         r"cells against $75$ on SeePerSea. The parameter count is a property of the "
         r"architecture and is given once; where the two datasets configure a detector "
         r"differently it is the Thames value. The arms of Section~\ref{sec:method:gap} "
         r"run on SeePerSea only. The pseudo-label stage of "
         r"Section~\ref{sec:method:pseudo} adds no module at inference, so the "
         r"combination of Table~\ref{tab:triple} costs what the last row "
         r"costs.}\label{tab:speed}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Thames} & \multicolumn{3}{c}{SeePerSea} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Configuration & \makecell{Param.\\/\num{e6}" + DOWN + r"}"
         r" & \makecell{median\\\unit{\milli\second}" + DOWN + r"}"
         r" & \makecell{p90\\\unit{\milli\second}" + DOWN + r"}"
         r" & \makecell{peak\\\unit{\mega\byte}" + DOWN + r"}"
         r" & \makecell{median\\\unit{\milli\second}" + DOWN + r"}"
         r" & \makecell{p90\\\unit{\milli\second}" + DOWN + r"}"
         r" & \makecell{peak\\\unit{\mega\byte}" + DOWN + r"} \\", r"\midrule"]
    for label, stem in SPEED_ROWS:
        L.append(f"{label} & " + " & ".join(
            [params(stem)] + cells("london", stem) + cells("seepersea", stem)) + r" \\")
    L.append(r"\midrule")
    for label, stem in SPEED_OURS:
        L.append(f"{label} & " + " & ".join(
            [params(stem)] + cells("london", stem) + cells("seepersea", stem)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# --- Experiment E.VII, the benchmark ----------------------------------------------
# (display name, config stem, framework tag). The order groups the published detectors
# by the representation they build and closes with our own two rows.
BENCH = [
    ("PointPillars", "pointpillar"), ("SECOND", "second"), ("CenterPoint", "centerpoint"),
    ("Voxel R-CNN", "voxel_rcnn"), ("PV-RCNN", "pv_rcnn"), ("VoxelNeXt", "voxelnext"),
    ("TransFusion", "transfusion"), ("DSVT, pillar", "dsvt"), ("DSVT, voxel", "dsvt_voxel"),
    ("TED-S", "ted_s"), ("LION-Mamba", "lion_mamba"),
]

# Where a dataset needs a different configuration of the same detector. The Thames TED-S
# cell comes from the repaired per-class config, which landed on 2026-08-28. TED's RoI
# target assigner reads the threshold lists in TARGET_CONFIG as one entry per class under
# CLS_SCORE_TYPE roi_iou_x, and ted_s.yaml carried the length-one lists of the upstream
# single-class KITTI recipe, so every proposal matched to a medium or large vessel was
# trained as background whatever its overlap and the run produced no large-vessel
# detection at all (mAP 30.49, 0.437 detections per frame). ted_s_perclass.yaml is that
# config with the five lists extended to three entries, and it reaches 90.42 and 85.78 at
# an overlap of 0.5 with 4.435 detections per frame. SeePerSea is single class, where a
# length-one list is correct, so it keeps the original config. This replaces the blanking
# rule that stood here while the repair was training.
BENCH_STEM = {("london", "ted_s"): "ted_s_perclass"}

# Our own configurations at full supervision, as the manuscript names them. The first is
# the pipeline of Section 2.2 with the in-domain pretraining of Section 2.3, which is the
# row a reader should compare against. The rest are the variants of Sections 2.3.2 and 2.4.
MSTNet_RUNS = [
    ("mstnet", "mstnet_ssl_%s", "the mean encoder and in-domain pretraining"),
    ("mstnet_c", "mstnet_c_ssl_%s", "the learned encoder and in-domain pretraining"),
    ("mstnet", "mstnet_ssl_pohang_%s", "the mean encoder and combined pretraining"),
    ("mstnet", "mstnet_ssl_pohang",
     "the mean encoder and cross-domain maritime pretraining"),
    ("mstnet", "mstnet_ssl_kitti",
     "the mean encoder and out-of-domain automotive pretraining"),
    ("mstnet", "mstnet_ssl_%s_pseudo2",
     "the mean encoder, in-domain pretraining and the pseudo-label stage"),
    ("mstnet_zaware", "mstnet_ssl_%s_zaware",
     "the height-aware encoder and in-domain pretraining"),
    ("mstnet_stage2", "mstnet_ssl_%s",
     "the mean encoder, in-domain pretraining and the local refinement stage"),
    ("mstnet_zaware_stage2", "mstnet_ssl_%s_zaware",
     "the height-aware encoder, in-domain pretraining and the local refinement stage"),
    ("mstnet_zaware_stage2_zonly", "mstnet_ssl_%s_zaware",
     "the height-aware encoder, in-domain pretraining and the refinement stage "
     "restricted to the center height and the box height"),
    # The triple combination. Added to this list on 2026-08-27, before its full
    # supervision cell finished, so that the row updates itself the moment the run
    # lands rather than staying on a configuration the study has since overtaken.
    ("mstnet_zaware_stage2_zonly", "mstnet_ssl_%s_zaware_pseudo2",
     "the height-aware encoder, in-domain and pseudo-label pretraining and the "
     "refinement stage restricted to the center height and the box height"),
]


def best_mstnet(rows, ds):
    """The variant of Table~\ref{tab:budget} and Sections 2.3.2 and 2.4 that reaches the
    highest 3D value at full supervision, with the phrase that names it. Chosen from the
    CSV rather than written down, so the row cannot drift from the runs."""
    best = None
    for stem, ck, label in MSTNet_RUNS:
        c = ck % ds if "%s" in ck else ck
        n = name_of(ds, stem, c, "100")
        v3 = cell(rows, ds, n, "3d")
        if v3 is None:
            continue
        if best is None or v3 > best[0]:
            best = (v3, cell(rows, ds, n, "bev"), label, stem, c)
    return best


def benchmark_table(rows):
    """The eleven published detectors and our own, under one protocol, on both datasets."""
    L = [r"\begin{table}[htbp]", r"\centering"]
    picks = {ds: best_mstnet(rows, ds) for ds in ("london", "seepersea")}
    named = ", ".join(
        f"on {'Thames' if ds == 'london' else 'SeePerSea'} it is the configuration with "
        f"{picks[ds][2]}" for ds in ("london", "seepersea") if picks[ds])
    L += [r"\caption{Experiment~E.VII. Average precision at $40$ recall positions in the "
          r"bird's-eye view (BEV) and in 3D, at both overlap thresholds and at "
          r"\qty{100}{\percent} labels. A Thames value is the mean over the three vessel "
          r"classes, a SeePerSea value the single class. \emph{MSTNet} is the pipeline of "
          r"Section~\ref{sec:method:mstnet} with the in-domain pretraining of "
          r"Section~\ref{sec:method:ssl}. \emph{MSTNet, strongest variant} is whichever "
          r"of our configurations reaches the highest 3D value at an overlap of $0.5$ on "
          r"that dataset, which is read off the measured values and is therefore not a "
          r"selection a practitioner could make in advance; " + named + r". That one "
          r"configuration supplies all four of its columns, so the row describes a single "
          r"system and not the best value we reached per metric. The highest value of each "
          r"column is in bold.}\label{tab:benchmark}",
          r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}", r"\toprule",
          r" & \multicolumn{4}{c}{Thames} & \multicolumn{4}{c}{SeePerSea} \\",
          r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
          r" & \multicolumn{2}{c}{overlap $0.5$} & \multicolumn{2}{c}{overlap $0.7$}"
          r" & \multicolumn{2}{c}{overlap $0.5$} & \multicolumn{2}{c}{overlap $0.7$} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
          r"Detector & " + " & ".join([f"{v}{UP}" for v in ("BEV", "3D")] * 4) + r" \\",
          r"\midrule"]

    def four(ds, stem, ck, suppress=False):
        """The four metric combinations of one cell, or four blanks when suppressed."""
        if suppress:
            return [None] * 4
        n = name_of(ds, stem, ck, "100")
        return [cell(rows, ds, n, v, i) for i in ("0.5", "0.7") for v in ("bev", "3d")]

    grid = OrderedDict()
    for label, stem in BENCH:
        grid[label] = (four("london", BENCH_STEM.get(("london", stem), stem), None)
                       + four("seepersea", BENCH_STEM.get(("seepersea", stem), stem), None))
    grid["MSTNet"] = (four("london", "mstnet", "mstnet_ssl_london")
                       + four("seepersea", "mstnet", "mstnet_ssl_seepersea"))
    # The strongest variant differs per dataset, so its four columns are read from the
    # run best_mstnet picked rather than from one config stem.
    strongest = []
    for ds in ("london", "seepersea"):
        stem, ck = (picks[ds][3], picks[ds][4]) if picks[ds] else (None, None)
        strongest += four(ds, stem, ck) if stem else [None] * 4
    grid["MSTNet, strongest variant"] = strongest

    best = [max([g[i] for g in grid.values() if g[i] is not None], default=None)
            for i in range(8)]
    for label, vals in grid.items():
        if label == "MSTNet":
            L.append(r"\midrule")
        cs = [fmt(v, v is not None and v == best[i]) for i, v in enumerate(vals)]
        L.append(f"{label} & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)



# --- The pseudo-label stage on the strongest configuration, an ablation of E.VI ----
# The same question Table~\ref{tab:pseudo} asks of MSTNet, asked of the arm that reaches
# the highest 3D value. It sits with E.VI and not with E.V because the configuration it
# is applied to is only introduced in Section 2.4.
# It ran at the three smallest budgets first, on the reading that the stage is neutral on
# MSTNet from 20 percent labels upward. That crossing does not transfer: on this arm the
# stage still adds 6.04 points in 3D at 20 percent, more than it adds to MSTNet at 10, so
# the three larger budgets were added on 2026-08-27 and the full supervision cell landed
# on 2026-08-28. The column is therefore complete and the budget list is the full six.
TRIPLE_BUDGETS = ["5", "10", "20", "50", "80", "100"]


def triple_table(rows):
    """The pseudo-label stage added to the strongest arm of E.VI."""
    stem, ck = "mstnet_zaware_stage2_zonly", "mstnet_ssl_seepersea_zaware"
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The pseudo-label stage of Section~\ref{sec:method:pseudo} added in "
         r"front of the \emph{+ both, vertical only} column of Table~\ref{tab:height} on "
         r"SeePerSea, which is the height-aware encoder together with the refinement stage "
         r"restricted to the center height and the box height. "
         r"Average precision at $40$ recall positions in the bird's-eye view (BEV) and in "
         r"3D. The left group repeats that column from Table~\ref{tab:height}, the "
         r"middle group runs the pseudo-label stage in front of it, and the right group is "
         r"their difference. The higher of the two "
         r"configurations is in bold.}\label{tab:triple}",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{\makecell{+ both,\\vertical only}}"
         r" & \multicolumn{3}{c}{+ pseudo-label stage}"
         r" & \multicolumn{3}{c}{Difference} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"Budget & " + " & ".join([f"BEV{UP}", f"3D{UP}", r"\makecell{3D\\$0.7$" + UP + r"}"] * 3)
         + r" \\", r"\midrule"]
    METRICS = [("bev", "0.5"), ("3d", "0.5"), ("3d", "0.7")]
    for b in TRIPLE_BUDGETS:
        n = name_of("seepersea", stem, ck, b)
        base = [cell(rows, "seepersea", n, v, i) for v, i in METRICS]
        pse = [cell(rows, "seepersea", name_of("seepersea", stem, ck + "_pseudo2", b), v, i)
               for v, i in METRICS]
        cs = []
        for k in range(3):
            hi = None if base[k] is None or pse[k] is None else max(base[k], pse[k])
            cs.append(fmt(base[k], base[k] is not None and base[k] == hi))
        for k in range(3):
            hi = None if base[k] is None or pse[k] is None else max(base[k], pse[k])
            cs.append(fmt(pse[k], pse[k] is not None and pse[k] == hi))
        for k in range(3):
            d = None if base[k] is None or pse[k] is None else pse[k] - base[k]
            cs.append("--" if d is None else f"${d:+.2f}$")
        L.append(f"\\qty{{{b}}}{{\\percent}} & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


# The training loss of the eight full-supervision cells, from results/train_loss.csv
# written by export_train_loss.py. Sections 4.1 and 4.2 argue from these curves, so they
# belong in the results rather than appearing first in the discussion.
LOSS_SRC = ROOT / "results" / "train_loss.csv"
LOSS_STYLES = [("solid", "mean", "random", "Mean encoder, random"),
               ("dashed", "mean", "pretrained", "Mean encoder, pretrained"),
               ("dotted", "learned", "random", "Learned encoder, random"),
               ("dashdotted", "learned", "pretrained", "Learned encoder, pretrained")]


def train_loss_curves():
    """The training loss of the full-supervision cells, or None without the CSV.

    Built from export_train_loss.py, which has to have run first.
    """
    if not LOSS_SRC.exists():
        return None
    src = list(csv.DictReader(LOSS_SRC.open()))
    L = [HDR, r"\begin{tikzpicture}"]
    for k, (ds, dlabel) in enumerate((("london", "Thames"), ("seepersea", "SeePerSea"))):
        pos = r"  at={(ax0.outer east)}, anchor=outer west," if k else r"  at={(0,0)},"
        L += [r"\begin{axis}[", r"  name=ax%d," % k, pos,
              r"  width=0.46\textwidth, height=0.38\textwidth,",
              r"  title={%s}, title style={font=\small, yshift=-1mm}," % dlabel,
              r"  xlabel={Epoch}, ylabel={Training loss},",
              r"  xmin=0, xmax=82, ymin=0, ymax=%d," % (12 if ds == "london" else 20),
              r"  xtick={1,20,40,60,80}, xticklabel style={font=\small},",
              r"  yticklabel style={font=\small}, label style={font=\small},",
              r"  tick align=outside, grid=none,",
              r"  every axis plot/.append style={line width=0.7pt, black, mark=none},",
              r"]"]
        for style, enc, init, _ in LOSS_STYLES:
            pts = [(int(r["epoch"]), float(r["loss"])) for r in src
                   if r["dataset"] == ds and r["encoder"] == enc and r["init"] == init]
            if not pts:
                continue
            coords = " ".join(f"({e},{v:.3f})" for e, v in sorted(pts))
            L.append(r"\addplot[%s] coordinates {%s};" % (style, coords))
        L.append(r"\end{axis}")
    L += [r"\begin{axis}[hide axis, scale only axis, width=1mm, height=1mm,",
          r"  xmin=0, xmax=1, ymin=0, ymax=1,",
          r"  at={(ax0.outer south east)}, anchor=north, yshift=-5mm,",
          r"  legend style={font=\small, draw=black, at={(0.5,1)}, anchor=north, "
          r"legend columns=2, column sep=6pt, inner sep=3pt},",
          r"  legend cell align=left,]"]
    for style, _, _, label in LOSS_STYLES:
        L.append(r"\addlegendimage{%s, line width=0.7pt, black, mark=none}" % style)
        L.append(r"\addlegendentry{%s}" % label)
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


# E.I and E.III as curves, the plotted form of tab_budget and the figure the graphical
# abstract sketches. Four panels, the two metrics over the two datasets, three
# configurations each. Dash pattern and marker follow the graphical abstract, solid for
# the pretrained encoder, dashed for the random one, dotted for the baseline.
HDR = ("%% Generated by make_paper_tables.py. Do not edit by hand.\n"
       "%% Requires in the preamble:\n"
       "%%   \\usepackage{pgfplots}\n"
       "%%   \\pgfplotsset{compat=1.18}   %% pgfplots 1.18 or newer\n"
       "%%   \\usepackage{siunitx}        %% for \\unit{\\percent} in the axis label")

BUDGET_STYLES = [("solid", "*", "MSTNet, pretrained", "mstnet", "mstnet_ssl_%s"),
                 ("dashed", "square*", "MSTNet, random", "mstnet", None),
                 ("dotted", "triangle*", "CenterPoint", "centerpoint", None)]


def budget_curves(rows):
    """E.I and E.III as four panels of curves over the label budget."""
    L = [HDR, r"\begin{tikzpicture}"]
    panels = [(v, d) for v in ("bev", "3d") for d in ("london", "seepersea")]
    for k, (view, ds) in enumerate(panels):
        vlabel = "Bird's-eye view" if view == "bev" else "3D"
        dlabel = "Thames" if ds == "london" else "SeePerSea"
        # Thames carries three vessel classes and the value is their mean, SeePerSea
        # carries one, where the mean and the class value coincide.
        metric = "mAP" if ds == "london" else "AP"
        if k == 0:
            pos = r"  at={(0,0)},"
        elif k % 2:
            pos = r"  at={(ax%d.outer east)}, anchor=outer west," % (k - 1)
        else:
            pos = r"  at={(ax%d.outer south west)}, anchor=outer north west," % (k - 2)
        L += [r"\begin{axis}[", r"  name=ax%d," % k, pos,
              r"  width=0.46\textwidth, height=0.38\textwidth,",
              r"  xmode=log, log basis x=10,",
              r"  title={%s, %s}, title style={font=\small, yshift=-1mm}," % (dlabel, vlabel),
              r"  xlabel={Labels in \unit{\percent}}, ylabel={%s at $0.5$}," % metric,
              r"  xmin=4, xmax=125, ymin=0, ymax=100,",
              r"  xtick={5,10,20,50,100}, xticklabels={5,10,20,50,100},",
              r"  xticklabel style={font=\small}, yticklabel style={font=\small},",
              r"  label style={font=\small}, tick align=outside, grid=none,",
              r"  every axis plot/.append style={line width=0.7pt, mark size=1.5pt, black},",
              r"]"]
        for style, mark, label, stem, ck in BUDGET_STYLES:
            c = ck % ds if ck else None
            pts = [(int(b), cell(rows, ds, name_of(ds, stem, c, b), view)) for b in BUDGETS]
            coords = " ".join(f"({x},{y:.2f})" for x, y in pts if y is not None)
            L.append(r"\addplot[%s, mark=%s] coordinates {%s};" % (style, mark, coords))
        L.append(r"\end{axis}")
    L += [r"\begin{axis}[hide axis, scale only axis, width=1mm, height=1mm,",
          r"  xmin=0, xmax=1, ymin=0, ymax=1,",
          r"  at={(ax2.outer south east)}, anchor=north, yshift=-5mm,",
          r"  legend style={font=\small, draw=black, at={(0.5,1)}, anchor=north, "
          r"legend columns=3, column sep=6pt, inner sep=3pt},",
          r"  legend cell align=left,]"]
    for style, mark, label, _, _ in BUDGET_STYLES:
        L.append(r"\addlegendimage{%s, mark=%s, line width=0.7pt, mark size=1.5pt, black}"
                 % (style, mark))
        L.append(r"\addlegendentry{%s}" % label)
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


# The label budget curves of E.IV, the plotted form of tab_pretrain. Drawn with pgfplots
# so the text is set in the document font, and distinguished by dash pattern and marker
# rather than by colour alone, which the figure guidelines require for grayscale printing
# and for colorblind readers.
STYLES = [("solid", "*", "In-domain"),
          ("dashed", "square*", "Cross-domain maritime"),
          ("dotted", "triangle*", "Out-of-domain automotive"),
          ("dashdotted", "diamond*", "Combined")]


def pretrain_curves(rows):
    """Four panels, the two metrics over the two datasets, one curve per pretraining
    source. Same layout and same budget axis as the E.I figure so the two read alike."""
    L = [HDR, r"\begin{tikzpicture}"]
    panels = [(v, d) for v in ("bev", "3d") for d in ("london", "seepersea")]
    for k, (view, ds) in enumerate(panels):
        vlabel = "Bird's-eye view" if view == "bev" else "3D"
        dlabel = "Thames" if ds == "london" else "SeePerSea"
        metric = "mAP" if ds == "london" else "AP"
        if k == 0:
            pos = r"  at={(0,0)},"
        elif k % 2:
            pos = r"  at={(ax%d.outer east)}, anchor=outer west," % (k - 1)
        else:
            pos = r"  at={(ax%d.outer south west)}, anchor=outer north west," % (k - 2)
        L += [r"\begin{axis}[", r"  name=ax%d," % k, pos,
              r"  width=0.46\textwidth, height=0.38\textwidth,",
              r"  xmode=log, log basis x=10,",
              r"  title={%s, %s}, title style={font=\small, yshift=-1mm}," % (dlabel, vlabel),
              r"  xlabel={Labels in \unit{\percent}}, ylabel={%s at $0.5$}," % metric,
              r"  xmin=4, xmax=125, ymin=0, ymax=%d," % (100 if ds == "london" else 85),
              r"  xtick={5,10,20,50,100}, xticklabels={5,10,20,50,100},",
              r"  xticklabel style={font=\small}, yticklabel style={font=\small},",
              r"  label style={font=\small}, tick align=outside, grid=none,",
              r"  every axis plot/.append style={line width=0.7pt, mark size=1.5pt, black},",
              r"]"]
        for style, mark, label in STYLES:
            ck = dict(STYLE_CK)[label]
            ck = ck % ds if "%s" in ck else ck
            pts = [(int(b), cell(rows, ds, name_of(ds, "mstnet", ck, b), view)) for b in BUDGETS]
            coords = " ".join(f"({x},{y:.2f})" for x, y in pts if y is not None)
            L.append(r"\addplot[%s, mark=%s] coordinates {%s};" % (style, mark, coords))
        L.append(r"\end{axis}")
    L += [r"\begin{axis}[hide axis, scale only axis, width=1mm, height=1mm,",
          r"  xmin=0, xmax=1, ymin=0, ymax=1,",
          r"  at={(ax2.outer south east)}, anchor=north, yshift=-5mm,",
          r"  legend style={font=\small, draw=black, at={(0.5,1)}, anchor=north, "
          r"legend columns=2, column sep=6pt, inner sep=3pt},",
          r"  legend cell align=left,]"]
    for style, mark, label in STYLES:
        L.append(r"\addlegendimage{%s, mark=%s, line width=0.7pt, mark size=1.5pt, black}"
                 % (style, mark))
        L.append(r"\addlegendentry{%s}" % label)
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(L)


STYLE_CK = [("In-domain", "mstnet_ssl_%s"), ("Cross-domain maritime", "mstnet_ssl_pohang"),
            ("Out-of-domain automotive", "mstnet_ssl_kitti"), ("Combined", "mstnet_ssl_pohang_%s")]


# Runs that were executed but are not part of the study the paper describes. Pretraining
# one annotated dataset on the frames of the other is not one of the four schemes of the
# pretraining section, and the author excluded it on 2026-08-17. The rows stay in
# all_results.csv as a record of what ran and are dropped here so no table can pick them up.
EXCLUDED = [
    re.compile(r"^CAM_seepersea_mstnet(_\d+pct)?_mstnet_ssl_london$"),
    re.compile(r"^CAM_london_mstnet(_\d+pct)?_mstnet_ssl_seepersea$"),
    # The pseudo-label arm is two stages, the cluster boxes first and the real labels of
    # the budget after. Until 2026-08-21 the campaign ran only the first and evaluated it
    # against the real validation labels, which measures a detector that has never seen an
    # annotated vessel and scores about 5 AP at every budget. Those rows stay in the CSV as
    # a record of what ran; the reportable cells carry the pseudo2 tag.
    re.compile(r"^CAM_seepersea_mstnet_pseudo(_\d+pct)?_mstnet_ssl_seepersea$"),
    # The front-view token branch, a second stream over the (x, z) plane fused into the
    # pseudo image. It was an early attempt at the vertical deficit that Section 2.4 now
    # attacks with the height-aware token and the refinement stage, it exists at full
    # supervision only, and the author retired it on 2026-08-25. Excluded here so no
    # table can pick it up; the row stays in all_results.csv as a record of what ran.
    re.compile(r"^CAM_seepersea_mstnet_mv(_\d+pct)?_mstnet_ssl_seepersea$"),
]


def main():
    """Write every table and figure of the paper from the collected results."""
    rows = list(csv.DictReader(SRC.open()))
    dropped = [r for r in rows if any(p.match(r["experiment"]) for p in EXCLUDED)]
    rows = [r for r in rows if r not in dropped]
    if dropped:
        print(f"excluded {len(dropped)} cross-dataset pretraining rows: "
              + ", ".join(sorted(r["experiment"] for r in dropped)))
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "tab_hyperparams.tex").write_text(
        hyperparam_table(rows) + "\n\n" + suppression_table(rows) + "\n")
    (OUTD / "tab_all_results.tex").write_text(
        results_table(rows) + "\n\n" + r11_table(rows) + "\n")
    for fn, body in (("tab_budget.tex", budget_table(rows)),
                     ("tab_encoder.tex", encoder_table(rows)),
                     ("tab_perclass.tex", perclass_table(rows)),
                     ("tab_perclass_budget.tex", perclass_budget_table(rows)),
                     ("tab_pretrain.tex", pretrain_table(rows)),
                     ("tab_pseudo.tex", pseudo_table(rows)),
                     ("tab_height.tex", height_table(rows)),
                     ("tab_localization.tex", localization_table()),
                     ("tab_benchmark.tex", benchmark_table(rows)),
                     ("tab_triple.tex", triple_table(rows)),
                     ("tab_speed.tex", speed_table()),
                     ("tab_seeds.tex", seed_table(rows)),
                     ("fig_pretrain_curves.tikz", pretrain_curves(rows)),
                     ("fig_budget_curves.tikz", budget_curves(rows)),
                     ("fig_train_loss.tikz", train_loss_curves())):
        if body is None:
            print(f"  skipped {fn}, its measurement script has not been run")
            continue
        (OUTD / fn).write_text(body + "\n")
    print(f"{len(rows)} rows -> {OUTD}/tab_hyperparams.tex (protocol, exceptions, grid, "
          f"suppression), tab_all_results.tex (R40 and R11)")


if __name__ == "__main__":
    main()
