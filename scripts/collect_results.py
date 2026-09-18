"""Collect every evaluation we have into one CSV.

Until now the numbers lived in 15 separate jsonl files written by 15 separate runner
scripts, several of which recorded only the return code and not the metrics. The
evaluation logs are the single source that always contains them, so this script parses
those instead and treats the jsonl files as nothing more than a naming hint.

Output columns hold both KITTI protocols (11 and 40 recall positions) at both IoU
thresholds, for bird's-eye view and three dimensions, per class and averaged. Nothing
is filtered or rounded here; selection belongs in the table builder, not in the
collection step.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGD = ROOT / "runs" / "logs"
REEVALD = ROOT / "runs" / "logs_reeval"
OUT = ROOT / "results" / "all_results.csv"

# "Car AP_R40@0.70, 0.50, 0.50:" -> the three numbers are the IoU thresholds for
# 2D bbox, bird's-eye view and 3D. Only the last two matter for us.
HEAD = re.compile(r"^(\w+) AP(_R40)?@([\d.]+), ([\d.]+), ([\d.]+):\s*$")
ROW = re.compile(r"^(bev|3d|aos)\s+AP:([\d.]+)")
# Each block also carries a 'bbox AP:' line for the 2D projection, which we do not
# report. It must be recognised all the same, otherwise it looks like the start of a
# new section and the two lines we do want are discarded.
IGNORE_ROW = re.compile(r"^bbox\s+AP:")
# The evaluation now emits, after the aggregate block, one block per real class and a
# mean over them. Inside a per-class block the header still says "Car", because the
# KITTI evaluator only knows its own category names, so the real class has to be taken
# from the section marker rather than from the header.
SECTION = re.compile(r"^-+ per-class evaluation, (\w+) -+\s*$")
MAP_SECTION = re.compile(r"^-+ mean over classes \(mAP\) -+\s*$")
MAP_ROW = re.compile(r"^\s+(\w+)_(bev|3d|aos)/(\w+?)(_R40)?:\s*([\d.]+)\s*$")


CFG_LINE = re.compile(r"cfg\.([A-Za-z0-9_.]+): (.*)$")
# The evaluated checkpoint, echoed by the argument dump. It is what identifies an
# evaluation, so it decides which rows may be merged.
CKPT_LINE = re.compile(r"^ckpt\s+(\S+\.pth)\s*$")

# Everything the hyperparameter table needs is echoed into the log header by
# log_config_to_file, so the configuration is read back from the run that actually
# happened rather than from a yaml file that may have been edited since.
WANTED = {
    "CLASS_NAMES": "classes",
    "DATA_CONFIG.DATA_PATH": "data_path",
    "DATA_CONFIG.POINT_CLOUD_RANGE": "pc_range",
    "DATA_CONFIG.DATA_SPLIT.train": "train_split",
    "MODEL.NAME": "detector",
    "MODEL.VFE.NAME": "vfe",
    "MODEL.MAP_TO_BEV.NAME": "map_to_bev",
    "MODEL.BACKBONE_3D.NAME": "backbone_3d",
    "MODEL.BACKBONE_2D.NAME": "backbone_2d",
    "MODEL.DENSE_HEAD.NAME": "dense_head",
    "MODEL.ROI_HEAD.NAME": "roi_head",
    "OPTIMIZATION.BATCH_SIZE_PER_GPU": "batch_size",
    "OPTIMIZATION.NUM_EPOCHS": "epochs",
    "OPTIMIZATION.OPTIMIZER": "optimizer",
    "OPTIMIZATION.LR": "lr",
    "OPTIMIZATION.WEIGHT_DECAY": "weight_decay",
    "OPTIMIZATION.GRAD_NORM_CLIP": "grad_clip",
}


def parse_config(path):
    """Pull the echoed configuration out of a log header. VOXEL_SIZE is dug out of the
    DATA_PROCESSOR list, which is printed as one python repr rather than as separate
    keys."""
    out = {}
    for line in path.read_text(errors="ignore").splitlines()[:400]:
        line = line.split("INFO  ")[-1].strip()
        c = CKPT_LINE.match(line)
        if c:
            out["ckpt"] = c.group(1).strip()
        m = CFG_LINE.match(line)
        if not m:
            continue
        key, val = m.groups()
        if key in WANTED:
            out[WANTED[key]] = val.strip()
        if key.endswith("DATA_PROCESSOR"):
            v = re.search(r"'VOXEL_SIZE': (\[[^\]]*\])", val)
            if v:
                out["voxel_size"] = v.group(1)
    if "data_path" in out:
        out["dataset"] = out.pop("data_path").rsplit("custom_", 1)[-1]
    return out


def parse(path):
    """Return {(cls, protocol, iou, view): ap}. Later blocks overwrite earlier ones,
    which is what we want when a log contains a re-run of the same evaluation."""
    out = {}
    cur = None
    section = None          # None while in the aggregate block, else the real class
    in_map = False
    for line in path.read_text(errors="ignore").splitlines():
        line = line.split("INFO  ")[-1]

        m = SECTION.match(line.strip())
        if m:
            section, cur, in_map = m.group(1), None, False
            continue
        if MAP_SECTION.match(line.strip()):
            section, cur, in_map = None, None, True
            continue
        if in_map:
            m = MAP_ROW.match(line)
            if m:
                _cls, view, diff, r40, ap = m.groups()
                if view != "aos" and diff.startswith("easy"):
                    # The three difficulty columns are identical by construction, see
                    # the comment in the results table, so one of them is enough.
                    out[("mAP", "R40" if r40 else "R11", "0.7", view)] = float(ap)
            elif line.strip() and not line.startswith(" "):
                in_map = False
            continue

        m = HEAD.match(line)
        if m:
            cls, r40, _b2d, iou_bev, iou_3d = m.groups()
            cur = (section or cls, "R40" if r40 else "R11", iou_bev, iou_3d)
            continue
        if cur is None:
            continue
        m = ROW.match(line)
        if m:
            view, ap = m.group(1), float(m.group(2))
            if view == "aos":
                continue
            iou = cur[2] if view == "bev" else cur[3]
            out[(cur[0], cur[1], iou, view)] = ap
        elif not IGNORE_ROW.match(line) and line.strip():
            cur = None
    return out


def iter_logs():
    """Two sources. runs/logs holds the copies our runner scripts made, keyed by
    the tag we gave the experiment. The framework's own output trees hold one log per
    evaluation, including several that no runner script ever recorded, among them the
    TED-S runs made after the weighted box fusion fix. The output trees are commonly
    placed on a separate disk behind a symlink, so the walk has to follow links.
    """
    # An experiment can appear twice, once from the campaign and once from the
    # re-evaluation of 2026-08-17 that moved every detector onto the uniform
    # post-processing the paper describes. Which of the two is current depends on the
    # direction: a run older than that date was superseded by the re-evaluation, a run
    # retrained after it superseded the re-evaluation in turn. Modification time decides,
    # so nothing has to be hard-coded. Reading the campaign copy blindly is what left 217
    # stale cells in all_results.csv until 2026-08-21.
    newest = {}
    for d, pre in ((LOGD, "CAM_"), (REEVALD, "RE_")):
        for p in sorted(d.glob("*_eval.log")):
            name = "CAM_" + p.name[len(pre):-len("_eval.log")]
            if name not in newest or p.stat().st_mtime > newest[name].stat().st_mtime:
                newest[name] = p
    for name, p in sorted(newest.items()):
        yield name, p
    for fw in ("OpenPCDet", "TED", "LION"):
        base = ROOT / fw / "output"
        for p in sorted(base.glob("**/log_eval_*.txt")):
            # .../<cfg_group>/<cfg_name>/<extra_tag>/eval/epoch_N/<split>/default/log
            parts = p.relative_to(base).parts
            if len(parts) < 4:
                continue
            cfg_name, tag = parts[1], parts[2]
            epoch = next((x for x in parts if x.startswith("epoch_")), "")
            yield f"{cfg_name}/{tag}@{epoch}", p


def main():
    """Read every evaluation log and write one row per evaluation into the CSV.

    The row records what was evaluated, the configuration it came from and all four
    metric combinations, so no number in the paper has to be transcribed by hand.
    """
    rows = []
    classes = set()
    seen = set()
    for name, log in iter_logs():
        d = parse(log)
        if not d:
            continue
        if name in seen:
            continue
        seen.add(name)
        rec = {"experiment": name, "source": str(log.relative_to(ROOT.parent))}
        rec.update(parse_config(log))
        # The batch size echoed in an eval log is the one used for evaluation. The
        # training batch size is the one the table must report, so prefer the train log.
        tlog = LOGD / f"{name}_train.log"
        if "/" not in name and tlog.exists():
            t = parse_config(tlog)
            for k in ("batch_size", "epochs", "lr"):
                if k in t:
                    rec[k] = t[k]
        for (cls, proto, iou, view), ap in d.items():
            classes.add(cls)
            rec[f"{proto}_{view}_{iou.rstrip('0').rstrip('.')}_{cls}"] = ap
        rows.append(rec)

    # The same evaluation is reachable under two names, the tag a runner script gave it
    # and the framework's own output path. Collapse them on the metric tuple and keep
    # the runner tag, which is the name the rest of the project refers to.
    #
    # The signature includes the checkpoint the numbers came from, not the numbers
    # alone. An earlier version keyed on the metric tuple only, which merged genuinely
    # different experiments whenever they happened to score identically. That is
    # precisely what happens to failed runs, which all collapse to the same near-zero
    # tuple, so the control runs one most wants to see were the ones that disappeared.
    # One real case: a TED-S control and a different-learning-rate run were folded into
    # a single row and the control vanished from the table entirely.
    # Aliases are kept as a list so a third collision can no longer overwrite a second.
    # The checkpoint path is echoed into every evaluation log and is what actually
    # identifies an evaluation. Two log files naming the same checkpoint are the same
    # measurement under two names and may be merged; two runs naming different
    # checkpoints are different experiments and must both survive, however similar
    # their numbers.
    def ckpt_of(rec):
        """The checkpoint a record was evaluated on, falling back to its log path."""
        return rec.get("ckpt") or rec.get("source", "")

    # The evaluator prints an explicit mAP block only at the stricter threshold, so a
    # Thames row arrives with a mean over classes at 0.7 and none at 0.5, which is the
    # figure the paper reports. Derive the missing means from the per-class columns that
    # were parsed from the same log. Verified against eval_london_perclass on 232
    # combinations with zero deviation, so this reproduces the evaluator rather than
    # approximating it. Rows that carry no per-class columns, i.e. single-class
    # SeePerSea, are left untouched.
    VESSELS = ("small_boat", "medium_boat", "large_boat")
    for r in rows:
        for rec in ("R40", "R11"):
            for view in ("bev", "3d"):
                for iou in ("0.5", "0.7"):
                    key = f"{rec}_{view}_{iou}_mAP"
                    per = [r.get(f"{rec}_{view}_{iou}_{c}") for c in VESSELS]
                    if all(v not in (None, "") for v in per):
                        r[key] = round(sum(float(v) for v in per) / len(per), 4)

    # A checkpoint that a runner log (CAM_/RE_) covers is reported under that name only. The
    # framework's own log_eval_*.txt rows exist to catch evaluations no runner recorded; for
    # checkpoints the runner did record they are older evaluations of the same weights (the
    # pre-threshold passes) and would reappear as duplicate rows with stale numbers.
    # (2026-08-21: 67 such rows had reached the appendix table under "Other".)
    runner_ckpts = {ckpt_of(r) for r in rows if "/" not in r["experiment"] and ckpt_of(r)}
    rows = [r for r in rows if "/" not in r["experiment"] or ckpt_of(r) not in runner_ckpts]

    by_sig = {}
    for r in rows:
        sig = (r.get("dataset"), ckpt_of(r), tuple(sorted(
            (k, v) for k, v in r.items() if k.startswith(("R11_", "R40_", "mAP")))))
        keep = by_sig.get(sig)
        if keep is None:
            by_sig[sig] = r
            continue
        # Prefer the runner tag over the framework's own output path as the row name.
        primary, secondary = (r, keep) if ("/" in keep["experiment"] and
                                           "/" not in r["experiment"]) else (keep, r)
        aliases = [a for a in (primary.get("alias", ""), secondary.get("alias", ""),
                               secondary["experiment"]) if a]
        primary["alias"] = "; ".join(dict.fromkeys(aliases))
        by_sig[sig] = primary
    rows = list(by_sig.values())

    meta = ["source", "alias", "dataset", "classes", "detector", "vfe", "map_to_bev", "backbone_3d",
            "backbone_2d", "dense_head", "roi_head", "train_split", "voxel_size",
            "pc_range", "batch_size", "epochs", "optimizer", "lr", "weight_decay",
            "grad_clip"]
    keys = [k for k in meta if any(k in r for r in rows)]
    keys += sorted({k for r in rows for k in r if k not in keys and k != "experiment"})
    # results/ is not shipped; the reporting scripts create it on first write.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] + keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["experiment"]):
            w.writerow(r)
    print(f"{len(rows)} evaluations -> {OUT}")
    print("classes found:", sorted(classes))
    print("metric columns:", len(keys))


if __name__ == "__main__":
    main()
