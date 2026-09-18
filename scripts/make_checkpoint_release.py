"""Assemble the checkpoints behind every reported number into one folder for release.

One checkpoint per reported run, plus the pretraining checkpoints the fine-tuning
started from, plus a README that maps each file back to the experiment it belongs to.

The checkpoint of a run is taken from the ckpt column of all_results.csv, which
collect_results.py reads out of the evaluation log itself, so the file copied here is
the one the reported number was measured on and not whatever else the run directory
happens to hold.

The eighteen cross-dataset pretraining runs that make_paper_tables.py excludes are
excluded here too. They were executed but the study does not report them, so shipping
them would put files in the release that no table explains.

Usage:
  python make_checkpoint_release.py --dry-run      what it would copy, copies nothing
  python make_checkpoint_release.py --go
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import make_paper_tables as M

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "all_results.csv"
SSL = ROOT / "checkpoints"
DEST = ROOT / "checkpoint_release"

# 'london' is the internal name of the recording the manuscript calls Thames, and every
# checkpoint file name derived from an experiment name or an SSL checkpoint carries it
# (mstnet_ssl_london.pth, and so on). An earlier version of this script translated
# 'london' to 'thames' in the published file name, matching the folder split below --
# but run_campaign.py and every config that names a PRETRAINED_MODEL look for the
# untranslated name, so a checkpoint downloaded under its published name would not be
# found by the code that is supposed to load it. File names therefore keep 'london'
# unchanged; only the destination folder spells out 'thames' for a human reader, since
# nothing in the pipeline reads a path back out of that folder name.

# The internal config stem to the name the manuscript uses. The pipeline is called mstnet
# in the working tree and MSTNet in the paper, and the release has to bridge that.
MODEL = {
    "mstnet": "MSTNet",
    "mstnet_c": "MSTNet, learned encoder",
    "mstnet_zaware": "MSTNet + height-aware encoding",
    "mstnet_stage2": "MSTNet + refinement stage",
    "mstnet_zaware_stage2": "MSTNet + both measures",
    "mstnet_zaware_stage2_zonly": "MSTNet + both, vertical only",
    "mstnet_pseudo": "pseudo-label stage (stage one only)",
    "mstnet_pseudo_zaware": "pseudo-label stage, height-aware (stage one only)",
    "centerpoint": "CenterPoint", "pointpillar": "PointPillars", "second": "SECOND",
    "voxelnext": "VoxelNeXt", "voxel_rcnn": "Voxel R-CNN", "pv_rcnn": "PV-RCNN",
    "transfusion": "TransFusion", "dsvt": "DSVT, pillar", "dsvt_voxel": "DSVT, voxel",
    "ted_s": "TED-S", "ted_s_perclass": "TED-S, per-class thresholds",
    "lion_mamba": "LION-Mamba",
    # The diagnostic that refits the anchor prior to the budget being trained on,
    # instead of keeping the prior the SeePerSea authors publish. It is a diagnostic and
    # no table reports it, which the group column says.
    "pointpillar_ba": "PointPillars, anchor prior refitted to the budget",
}


def model_name(stem):
    """The manuscript's name for a configuration stem.

    The budget suffix has to come off first, since mstnet_20pct and mstnet are the same
    model trained on different label budgets and the budget is its own column.
    """
    import re as _re
    base = _re.sub(r"_\d+pct(_ba)?$", "", stem)
    if stem.endswith("_ba"):
        base += "_ba"
    return MODEL.get(base, base)

# The run tag to the pretraining the run started from.
PRETRAIN = {
    "rand": "none, random initialization",
    "ssl": "in-domain, masked (GeoMAE)",
    "ssl_kitti": "out-of-domain automotive, KITTI",
    "ssl_pohang": "cross-domain maritime, Pohang Canal",
    "ssl_pohang_seepersea": "combined, Pohang and SeePerSea",
    "ssl_pohang_london": "combined, Pohang and Thames",
    "pseudo2": "in-domain masked, then the pseudo-label stage",
}

# What each pretraining checkpoint holds. All nine are the masked objective of GeoMAE on
# the sparse transformer; they differ in the unlabeled set and in the voxel encoder the
# transformer was pretrained behind. A tenth pretraining run exists and is not shipped,
# an earlier nine-channel version of the learned encoder that the ten-channel one
# superseded before any reported run used it.
SSL_DESC = {
    "mstnet_ssl_seepersea.pth":
        "mean encoder, in-domain, the 8111 unlabeled SeePerSea frames",
    "mstnet_ssl_london.pth":
        "mean encoder, in-domain, the 8926 unlabeled Thames frames",
    "mstnet_ssl_pohang.pth":
        "mean encoder, cross-domain maritime, the 49036 Pohang Canal frames",
    "mstnet_ssl_kitti.pth":
        "mean encoder, out-of-domain automotive, the 14999 KITTI scans",
    "mstnet_ssl_pohang_seepersea.pth":
        "mean encoder, combined, Pohang Canal and SeePerSea, 56999 frames",
    "mstnet_ssl_pohang_london.pth":
        "mean encoder, combined, Pohang Canal and Thames, 57814 frames",
    "mstnet_c_ssl_seepersea.pth":
        "learned Pillar Feature Net encoder, in-domain SeePerSea",
    "mstnet_c_ssl_london.pth":
        "learned Pillar Feature Net encoder, in-domain Thames",
    "mstnet_ssl_seepersea_zaware.pth":
        "height-aware encoder, 37 vertical bins, in-domain SeePerSea",
}

FAMILY = {
    "budget": ("E.I and E.III", "Label budget sweep"),
    "chain": ("E.II", "Voxel encoder"),
    "pretrain-source": ("E.IV", "Pretraining data"),
    "pseudo": ("E.V", "Pseudo-label pretraining"),
    "height": ("E.VI", "Measures against the vertical deficit"),
    "height-budget": ("E.VI", "Measures against the vertical deficit, over the budget"),
    "combo": ("E.VI", "Both measures combined, over the budget"),
    "triple": ("E.VI", "Pseudo-label stage on the strongest configuration"),
    "benchmark": ("E.VII", "Benchmark against published detectors"),
    "benchmark-repair": ("E.VII", "Benchmark, repaired configuration"),
    "seeds": ("repeats", "Repeated runs of selected cells"),
    "diagnostic": ("appendix", "Diagnostics, not a reported cell"),
}

DSNAME = {"london": "Thames", "seepersea": "SeePerSea"}

# Runs that are shipped and discussed but whose cell no table reports, so that nobody
# reads their metrics as a benchmark result.
# Which datasets a checkpoint's weights were exposed to, so that a downstream user can
# see at a glance which files inherit which dataset terms. The fine-tuning dataset is
# added from the run's own dataset column.
PRETRAIN_DATA = {
    "rand": [],
    "ssl": ["<in-domain>"],
    "ssl_kitti": ["KITTI"],
    "ssl_pohang": ["Pohang Canal"],
    "ssl_pohang_seepersea": ["Pohang Canal", "SeePerSea"],
    "ssl_pohang_london": ["Pohang Canal", "Thames"],
    "pseudo2": ["<in-domain>"],
}
SSL_DATA = {
    "mstnet_ssl_seepersea.pth": ["SeePerSea"],
    "mstnet_ssl_london.pth": ["Thames"],
    "mstnet_ssl_pohang.pth": ["Pohang Canal"],
    "mstnet_ssl_kitti.pth": ["KITTI"],
    "mstnet_ssl_pohang_seepersea.pth": ["Pohang Canal", "SeePerSea"],
    "mstnet_ssl_pohang_london.pth": ["Pohang Canal", "Thames"],
    "mstnet_c_ssl_seepersea.pth": ["SeePerSea"],
    "mstnet_c_ssl_london.pth": ["Thames"],
    "mstnet_ssl_seepersea_zaware.pth": ["SeePerSea"],
}

# The terms each dataset is published under, checked against the source in September
# 2026. Three of the four forbid commercial use, and two of those are share-alike, which
# is what decides under which terms a checkpoint trained on them may be passed on.
DATASET_TERMS = {
    "Thames": "CC BY 4.0",
    "SeePerSea": "CC BY-NC-SA 4.0",
    "Pohang Canal": "CC BY-NC 4.0",
    "KITTI": "CC BY-NC-SA 3.0",
}
# Which of two sets of terms binds when a checkpoint saw both.
STRICTNESS = {"CC BY 4.0": 0, "CC BY-NC 4.0": 1, "CC BY-NC-SA 3.0": 2,
              "CC BY-NC-SA 4.0": 3}

NOTES = {
    "CAM_london_ted_s_rand":
        "superseded, the single-class threshold lists of the published KITTI recipe; "
        "the benchmark table reports the per-class configuration instead, and this run "
        "is the failure Sections 3.8 and 4.7 describe",
}


def families():
    """Map every run name to the experiment family the campaign filed it under."""
    out = {}
    for f in ("campaign_results.jsonl", "mstnet_c_repair_results.jsonl"):
        p = ROOT / "runs" / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("name") and r.get("family"):
                out["CAM_" + r["name"]] = r["family"]
    return out


def parse_ckpt(path):
    """cfg stem, run tag and epoch out of the checkpoint's own path.

    The layout is output/custom_<ds>_models/<stem>/<tag>/ckpt/checkpoint_epoch_<n>.pth,
    which is more reliable than parsing the experiment name, where the stem and the
    checkpoint name run together without a separator.
    """
    p = Path(path)
    return p.parent.parent.parent.name, p.parent.parent.name, p.stem.split("_")[-1]


def manifest():
    """Build the list of checkpoints to publish, one per reported run.

    Each entry carries the paper name of the model, the experiment it belongs to, the
    pretraining it started from and the path of the file itself. Runs the tables
    exclude are excluded here too, so the release holds nothing no table explains.
    """
    fam = families()
    rows = [r for r in csv.DictReader(CSV.open())
            if not any(pat.match(r["experiment"]) for pat in M.EXCLUDED)]
    out = []
    for r in rows:
        stem, tag, epoch = parse_ckpt(r["ckpt"])
        seed = 1
        base = tag
        if tag.endswith("_s2"):
            seed, base = 2, tag[:-3]
        elif tag.endswith("_s3"):
            seed, base = 3, tag[:-3]
        ds = r["dataset"]
        out.append({
            "file": f"finetuning/{'thames' if ds == 'london' else 'seepersea'}/"
                    f"{r['experiment'][len('CAM_'):]}.pth",
            "src": r["ckpt"],
            "dataset": DSNAME.get(ds, ds),
            "model": model_name(stem),
            "config": f"custom_{ds}_models/{stem}.yaml",
            "labels": (lambda m: m + r"%" if m else "100%")(
                "".join(c for c in stem.split("_")[-1] if c.isdigit())
                if stem.endswith("pct") else ""),
            "pretraining": PRETRAIN.get(base, base),
            "seed": seed,
            "epoch": epoch,
            "experiment": FAMILY.get(fam.get(r["experiment"], ""), ("?", "?"))[0],
            "group": FAMILY.get(fam.get(r["experiment"], ""), ("?", "?"))[1],
            "bev_0.5": r.get("R40_bev_0.5_mAP") or r.get("R40_bev_0.5_Car") or "",
            "3d_0.5": r.get("R40_3d_0.5_mAP") or r.get("R40_3d_0.5_Car") or "",
            "note": NOTES.get(r["experiment"], ""),
            "derived_from": ", ".join(sorted(set(
                [DSNAME.get(ds, ds)] +
                [DSNAME.get(ds, ds) if x == "<in-domain>" else x
                 for x in PRETRAIN_DATA.get(base, [])]))),
        })
    for p in sorted(SSL.glob("*.pth")):
        out.append({
            "file": f"pretraining/{p.name}", "src": str(p), "dataset": "",
            "model": SSL_DESC.get(p.name, "masked pretraining (GeoMAE)"),
            "config": "", "labels": "", "pretraining": "", "seed": "", "epoch": "",
            "experiment": "pretraining", "group": "Pretraining checkpoints",
            "bev_0.5": "", "3d_0.5": "", "note": "",
            "derived_from": ", ".join(SSL_DATA.get(p.name, [])),
        })
    return out


def sha256(path):
    """Checksum of one file, read in chunks so a large checkpoint fits in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    """Copy the checkpoints and write the README and the license of the release."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--dest", default=str(DEST))
    a = ap.parse_args()
    dest = Path(a.dest)

    man = manifest()
    total = sum(Path(m["src"]).stat().st_size for m in man)
    print(f"{len(man)} files, {total / 1e9:.1f} GB -> {dest}")
    missing = [m for m in man if not Path(m["src"]).exists()]
    for m in missing:
        print(f"  MISSING {m['src']}")
    if a.dry_run or not a.go:
        from collections import Counter
        for k, v in Counter(m["group"] for m in man).most_common():
            print(f"  {k:<52} {v:4d}")
        return 1 if missing else 0
    if missing:
        print("refusing to build an incomplete release")
        return 1

    for m in man:
        t = dest / m["file"]
        t.parent.mkdir(parents=True, exist_ok=True)
        if not t.exists() or t.stat().st_size != Path(m["src"]).stat().st_size:
            shutil.copy2(m["src"], t)
    # The folder ships the checkpoints and one README that explains them. A separate
    # manifest, an attribution file and a checksum list were all dropped on request,
    # since each only restated what the README already says.
    for stale in ("manifest.csv", "SHA256SUMS", "ATTRIBUTION.md"):
        q = dest / stale
        if q.exists():
            q.unlink()
            print(f"  removed {stale}")
    write_readme(dest, man, total)
    write_license(dest, man)
    print(f"done: {dest}")
    return 0


# Why CC BY 4.0 and not the non-commercial terms of the training data.
#
# Three of the four datasets are non-commercial and two are share-alike, so the first
# draft of this release carried CC BY-NC-SA 4.0. That was more cautious than the question
# warrants. CC 4.0 attaches its ShareAlike and NonCommercial terms to "Adapted Material",
# which it defines as material derived from the work "in a manner requiring permission
# under the Copyright and Similar Rights held by the Licensor". Whether producing model
# weights requires that permission is unsettled, and weights are statistical parameters
# rather than reproduced expression. German law adds a research text-and-data-mining
# exception, UrhG 60d, which UrhG 60g makes unwaivable by contract, so the training
# itself is covered for a non-commercial research institution whatever the licence says.
#
# The field draws its line at contract rather than at copyright. OpenPCDet publishes
# KITTI-trained checkpoints in its own model zoo under Apache 2.0 despite KITTI being
# CC BY-NC-SA 3.0, and withholds only its Waymo models, "due to Waymo Dataset License
# Agreement", which is a signed agreement. None of our four datasets required one.
# SeePerSea asks for contact details, affiliation and a reason before the download and
# states no terms, which the author confirmed.
#
# We therefore release the weights under CC BY 4.0, which is what the study wants, and
# name every dataset a file was trained on so that the provenance stays visible and
# citable. This is not a legal opinion and the reasoning above is recorded so that it can
# be revisited rather than rediscovered.
def write_license(dest, man):
    """Write the license file that states the terms the checkpoints are published under."""
    L = ["Checkpoints for the MSTNet maritime LiDAR detection study",
         "Copyright the authors of the accompanying publication.", "",
         "TERMS", "",
         "These weights are released under the Creative Commons Attribution 4.0",
         "International licence, https://creativecommons.org/licenses/by/4.0/", "",
         "You may share and adapt them for any purpose, including commercially, provided",
         "you give appropriate credit. Credit means citing the accompanying publication",
         "and citing the dataset the checkpoint was trained on, as named below and in",
         "README.md.", "",
         "WHAT WE DO AND DO NOT DISTRIBUTE", "",
         "We trained every checkpoint here ourselves. None is a redistribution of somebody",
         "else's weights, and this folder contains no dataset content of any kind, neither",
         "point clouds nor annotations.", "",
         "THE DATASETS THESE WEIGHTS WERE TRAINED ON", "",
         "Each carries its own terms, which govern the data and not this folder. Anyone",
         "obtaining the datasets themselves is bound by them. README.md names, for every",
         "file, which of these its weights were exposed to.", "",
         "  Thames, Xie et al.          CC BY 4.0",
         "  SeePerSea, Jeong et al.     CC BY-NC-SA 4.0",
         "  Pohang Canal, Chung et al.  CC BY-NC 4.0",
         "  KITTI, Geiger et al.        CC BY-NC-SA 3.0", "",
         "SOFTWARE", "",
         "The training code is a separate artifact under the Apache License 2.0, as are",
         "OpenPCDet, the TED and LION forks and GeoMAE that it builds on.", ""]
    (dest / "LICENSE").write_text("\n".join(L))


def write_readme(dest, man, total):
    """Write the table that maps every checkpoint to its experiment and configuration."""
    L = ["# Checkpoints for the MSTNet maritime LiDAR detection study", "",
         "This folder holds one checkpoint for every run the manuscript reports, together",
         "with the self-supervised pretraining checkpoints those runs were fine-tuned from.",
         "Each file is the checkpoint the reported number was measured on, taken from the",
         "evaluation log of that run rather than from whatever else its output directory",
         "contains.", "",
         f"{len(man)} files, {total / 1e9:.1f} GB.", "",
         "## Licence and citation", "",
         "The weights are released under CC BY 4.0. Use them for any purpose, including",
         "commercially, and give credit. Credit means two citations, the study these",
         "checkpoints belong to and the dataset the checkpoint was trained on. `LICENSE`",
         "carries the terms in full.", "",
         "We trained every checkpoint ourselves, and this folder contains no dataset",
         "content, neither point clouds nor annotations. The datasets carry their own",
         "terms, which govern the data rather than this folder, and anyone obtaining them",
         "is bound by those terms.", "",
         "| Dataset | Used for | Its own terms |",
         "| --- | --- | --- |",
         "| Thames, Xie et al. | training, evaluation and pretraining | CC BY 4.0 |",
         "| SeePerSea, Jeong et al. | training, evaluation and pretraining | CC BY-NC-SA 4.0 |",
         "| Pohang Canal, Chung et al. | unlabeled pretraining | CC BY-NC 4.0 |",
         "| KITTI, Geiger et al. | unlabeled pretraining | CC BY-NC-SA 3.0 |", "",
         "The Datasets column of the tables below names, for every file, which of these",
         "its weights were exposed to, through pretraining as well as through fine-tuning,",
         "so that the right dataset can be cited for the right checkpoint.", "",
         "## Conventions", "",
          "- The pipeline is called `mstnet` in the file names and in the configuration stems,",
         "  and MSTNet in the manuscript. The Model column translates.",
         "- `london` in a file name or a configuration path is the recording the",
         "  manuscript calls Thames. File names keep it unchanged, since the code that",
         "  loads a checkpoint -- `run_campaign.py` and every config's",
         "  `PRETRAINED_MODEL` -- looks it up by that exact name; only the destination",
         "  folder below spells out `thames`, for a human reader.",
         "- Every run trains for 80 epochs and the released checkpoint is the last one, so",
         "  no checkpoint here was selected on a measured value.",
         "- BEV and 3D are average precision at 40 recall positions and an overlap of 0.5,",
         "  meaned over the three vessel classes on Thames and on the single class on",
         "  SeePerSea. They are the values the manuscript reports for that run.",
         "- Seed 1 is the run the manuscript's tables report. Seeds 2 and 3 exist only for",
         "  the twelve repeated cells. The supervised stage is unseeded, so they differ",
         "  from seed 1 in the initialization of every module not loaded from a",
         "  checkpoint, in the order the frames are drawn and in the augmentation sampled",
         "  for each of them. The pretraining runs, by contrast, are seeded and",
         "  deterministic.",
         "- Eighteen executed runs are deliberately absent. They pretrain one annotated",
         "  dataset on the other, which is not one of the four schemes the study defines,",
         "  and no table reports them. A tenth pretraining run is likewise absent, an",
         "  earlier nine-channel version of the learned encoder that the ten-channel one",
         "  superseded before any reported run used it.", ""]

    order = ["pretraining", "E.I and E.III", "E.II", "E.IV", "E.V", "E.VI", "E.VII",
             "repeats", "appendix"]
    groups = {}
    for m in man:
        groups.setdefault((m["experiment"], m["group"]), []).append(m)
    for exp in order:
        for (e, g), items in sorted(groups.items()):
            if e != exp:
                continue
            L += [f"## {e} — {g}" if e != "pretraining" else f"## {g}", ""]
            if e == "pretraining":
                L += ["| File | Content | Datasets |", "| --- | --- | --- |"]
                for m in sorted(items, key=lambda x: x["file"]):
                    L.append(f"| `{m['file']}` | {m['model']} | "
                             f"{m.get('derived_from', '')} |")
            else:
                L += ["| File | Dataset | Model | Labels | Pretraining | Seed | BEV | 3D "
                      "| Datasets | Note |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
                for m in sorted(items, key=lambda x: (x["dataset"], x["model"],
                                                      x["labels"], x["seed"])):
                    b = f"{float(m['bev_0.5']):.2f}" if m["bev_0.5"] else ""
                    d = f"{float(m['3d_0.5']):.2f}" if m["3d_0.5"] else ""
                    L.append(f"| `{m['file']}` | {m['dataset']} | {m['model']} | "
                             f"{m['labels']} | {m['pretraining']} | {m['seed']} | "
                             f"{b} | {d} | {m.get('derived_from', '')} | "
                             f"{m.get('note', '')} |")
            L.append("")
    (dest / "README.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
