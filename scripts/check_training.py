"""Static checks on the training setup, before the full retraining.

Static checks only; the smoke test that actually runs a model is separate. Every check
prints PASS or FAIL with the measured value, so the output is a record of the state the
runs started from.
"""
import glob
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SSL = ROOT / "data" / "ssl_nuscenes"
random.seed(0)
fails = []


def report(ok, name, detail=""):
    """Print one check result and record it if it failed."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def sample_intensity(files, npy=False, n=60):
    """Intensity values of a random sample of frames, to check they share one scale."""
    v = []
    for f in random.sample(files, min(n, len(files))):
        a = np.load(f) if npy else np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        v.append(a[:, 3])
    return np.concatenate(v)


print("1. Intensity is on the unit interval everywhere")
for name in ["seepersea", "london", "kitti", "pohang", "pohang_seepersea", "pohang_london"]:
    fs = sorted(glob.glob(str(SSL / name / "**" / "*.bin"), recursive=True))
    if not fs:
        report(False, f"pretraining {name}", "no files")
        continue
    v = sample_intensity(fs)
    report(0.0 <= v.min() and v.max() <= 1.0, f"pretraining {name}",
           f"min {v.min():.3f} max {v.max():.3f} p50 {np.percentile(v, 50):.3f}")
for ds in ["custom_seepersea", "custom_london"]:
    fs = sorted(glob.glob(str(DATA / ds / "points" / "*.npy")))
    v = sample_intensity(fs, npy=True)
    report(0.0 <= v.min() and v.max() <= 1.0, f"detection {ds}",
           f"min {v.min():.3f} max {v.max():.3f} p50 {np.percentile(v, 50):.3f}")

print("\n2. The same frame is identical in both paths")
det = DATA / "custom_seepersea" / "points"
same = diff = 0
for f in sorted(glob.glob(str(SSL / "seepersea" / "**" / "*.bin"), recursive=True))[:300]:
    d = det / f"{Path(f).stem}.npy"
    if not d.exists():
        continue
    a = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    b = np.load(d)
    if a.shape == b.shape and np.abs(a - b).max() < 1e-6:
        same += 1
    else:
        diff += 1
report(diff == 0 and same > 0, "SeePerSea pretraining equals detection",
       f"{same} identical, {diff} differing")

print("\n3. Budgets are nested and match their info files")
for ds in ["custom_seepersea", "custom_london"]:
    B = DATA / ds
    s = {p: set((B / f"ImageSets/train_{p}pct.txt").read_text().split()) for p in
         [5, 10, 20, 50, 80]}
    s["full"] = set((B / "ImageSets/train.txt").read_text().split())
    order = [5, 10, 20, 50, 80, "full"]
    nested = all(s[a] <= s[b] for a, b in zip(order, order[1:]))
    report(nested, f"{ds} nested", " ".join(f"{p}%:{len(s[p])}" for p in [5, 10, 20, 50, 80]))
    ok = True
    for p in [5, 10, 20, 50, 80]:
        got = {i["point_cloud"]["lidar_idx"]
               for i in pickle.load(open(B / f"custom_infos_train_{p}pct.pkl", "rb"))}
        ok &= got == s[p]
    report(ok, f"{ds} info files match the splits")

print("\n4. No frame is in both training and validation")
for ds in ["custom_seepersea", "custom_london"]:
    B = DATA / ds / "ImageSets"
    tr = set((B / "train.txt").read_text().split())
    va = set((B / "val.txt").read_text().split())
    report(len(tr & va) == 0, f"{ds} splits disjoint", f"overlap {len(tr & va)}")

print("\n5. Sampling databases are on the new scale and drawn from their own budget")
for ds in ["custom_seepersea", "custom_london"]:
    B = DATA / ds
    for sub, split in [("gt_database", "train"), ("gt_database_train_5pct", "train_5pct")]:
        fs = sorted(glob.glob(str(B / sub / "*.bin")))
        if not fs:
            report(False, f"{ds}/{sub}", "empty")
            continue
        v = sample_intensity(fs, n=200)
        report(v.max() <= 1.0, f"{ds}/{sub} scale",
               f"{len(fs)} objects, max {v.max():.3f}")
        import re
        d = pickle.load(open(B / f"custom_dbinfos_{split}.pkl", "rb"))
        ids = set((B / f"ImageSets/{split}.txt").read_text().split())
        used = set()
        for k in d:
            for e in d[k]:
                m = re.match(r"^(.*)_" + re.escape(k) + r"_\d+\.bin$", Path(e["path"]).name)
                if m:
                    used.add(m.group(1))
        report(not (used - ids), f"{ds}/{split} database source frames",
               f"{len(used)} frames, {len(used - ids)} outside")

print("\n6. No stale results remain")
for p, name in [(ROOT / "results" / "all_results.csv", "all_results.csv"),
                (ROOT / "benchmark_speed.csv", "benchmark_speed.csv")]:
    report(not p.exists(), f"{name} cleared")
for d in [ROOT / "OpenPCDet" / "output", ROOT / "TED" / "output",
          ROOT / "LION" / "output"]:
    n = len(list(d.glob("*"))) if d.exists() else -1
    report(n == 0, f"{d.parent.name} output tree empty", f"{n} entries")
report(len(list((ROOT / "runs" / "logs").glob("*"))) == 0, "log directory empty")

print("\n" + ("all checks passed" if not fails else f"{len(fails)} FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
