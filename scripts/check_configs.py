"""Check every benchmark configuration before a full retraining.

The point is to catch anything that would force a second retraining. Each check states
what it expects and why, and prints only what deviates, so a clean run is a short run.
Resolves _BASE_CONFIG_ inheritance through the framework's own loader, because a
list-valued key such as DATA_PROCESSOR is replaced wholesale by config merging and a
partial override silently drops the rest.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = {
    "OpenPCDet": ROOT / "OpenPCDet" / "tools",
    "TED": ROOT / "TED" / "tools",
    "LION": ROOT / "LION" / "tools",
}
DETECTORS = ["pointpillar", "second", "centerpoint", "voxel_rcnn", "pv_rcnn",
             "dsvt", "voxelnext", "transfusion", "mstnet"]

EXPECT = {
    "seepersea": {"range": [-100, -100, -20, 100, 148, 25], "classes": ["Ship"]},
    "london": {"range": [-100, -100, -20, 100, 100, 35],
               "classes": ["small_boat", "medium_boat", "large_boat"]},
}
# MSTNet runs at the rate with the lower final training loss (Section 2.4.2 of the paper).
EXPECT_LR = {"mstnet": 0.001}
# Deviations that are intrinsic to a method and are documented rather than forced away.
ALLOWED = {
    "dsvt": "window shape needs a grid that is a multiple of 12, hence the padded range",
    "lion_mamba": "window shape needs 32 vertical bins, hence the coarser z voxel",
    "pv_rcnn": "learning rate of the reference configuration",
    "transfusion": "learning rate and gradient clip of the reference configuration",
}


def load(fw, cfg):
    """_BASE_CONFIG_ is resolved relative to the working directory, so the loader has
    to run from the framework's tools directory."""
    import importlib
    import os
    tools = FRAMEWORKS[fw]
    prev = os.getcwd()
    sys.path.insert(0, str(tools.parent))
    for m in [m for m in list(sys.modules) if m.startswith("pcdet")]:
        del sys.modules[m]
    try:
        os.chdir(tools)
        cfgmod = importlib.import_module("pcdet.config")
        c = cfgmod.EasyDict()
        cfgmod.cfg_from_yaml_file(cfg, c)
    finally:
        os.chdir(prev)
        sys.path.pop(0)
    return c


def processor(c, name):
    """Return the named entry of the data processor chain, or None."""
    for s in c.DATA_CONFIG.DATA_PROCESSOR:
        if s.NAME == name:
            return s
    return None


def check(fw, ds, det):
    """Resolve one configuration and return its deviations from the shared protocol.

    A deviation is anything the protocol table of the appendix would have to explain:
    a different range, cell size, epoch count, batch size, optimizer, learning rate or
    suppression setting. The four intrinsic exceptions printed at the end are known
    and are not reported here.
    """
    cfg_rel = f"cfgs/custom_{ds}_models/{det}.yaml"
    if not (FRAMEWORKS[fw] / cfg_rel).exists():
        return [f"{fw}/{ds}/{det}: config missing"]
    try:
        c = load(fw, cfg_rel)
    except Exception as e:
        return [f"{fw}/{ds}/{det}: will not load, {type(e).__name__}: {str(e)[:90]}"]

    out = []
    exp = EXPECT[ds]
    tag = f"{ds}/{det}"
    allow = ALLOWED.get(det, "")

    if list(c.DATA_CONFIG.POINT_CLOUD_RANGE) != exp["range"] and not allow:
        out.append(f"{tag}: point cloud range {list(c.DATA_CONFIG.POINT_CLOUD_RANGE)}")
    if list(c.CLASS_NAMES) != exp["classes"]:
        out.append(f"{tag}: classes {list(c.CLASS_NAMES)}")

    sh = processor(c, "shuffle_points")
    if sh is None:
        out.append(f"{tag}: no shuffle_points step")
    elif not (sh.SHUFFLE_ENABLED.get("train") and sh.SHUFFLE_ENABLED.get("test")):
        out.append(f"{tag}: shuffling {dict(sh.SHUFFLE_ENABLED)}, both must be true")

    vx = processor(c, "transform_points_to_voxels") or \
        processor(c, "transform_points_to_voxels_placeholder")
    if vx is None:
        out.append(f"{tag}: no voxelization step")
    else:
        if "VOXEL_SIZE" in vx and list(vx.VOXEL_SIZE)[:2] != [0.5, 0.5]:
            out.append(f"{tag}: horizontal voxel {list(vx.VOXEL_SIZE)[:2]}")
        cap = vx.get("MAX_POINTS_PER_VOXEL", None)
        if cap is not None and cap < 128:
            out.append(f"{tag}: per-cell point cap {cap}, expected 128")

    o = c.OPTIMIZATION
    if o.NUM_EPOCHS != 80:
        out.append(f"{tag}: {o.NUM_EPOCHS} epochs")
    if abs(o.LR - EXPECT_LR.get(det, 0.003)) > 1e-9 and not allow:
        out.append(f"{tag}: learning rate {o.LR}")

    pp = c.MODEL.get("POST_PROCESSING", {})
    dh = c.MODEL.get("DENSE_HEAD", {}).get("POST_PROCESSING", {})
    th = [t for t in (pp.get("SCORE_THRESH"), dh.get("SCORE_THRESH")) if t is not None]
    if th and min(th) > 0.1:
        out.append(f"{tag}: score threshold {th}")

    aug = [a.NAME for a in c.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    want = ["gt_sampling", "random_world_flip", "random_world_rotation",
            "random_world_scaling"]
    if aug != want:
        out.append(f"{tag}: augmentation {aug}")
    return out


def main():
    """Check every configuration of both datasets and print the deviations."""
    problems = []
    for ds in ("seepersea", "london"):
        for det in DETECTORS:
            problems += check("OpenPCDet", ds, det)
        # On Thames, TED-S runs with one RoI assignment threshold per class (ted_s_perclass).
        problems += check("TED", ds, "ted_s_perclass" if ds == "london" else "ted_s")
        problems += check("LION", ds, "lion_mamba")

    if problems:
        print(f"{len(problems)} deviations\n")
        for p in problems:
            print("  " + p)
    else:
        print("no deviations")
    print("\nIntrinsic and documented, not reported above:")
    for k, v in ALLOWED.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
