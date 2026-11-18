"""Measure inference latency and peak GPU memory for one PCDet model config.
Runs in whatever conda env invokes it (pcdet or ted). run_speed_sweep.py passes the
trained checkpoint of each reported run through --ckpt; without it, the model is
untrained, which leaves the timing of the forward pass but not of the
post-processing representative. Prints one JSON line.

Usage: benchmark_speed.py <cfg.yaml> [--frames 50] [--warmup 5]
"""
import argparse, json, time
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def main():
    """Measure one configuration and print its cost as one JSON line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    # The debug checks of the SST are assertions only; they leave the output unchanged but
    # cost time, so the inference time is measured without them, as in the paper.
    sst = cfg.MODEL.get("MAP_TO_BEV", {}).get("SST", None)
    if sst is not None:
        sst.debug = False
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    if args.ckpt:
        model.load_params_from_file(args.ckpt, logger=logger)
    model.cuda().eval()

    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            load_data_to_gpu(b)
            torch.cuda.synchronize(); t0 = time.time()
            try:
                model(b)
            except Exception as e:
                print(json.dumps({"cfg": args.cfg, "error": f"{type(e).__name__}: {str(e)[:120]}"}))
                return
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(time.time() - t0)
            if i >= args.warmup + args.frames:
                break

    ms = np.array(times) * 1000
    lat_ms = float(ms.mean())
    # The mean alone hides what a maritime scene does to a sparse model. Frames differ by
    # an order of magnitude in occupied cells, so the median says what a typical frame
    # costs and the ninetieth percentile says what the worst frames cost, which is the
    # number a real-time budget has to hold.
    out = {
        "cfg": args.cfg,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "latency_ms": round(lat_ms, 1),
        "median_ms": round(float(np.median(ms)), 1),
        "p90_ms": round(float(np.percentile(ms, 90)), 1),
        "fps": round(1000.0 / lat_ms, 1),
        "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 0),
        "n_frames": len(times),
        "ckpt": args.ckpt,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
