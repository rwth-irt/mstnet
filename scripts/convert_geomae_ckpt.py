"""Convert a GeoMAE SSL checkpoint into an OpenPCDet --pretrained_model file.

The SSL run stores the whole GeoMAE encoder+decoder under 'backbone.' inside
'state_dict'. Our ported SST lives at 'map_to_bev_module.sst.' (OpenPCDet names
the attribute after module_topology's 'map_to_bev_module'), and the key sets match
1:1 -- 255/255, no shape mismatches -- because it is literally the same
MultiMAESSTSPChoose class. So the conversion is a pure prefix swap.

The decoder weights come along and are simply never invoked: SSTBEV only calls
get_voxel_info/forward_encoder/recover_bev. Keeping them makes the load exact and
mirrors what src does.

Usage:
  python convert_geomae_ckpt.py <ssl_epoch_N.pth> <out.pth>
  train.py --cfg_file cfgs/custom_seepersea_models/mstnet.yaml --pretrained_model <out.pth>
"""
import argparse
from pathlib import Path

import torch

SRC_PREFIX = "backbone."
DST_PREFIX = "map_to_bev_module.sst."


def convert(src, dst):
    """Rewrite a pretraining checkpoint into the detector's key names.

    The pretraining and the detector hold the same transformer under different
    prefixes. Only the encoder is carried over; the reconstruction decoder has no
    counterpart in the detector and is dropped.
    """
    ck = torch.load(src, map_location="cpu")
    if "state_dict" not in ck:
        raise SystemExit(f"{src}: no 'state_dict' key (found {list(ck.keys())})")
    sd = ck["state_dict"]

    out = {DST_PREFIX + k[len(SRC_PREFIX):]: v
           for k, v in sd.items() if k.startswith(SRC_PREFIX)}
    if not out:
        raise SystemExit(f"{src}: no '{SRC_PREFIX}*' keys to convert")

    skipped = len(sd) - len(out)
    # 'model_state' is the key OpenPCDet's load_params_from_file reads.
    torch.save({"model_state": out, "epoch": ck.get("meta", {}).get("epoch", -1)}, dst)
    print(f"{Path(src).name} -> {dst}")
    print(f"  converted {len(out)} tensors ({SRC_PREFIX}* -> {DST_PREFIX}*)"
          + (f", skipped {skipped} non-backbone" if skipped else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    a = p.parse_args()
    convert(a.src, a.dst)
