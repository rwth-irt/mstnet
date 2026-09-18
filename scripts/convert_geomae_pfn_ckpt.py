"""Convert a GeoMAE *learned-VFE* SSL checkpoint into an OpenPCDet MSTNet-C pretrained file.

Unlike the mean-VFE converter (convert_geomae_ckpt.py, SST only), this also carries
the learned pillar VFE. The GeoMAE checkpoint (ssl_seepersea_pfn10) was trained with
voxel_center_z=True + feat_channels=(32,64), so its VFE is structurally identical to
OpenPCDet's DynPillarVFE -- shapes match exactly ((32,10)+(64,64)); only the module
naming differs (mmdet3d '.0/.1' vs OpenPCDet '.linear/.norm').

Key maps:
  backbone.X                          -> map_to_bev_module.sst.X   (255 SST keys)
  voxel_encoder.pfn_layers.I.0.*      -> vfe.pfn_layers.I.linear.*
  voxel_encoder.pfn_layers.I.1.*      -> vfe.pfn_layers.I.norm.*

Usage: python convert_geomae_pfn_ckpt.py <ssl_pfn10_epoch.pth> <out.pth>
"""
import argparse, re
from pathlib import Path
import torch

SST_SRC, SST_DST = "backbone.", "map_to_bev_module.sst."
VFE_SRC = "voxel_encoder.pfn_layers."


def convert(src, dst):
    """Rewrite a pretraining checkpoint of the learned encoder variant.

    Same as convert_geomae_ckpt but for a run whose voxel encoder is a feature net
    rather than a mean, so the encoder weights transfer as well. The shape assertions
    at the end exist because a silent mismatch here produced a randomly initialized
    feature net under a pretrained transformer until 2026-08-17.
    """
    ck = torch.load(src, map_location="cpu")
    sd = ck["state_dict"]
    out = {}

    # SST: straight prefix swap
    for k, v in sd.items():
        if k.startswith(SST_SRC):
            out[SST_DST + k[len(SST_SRC):]] = v

    # VFE: rename mmdet3d PFNLayer (.0 linear / .1 norm) -> OpenPCDet (.linear / .norm)
    for k, v in sd.items():
        m = re.match(rf"{re.escape(VFE_SRC)}(\d+)\.(\d)\.(.+)", k)
        if not m:
            continue
        layer, sub, rest = m.group(1), m.group(2), m.group(3)
        part = "linear" if sub == "0" else "norm"
        out[f"vfe.pfn_layers.{layer}.{part}.{rest}"] = v

    n_sst = sum(1 for k in out if k.startswith(SST_DST))
    n_vfe = sum(1 for k in out if k.startswith("vfe."))

    # The whole point of this converter is the VFE. A source checkpoint whose voxel
    # encoder is named or shaped differently silently matched nothing here until
    # 2026-08-17, and every learned-encoder finetuning then ran with a randomly
    # initialised feature net under a pretrained transformer. Fail instead.
    expect = {"vfe.pfn_layers.0.linear.weight": (32, 10),
              "vfe.pfn_layers.1.linear.weight": (64, 64)}
    if n_vfe == 0:
        raise SystemExit(
            f"no VFE tensor matched '{VFE_SRC}*' in {src}. This checkpoint carries the "
            f"transformer alone, use convert_geomae_ckpt.py for a mean-encoder run or "
            f"pretrain with a DynamicPillarFeatureNet config such as ssl_*_pfn10.")
    for k, shape in expect.items():
        if k not in out:
            raise SystemExit(f"{k} missing after conversion, source keys do not match "
                             f"the detector's DynPillarVFE")
        if tuple(out[k].shape) != shape:
            raise SystemExit(f"{k} has shape {tuple(out[k].shape)}, the detector expects "
                             f"{shape}. Pretrain with voxel_center_z=True and "
                             f"feat_channels=(32, 64).")

    torch.save({"model_state": out, "epoch": ck.get("meta", {}).get("epoch", -1)}, dst)
    print(f"{Path(src).name} -> {dst}")
    print(f"  {n_sst} SST + {n_vfe} VFE tensors converted")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    a = p.parse_args()
    convert(a.src, a.dst)
