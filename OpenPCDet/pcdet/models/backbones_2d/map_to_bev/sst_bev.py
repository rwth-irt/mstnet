"""SST (Single-stride Sparse Transformer) as an OpenPCDet MAP_TO_BEV module.

This is the MSTNet encoder ported out of the project's own `src/` pipeline so it can
be trained and evaluated inside OpenPCDet next to every other benchmark model.
Motivation: the src framework measurably lags OpenPCDet on identical
architectures, which confounds any MSTNet-vs-baseline comparison. Running
MSTNet here removes that confound.

It slots into MAP_TO_BEV because its interface is exactly PointPillarScatter's --
(voxel_features [N,C], voxel_coords [N,4]) -> spatial_features [B,C,H,W] -- except
it runs sparse window attention over the pillar tokens before scattering them.

What follows the SST is deliberately left to BACKBONE_2D so the two designs can be
ablated against each other:
  A  BaseBEVBackbone            - downsampling multi-scale CNN (what src MSTNet uses)
  B  SSTOutConv-style           - stride-1 dilated convs (what the reference
                                  SST/GeoMAE uses, preserving single-stride)
The reference never downsamples after the SST; whether that matters on sparse
maritime data is an open question this port lets us answer.
"""
import copy

import torch.nn as nn

from .sst import MultiMAESSTSPChoose


def _drop_info(cfg_block):
    """YAML gives string keys ('0','1',...) and lists; the SST wants int keys/tuples."""
    return {
        int(k): {
            "max_tokens": v["max_tokens"],
            "drop_range": tuple(v["drop_range"]),
        }
        for k, v in cfg_block.items()
    }


class SSTBEV(nn.Module):
    """The single-stride sparse transformer in the map-to-bird's-eye-view slot.

    Replaces PointPillarScatter. Instead of writing the tokens onto an empty canvas
    it lets them attend to each other inside fixed windows first, then scatters the
    result at the same resolution. This is the module the masked pretraining
    initializes, and the only one MSTNet adds to CenterPoint besides the encoder.

    Everything under the SST config block is passed to the upstream implementation.
    Three fields are derived here rather than configured: the shift list, which is
    half a window in each direction; the output shape, which follows from the point
    cloud range and the voxel size; and the decoder, which stays unbuilt because
    detection uses the encoder alone.
    """

    def __init__(self, model_cfg, grid_size, point_cloud_range=None, voxel_size=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        args = copy.deepcopy(dict(model_cfg.SST))

        window_shape = tuple(args["window_shape"])
        sst_voxel_size = tuple(args["voxel_size"])
        pcr = list(point_cloud_range)

        args.update(
            window_shape=window_shape,
            voxel_size=sst_voxel_size,
            shifts_list=[(0, 0), (window_shape[0] // 2, window_shape[1] // 2)],
            drop_info=(_drop_info(args["drop_info"]["training"]),
                       _drop_info(args["drop_info"]["test"])),
            point_cloud_range=pcr,
            # [ny, nx] — row-major BEV, matching the src wrapper
            output_shape=[int(round((pcr[4] - pcr[1]) / sst_voxel_size[1])),
                          int(round((pcr[3] - pcr[0]) / sst_voxel_size[0]))],
            # detection uses the encoder only; the MAE reconstruction decoder would be
            # 1.15M never-invoked params inflating the reported model size
            build_decoder=False,
        )
        self.sst = MultiMAESSTSPChoose(**args)
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """Write `spatial_features`, the bird's-eye map of the encoded tokens.

        Takes the tokens from `pillar_features` or `voxel_features`, whichever the
        configured encoder wrote, runs window attention over them and scatters the
        result back to the bird's-eye grid.
        """
        # PillarVFE writes 'pillar_features'; MeanVFE writes 'voxel_features'.
        feats = batch_dict.get("pillar_features", batch_dict.get("voxel_features"))
        coords = batch_dict["voxel_coords"]
        bs = batch_dict["batch_size"]

        feats, ind_dict_list, voxel_info = self.sst.get_voxel_info(feats, coords, bs)
        tokens = self.sst.forward_encoder((feats, ind_dict_list, voxel_info))
        batch_dict["spatial_features"] = self.sst.recover_bev(tokens, voxel_info["coors"], bs)
        return batch_dict
