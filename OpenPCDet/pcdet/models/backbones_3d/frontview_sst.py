"""Front-view (x,z) token stream for MSTNet multi-view fusion (idea 3).

Second, independent tokenizer + SST encoder that reads the 'voxels_fv'/'voxel_coords_fv'/
'voxel_num_points_fv' keys written by transform_points_to_front_view_voxels (own,
from-scratch weights -- no GeoMAE SSL available for this branch). Plugs into the normally
unused BACKBONE_3D slot, which runs after 'vfe' and before 'map_to_bev_module' -- exactly
the ordering needed to produce 'spatial_features_fv' before HeightProfileFusion consumes it.

Mirrors SSTBEV (pcdet/models/backbones_2d/map_to_bev/sst_bev.py) almost exactly, with one
critical difference: the front-view DATA_PROCESSOR step collapses y (not z), so spconv's
voxel generator (always [z_idx, y_idx, x_idx] order) puts the RESOLVED height index in
column 1 and the dummy collapsed-y in column 2. MultiMAESSTSPChoose's window_partition/
recover_bev only ever read columns 2 and 3 -- so columns 1 and 2 are swapped here before
the SST sees them, moving resolved height into the row-index slot it actually uses.
Verified empirically (see project plan): pre-swap column 1 has ~36 distinct values, column
2 is uniformly 0.
"""
import copy

import torch
import torch.nn as nn

from ..backbones_2d.map_to_bev.sst import MultiMAESSTSPChoose
from ..backbones_2d.map_to_bev.sst_bev import _drop_info


class FrontViewSSTBranch(nn.Module):
    """A second token stream over the (x, z) plane, as a source of vertical evidence.

    The same transformer as SSTBEV, but its grid spans x and height rather than x and
    y, so the tokens carry the vertical structure the bird's-eye view discards.
    HeightProfileFusion folds the resulting canvas back into the bird's-eye pipeline.

    The upstream implementation names its two grid axes after x and y and reads the
    point cloud range only at indices 0, 3 and 1, 4. The height range is therefore
    passed in the slots the implementation calls y, and the true z entries are never
    read; the comment at the construction of `pc_range_sst` records this.
    """

    def __init__(self, model_cfg, input_channels, point_cloud_range=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = input_channels  # passthrough; build_backbone_3d reads this

        args = copy.deepcopy(dict(model_cfg.SST))
        window_shape = tuple(args["window_shape"])
        vx, vz = args.pop("voxel_size_xz")  # front-view resolution, e.g. [0.5, 0.5]
        # not a MultiMAESSTSPChoose constructor arg; consumed here to build voxel_size below

        pcr = list(point_cloud_range)  # real 3D range [xmin,ymin,zmin,xmax,ymax,zmax]
        x_min, x_max = pcr[0], pcr[3]
        z_min, z_max = pcr[2], pcr[5]

        # MultiMAESSTSPChoose only ever reads point_cloud_range[0]/[3] (paired with
        # voxel_size[0]) as the "x"/column extent and [1]/[4] (paired with voxel_size[1])
        # as the "y"/row extent (see window_partition's bev_shape_x/bev_shape_y). Since our
        # post-swap column 2 (row slot) holds resolved HEIGHT, the row extent here must be
        # the real z range, not y. Indices [2]/[5] and voxel_size[2] are never read.
        pc_range_sst = [x_min, z_min, 0.0, x_max, z_max, 1.0]
        voxel_size_sst = (vx, vz, 1.0)

        args.update(
            window_shape=window_shape,
            voxel_size=voxel_size_sst,
            shifts_list=[(0, 0), (window_shape[0] // 2, window_shape[1] // 2)],
            drop_info=(_drop_info(args["drop_info"]["training"]),
                       _drop_info(args["drop_info"]["test"])),
            point_cloud_range=pc_range_sst,
            # [n_height_bins, n_x_bins]
            output_shape=[int(round((z_max - z_min) / vz)),
                          int(round((x_max - x_min) / vx))],
            # encoder-only, as in SSTBEV (no MAE reconstruction decoder at detection time)
            build_decoder=False,
        )
        self.sst = MultiMAESSTSPChoose(**args)
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """Write `spatial_features_fv`, the (height, x) canvas of the front-view tokens.

        Consumes the `_fv`-suffixed voxel keys the data processor writes, mean-pools
        the points of each voxel as MeanVFE does, and runs the same window attention
        as SSTBEV over the front-view grid.
        """
        voxels_fv = batch_dict["voxels_fv"]
        num_points_fv = batch_dict["voxel_num_points_fv"]
        coords_fv = batch_dict["voxel_coords_fv"].long()
        bs = batch_dict["batch_size"]

        # mean-pool over points-in-voxel, same math as MeanVFE (no separate VFE slot is
        # available at this point in the module topology for a second stream)
        points_mean = voxels_fv.sum(dim=1)
        normalizer = torch.clamp_min(num_points_fv.view(-1, 1), min=1.0).type_as(voxels_fv)
        feat = (points_mean / normalizer).contiguous()

        # the critical column swap (see module docstring)
        coords_fv = coords_fv[:, [0, 2, 1, 3]]

        feat, ind_dict_list, voxel_info = self.sst.get_voxel_info(feat, coords_fv, bs)
        tokens = self.sst.forward_encoder((feat, ind_dict_list, voxel_info))
        batch_dict["spatial_features_fv"] = self.sst.recover_bev(tokens, voxel_info["coors"], bs)
        return batch_dict
