"""ZBinnedMeanVFE: MeanVFE, widened with K fine sub-pillar z-bin means -- the
finetuning-side counterpart of GeoMAE's DynamicZBinnedVFE
(Tsinghua-MARS-Lab/GeoMAE/mmdet3d/models/voxel_encoders/dynamic_zbinned_vfe.py).
Reads the standard 'voxels'/'voxel_num_points'/'voxel_coords' (defines the canonical
(batch,y,x) token set, unchanged) plus the '_zfine'-suffixed keys written by
transform_points_to_zbinned_voxels, and concatenates fine z-bin means onto each
canonical token via a dense scatter+gather by (batch,y,x) -- no index join, avoiding
the coordinate-matching bug class already hit once this project (FrontViewSSTBranch's
column-swap fix).

Output per token: [x, y, z, intensity, zbin_1, ..., zbin_K].
"""
import torch

from .vfe_template import VFETemplate


class ZBinnedMeanVFE(VFETemplate):
    """MeanVFE widened by the mean height of each of K sub-pillar bins.

    The mean encoder averages every point of a pillar into one token, which leaves the
    token blind to how the points are distributed over height. This variant keeps that
    token and appends K numbers to it, the mean z of the points in each of K equal
    slices of the pillar, so the vertical profile survives tokenization.

    Config key NUM_Z_BINS is K. The output token is
    [x, y, z, intensity, zbin_1, ..., zbin_K], and an empty bin contributes zero.
    """

    def __init__(self, model_cfg, num_point_features, grid_size, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.K = model_cfg.NUM_Z_BINS
        # grid_size from the dataset is [nx, ny, nz] in x,y,z order (OpenPCDet
        # convention); the canonical BEV grid is nx,ny (z already collapsed to 1).
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])

    def get_output_feature_dim(self):
        """Point features plus one entry per height bin."""
        return self.num_point_features + self.K

    def forward(self, batch_dict, **kwargs):
        """Write `voxel_features`, the mean token of each pillar with its height profile.

        The canonical token set stays exactly what MeanVFE would produce, so the
        detector downstream is unchanged. The fine bins arrive as a second voxelization
        under the `_zfine` keys and are joined to the canonical tokens through a dense
        scatter and gather on (batch, y, x) rather than through an index join, which is
        the operation an earlier coordinate-matching bug in this project came from.
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer  # [M, C]

        coors = batch_dict['voxel_coords']  # [M,4] = (batch, z=0, y, x)

        fine_voxels, fine_num_points = batch_dict['voxels_zfine'], batch_dict['voxel_num_points_zfine']
        fine_points_mean = fine_voxels[:, :, :].sum(dim=1, keepdim=False)
        fine_normalizer = torch.clamp_min(fine_num_points.view(-1, 1), min=1.0).type_as(fine_voxels)
        fine_points_mean = fine_points_mean / fine_normalizer  # [Mf, C]
        fine_coors = batch_dict['voxel_coords_zfine']  # [Mf,4] = (batch, z_idx, y, x)

        B = int(coors[:, 0].max().item()) + 1 if coors.numel() else 1
        dense = torch.zeros(B, self.K, self.ny, self.nx, device=points_mean.device, dtype=points_mean.dtype)
        b = fine_coors[:, 0].long()
        zk = fine_coors[:, 1].long().clamp(0, self.K - 1)
        y = fine_coors[:, 2].long().clamp(0, self.ny - 1)
        x = fine_coors[:, 3].long().clamp(0, self.nx - 1)
        dense[b, zk, y, x] = fine_points_mean[:, 2]  # store the z-COORDINATE mean per bin

        bb = coors[:, 0].long()
        by = coors[:, 2].long().clamp(0, self.ny - 1)
        bx = coors[:, 3].long().clamp(0, self.nx - 1)
        zbins = dense[bb, :, by, bx]  # [M, K]

        batch_dict['voxel_features'] = torch.cat([points_mean, zbins], dim=1).contiguous()
        return batch_dict
