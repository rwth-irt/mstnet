"""z-binned VFE: widens the standard mean-pillar token (x,y,z,intensity) with K
sub-pillar z-bin means, WITHOUT changing the number of tokens/pillars. Keeps the
existing DynamicSimpleVFE path untouched (used for the canonical (batch,y,x) token
set the SST windows over); adds a second, finer-z voxelization of the SAME points,
whose sub-bin means are matched back onto that canonical set via a dense lookup
grid, not an index join -- avoids the coordinate-matching bug class that has bitten
this project before (see FrontViewSSTBranch's column-swap fix in PCDet_Benchmark).

Output feature per canonical token: [x, y, z, intensity, zbin_1, ..., zbin_K]
(K missing/empty bins -> 0, distinguishable downstream only via the fact that a
real return at exactly the mean of an empty bin range is not required for the
sanity checks below; a presence mask could be added later if needed).
"""
import torch
import torch.nn as nn

from mmdet3d.ops import DynamicScatter
from .voxel_encoder import DynamicSimpleVFE
from ..builder import VOXEL_ENCODERS


@VOXEL_ENCODERS.register_module()
class DynamicZBinnedVFE(nn.Module):
    """Mean pillar token widened by the mean height of each of K sub-pillar bins.

    The pretraining-side counterpart of the detector's ZBinnedMeanVFE, and the two
    must stay in step: a checkpoint written here is loaded there, so the token layout
    [x, y, z, intensity, zbin_1, ..., zbin_K] is part of the interface between them.

    The number of tokens does not change. The canonical set stays what
    DynamicSimpleVFE produces, and a second scatter at a finer height resolution
    supplies the bins, matched back through a dense lookup grid instead of an index
    join.
    """

    def __init__(self, voxel_size, point_cloud_range, z_bin_size, grid_size):
        """
        Args:
            voxel_size: top-level (vx, vy, vz_full) -- unchanged from DynamicSimpleVFE,
                vz_full spans the whole z-range (single z-bin), defines the canonical
                (batch, y, x) token set.
            point_cloud_range: [x0,y0,z0,x1,y1,z1], SAME range used for both scatters.
            z_bin_size: height of each fine sub-pillar bin (e.g. 5.0).
            grid_size: [ny, nx] of the canonical BEV grid (must match voxel_size[:2]
                against point_cloud_range) -- needed to size the dense lookup tensor.
        """
        super().__init__()
        self.scatter_top = DynamicScatter(voxel_size, point_cloud_range, True)
        z_span = point_cloud_range[5] - point_cloud_range[2]
        self.K = max(1, int(round(z_span / z_bin_size)))
        fine_voxel_size = (voxel_size[0], voxel_size[1], z_span / self.K)
        self.scatter_fine = DynamicScatter(fine_voxel_size, point_cloud_range, True)
        self.ny, self.nx = grid_size
        self.z0 = point_cloud_range[2]
        self.z_bin_size = z_span / self.K
        self._pending_fine_coors = None

    def forward(self, features, coors):
        """Standard 2-arg VFE signature, for compatibility with code (extract_feat)
        that calls `self.voxel_encoder(voxels, coors)` and cannot be changed without
        duplicating it. The caller must set `self._pending_fine_coors` (a plain
        tensor attribute, NOT a submodule -- safe to set directly on an nn.Module
        instance) immediately before invoking forward. Cleared after use so a
        missing/stale value fails loudly rather than silently reusing old data."""
        fine_coors = self._pending_fine_coors
        if fine_coors is None:
            raise RuntimeError(
                "DynamicZBinnedVFE.forward called without _pending_fine_coors set; "
                "use forward_with_fine directly, or set the attribute first.")
        self._pending_fine_coors = None
        return self.forward_with_fine(features, coors, fine_coors)

    def forward_with_fine(self, features, coors, fine_coors):
        """Return the widened tokens and their canonical coordinates.

        Args:
            features: the points, [N, C].
            coors: their canonical voxel coordinates, one z bin over the full range.
            fine_coors: their coordinates under the finer height resolution.

        Returns:
            A pair of the token features [M, C + K] and the canonical coordinates
            [M, 4] as (batch, z, y, x), so the caller sees the same token set it
            would have seen without the widening.
        """
        feat_top, coors_top = self.scatter_top(features, coors)          # [M,4], [M,4]
        feat_fine, coors_fine = self.scatter_fine(features, fine_coors)  # [Mf,4], [Mf,4]=(b,zk,y,x)

        B = int(coors_top[:, 0].max().item()) + 1 if coors_top.numel() else 1
        device = features.device

        # Dense lookup: for each fine cell, scatter its z-value into a
        # [B, K, ny, nx] canvas. No join/sort -- direct index writes only.
        dense = torch.zeros(B, self.K, self.ny, self.nx, device=device, dtype=feat_fine.dtype)
        b = coors_fine[:, 0].long()
        zk = coors_fine[:, 1].long().clamp(0, self.K - 1)
        y = coors_fine[:, 2].long().clamp(0, self.ny - 1)
        x = coors_fine[:, 3].long().clamp(0, self.nx - 1)
        dense[b, zk, y, x] = feat_fine[:, 2]  # store the z-COORDINATE mean per bin

        # Gather at the canonical (batch,y,x) locations -> [M, K]
        by = coors_top[:, 2].long().clamp(0, self.ny - 1)
        bx = coors_top[:, 3].long().clamp(0, self.nx - 1)
        bb = coors_top[:, 0].long()
        zbins = dense[bb, :, by, bx]  # [M, K]

        out = torch.cat([feat_top, zbins], dim=1)  # [M, 4+K]
        return out, coors_top
