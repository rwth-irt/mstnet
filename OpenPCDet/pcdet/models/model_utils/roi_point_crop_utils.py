"""Pure-torch (no spconv/custom CUDA) helpers for the PointsCropConv3DHead stage-2
refinement module: crop raw points into a per-RoI local grid, and sample the dense
BEV feature map at each RoI's (x,y) location. Deliberately dense/scatter-based
(index_add_ into a flat destination, no join/sort) -- same pattern already verified
in ZBinnedMeanVFE/DynamicZBinnedVFE, chosen to avoid this project's repeated
coordinate-indexing bug class.
"""
import torch
import torch.nn.functional as F

from ...utils import common_utils


def crop_and_bin_points_to_roi_grid(points, rois, grid_size, enlarge_ratio):
    """Canonicalize points into each RoI's local (translated + rotated to RoI
    center/heading) frame -- same convention as RoIHeadTemplate.assign_targets --
    then mean-bin them into a fixed-size dense grid per RoI.

    Args:
        points: (Np, 3 + C) xyz + extra features (e.g. intensity), single sample
        rois: (M, 7 + C) x,y,z,dx,dy,dz,heading[,...], single sample
        grid_size: (gx, gy, gz) ints
        enlarge_ratio: float, RoI box is scaled by this before binning (context margin)

    Returns:
        grid: (M, 1 + 3 + C, gz, gy, gx) -- channels: [count, mean_dx, mean_dy, mean_dz, mean_extra...]
    """
    gx, gy, gz = grid_size
    M = rois.shape[0]
    Np = points.shape[0]
    device = points.device
    dtype = points.dtype
    extra_c = points.shape[-1] - 3
    out_c = 1 + 3 + extra_c

    if M == 0 or Np == 0:
        return torch.zeros(M, out_c, gz, gy, gx, device=device, dtype=dtype)

    roi_center = rois[:, 0:3]
    roi_size = rois[:, 3:6] * enlarge_ratio
    roi_heading = rois[:, 6]

    pts_xyz = points[:, 0:3]
    pts_extra = points[:, 3:]

    pts_rel = pts_xyz.unsqueeze(0) - roi_center.unsqueeze(1)          # (M, Np, 3)
    pts_rel = common_utils.rotate_points_along_z(pts_rel, -roi_heading)  # (M, Np, 3)

    half = torch.clamp_min(roi_size, min=1e-3).unsqueeze(1) / 2.0     # (M, 1, 3)
    mask = (pts_rel.abs() <= half).all(dim=-1)                        # (M, Np)

    m_idx, p_idx = mask.nonzero(as_tuple=True)
    if m_idx.numel() == 0:
        return torch.zeros(M, out_c, gz, gy, gx, device=device, dtype=dtype)

    grid_size_t = pts_rel.new_tensor([gx, gy, gz])
    cell_size = (2.0 * half) / grid_size_t                             # (M, 1, 3)
    idx = torch.floor((pts_rel + half) / cell_size).long()             # (M, Np, 3)
    idx[..., 0].clamp_(0, gx - 1)
    idx[..., 1].clamp_(0, gy - 1)
    idx[..., 2].clamp_(0, gz - 1)

    sel_idx = idx[m_idx, p_idx]                                        # (K, 3) x,y,z cell idx
    flat_cell = sel_idx[:, 2] * gy * gx + sel_idx[:, 1] * gx + sel_idx[:, 0]
    dest = m_idx * (gx * gy * gz) + flat_cell                          # (K,)

    if extra_c > 0:
        feat = torch.cat([pts_rel[m_idx, p_idx], pts_extra[p_idx]], dim=-1)  # (K, 3+extra_c)
    else:
        feat = pts_rel[m_idx, p_idx]

    n_cells = M * gx * gy * gz
    sums = torch.zeros(n_cells, 3 + extra_c, device=device, dtype=dtype)
    sums.index_add_(0, dest, feat)
    counts = torch.zeros(n_cells, device=device, dtype=dtype)
    counts.index_add_(0, dest, torch.ones(dest.shape[0], device=device, dtype=dtype))

    means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    grid = torch.cat([counts.unsqueeze(-1), means], dim=-1)            # (n_cells, out_c)
    grid = grid.view(M, gz, gy, gx, out_c).permute(0, 4, 1, 2, 3).contiguous()
    return grid


def sample_bev_features(spatial_features_2d, roi_xy, point_cloud_range):
    """Bilinearly sample the dense BEV feature map at each RoI's (x,y) center.
    Grid_sample's native (x,y)-normalized-coordinate convention matches the BEV
    map's own (x,y) axes directly -- no coordinate-column swap needed here (unlike
    the spconv voxel-index case in FrontViewSSTBranch, which is a different,
    coords-order convention entirely).

    Args:
        spatial_features_2d: (C, ny, nx) single sample
        roi_xy: (M, 2) world x,y
        point_cloud_range: length-6 [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        (M, C)
    """
    C = spatial_features_2d.shape[0]
    xmin, ymin, xmax, ymax = point_cloud_range[0], point_cloud_range[1], point_cloud_range[3], point_cloud_range[4]
    norm_x = 2.0 * (roi_xy[:, 0] - xmin) / (xmax - xmin) - 1.0
    norm_y = 2.0 * (roi_xy[:, 1] - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)     # (1, M, 1, 2)
    feat = F.grid_sample(spatial_features_2d.unsqueeze(0), grid, mode='bilinear',
                          padding_mode='zeros', align_corners=False)   # (1, C, M, 1)
    return feat.view(C, -1).t()                                       # (M, C)
