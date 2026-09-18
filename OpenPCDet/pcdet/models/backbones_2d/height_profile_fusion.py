"""Fuses the front-view token stream into the existing BEV pipeline (MSTNet multi-view
tokens, idea 3, Phase 1). Deliberately minimal: pool the front-view canvas over height to
get a per-x-column descriptor, reduce it, and broadcast it across the BEV canvas's
y-dimension as extra channels. This gives BaseBEVBackbone/CenterHead a genuine,
spatially-varying (per x-column) vertical signal -- unlike the abandoned water-level head,
which used one pooled scalar for the whole frame and was too imprecise (MAE 1.37m against
~1.8m box heights) to help 3D IoU@0.5. See project plan for the full rationale.

Uses GroupNorm, not BatchNorm, from the start: the water-level head's second failure was a
BatchNorm train/eval statistics mismatch in a similarly globally-pooled path.
"""
import torch
import torch.nn as nn


class HeightProfileFusion(nn.Module):
    """Appends a per-column height descriptor to the bird's-eye feature map.

    The front-view branch produces a canvas over (height, x). Pooling it over height
    leaves one descriptor per x column, which two 1x1 convolutions reduce and which is
    then repeated along y and concatenated to the bird's-eye channels. The bird's-eye
    backbone and the head therefore see a vertical signal that varies with the column
    instead of the single pooled scalar the earlier water-level head supplied.

    Config keys: FV_IN_CHANNELS (channels of the front-view canvas) and
    PROFILE_CHANNELS (width of the descriptor, 16 by default).
    """

    def __init__(self, model_cfg, num_bev_features, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        c_fv = model_cfg.FV_IN_CHANNELS
        c_out = model_cfg.get('PROFILE_CHANNELS', 16)
        n_groups = min(4, c_out)

        self.reduce = nn.Sequential(
            nn.Conv1d(c_fv, c_out, kernel_size=1),
            nn.GroupNorm(n_groups, c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=1),
            nn.GroupNorm(n_groups, c_out), nn.ReLU(),
        )
        self.num_bev_features = num_bev_features + c_out

    def forward(self, batch_dict):
        """Widen `spatial_features` by PROFILE_CHANNELS and update `num_bev_features`.

        Reads `spatial_features` [B, C, ny, nx] and `spatial_features_fv`
        [B, C_fv, n_height_bins, nx]. Both canvases span the same x range, which the
        assertion below enforces, because the descriptor is broadcast by column.
        """
        bev = batch_dict['spatial_features']       # [B, C_bev, ny, nx]
        fv = batch_dict['spatial_features_fv']      # [B, C_fv, n_height_bins, nx_fv]

        assert fv.shape[-1] == bev.shape[-1], (
            f"front-view x-resolution/range must match BEV's (both use the same real "
            f"point_cloud_range x-span): got fv nx={fv.shape[-1]} vs bev nx={bev.shape[-1]}"
        )

        profile = fv.mean(dim=2)          # pool over height -> [B, C_fv, nx]
        profile = self.reduce(profile)    # -> [B, C_out, nx]
        profile_bc = profile.unsqueeze(2).expand(-1, -1, bev.shape[2], -1)  # -> [B,C_out,ny,nx]

        batch_dict['spatial_features'] = torch.cat([bev, profile_bc], dim=1)
        return batch_dict
