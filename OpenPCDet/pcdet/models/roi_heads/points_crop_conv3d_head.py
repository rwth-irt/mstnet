"""Second-stage refinement of the vertical box parameters from raw points.

The first stage localizes a vessel well on the water surface and badly along the
height axis. This head takes the proposals of that stage, crops the raw points that
fall inside each one into a small grid in the proposal's own frame, and refines the
box from that crop together with the transformer feature sampled at the proposal
centre. With REFINE_Z_ONLY the refinement is restricted to z and the box height, so
the stage can only touch the axis that carries the error.
"""
import torch
import torch.nn as nn

from ..model_utils.roi_point_crop_utils import crop_and_bin_points_to_roi_grid, sample_bev_features
from .roi_head_template import RoIHeadTemplate


class PointsCropConv3DHead(RoIHeadTemplate):
    """Stage-2 refinement for MSTNet: for each RoI from CenterHead's top-K proposals,
    crop raw points into a small local grid (canonicalized to the RoI's own frame,
    see roi_point_crop_utils.crop_and_bin_points_to_roi_grid), run a plain dense
    Conv3d stack over it (no spconv/custom CUDA -- deliberately, since a small fixed
    crop restricted to already-likely regions doesn't need scene-scale sparsity
    machinery), and fuse with the SST/BEV feature sampled at the RoI's (x,y)
    location (global SSL-pretrained context the local crop alone lacks). Reuses
    RoIHeadTemplate's target assignment / loss / box decode unchanged, following
    VoxelRCNNHead's structural pattern (only the pooling step differs).
    """

    def __init__(self, model_cfg, input_channels, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.grid_size = model_cfg.GRID_SIZE          # [gx, gy, gz]
        self.enlarge_ratio = model_cfg.ENLARGE_RATIO
        self.pool_size = model_cfg.POOL_SIZE

        # The crop reads RAW points from batch_dict['points'], whose width is fixed by the
        # dataset (x,y,z,intensity), NOT by the VFE. Sizing this conv from
        # model_info_dict['num_point_features'] is therefore only correct while the VFE
        # happens to leave that width untouched (MeanVFE). A widening VFE such as
        # ZBinnedMeanVFE reports 4+K and the first Conv3d is then built for 4+K+1 input
        # channels while the crop still delivers 5 -- a shape error at the first batch.
        # RAW_POINT_FEATURES pins the true width; it defaults to input_channels so that
        # every existing config keeps its current behaviour byte for byte.
        # REFINE_Z_ONLY restricts the second stage to the vertical box parameters, i.e.
        # centre height z and box height dz. The measurement that motivates it is that
        # adding this head to MSTNet-SSL improved 3D@0.5 while BEV@0.5 stayed unchanged,
        # so none of its gain came from x, y, length, width or yaw. On top of the z-aware
        # encoder the same head then cost several BEV@0.5 points, so its horizontal
        # refinement is not merely useless but harmful once the first stage is strong. Freezing the horizontal residuals to zero makes the
        # decoded box keep the proposal's x, y, dx, dy and ry exactly, because the
        # residual encoding is relative and a zero residual decodes to the identity.
        # Default False, so every existing config keeps its behaviour unchanged.
        self.refine_z_only = model_cfg.get('REFINE_Z_ONLY', False)

        raw_point_c = model_cfg.get('RAW_POINT_FEATURES', input_channels)
        extra_c = raw_point_c - 3                      # e.g. 1 (intensity)
        conv_in_ch = 1 + 3 + extra_c                   # count + mean-relative-xyz + extra

        conv_layers = []
        c_prev = conv_in_ch
        for c in model_cfg.CONV_CHANNELS:
            n_groups = min(8, c)
            conv_layers += [
                nn.Conv3d(c_prev, c, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(n_groups, c),
                nn.ReLU(inplace=True),
            ]
            c_prev = c
        conv_layers.append(nn.AdaptiveAvgPool3d(self.pool_size))
        self.conv3d_layers = nn.Sequential(*conv_layers)
        conv_out_flat = c_prev * (self.pool_size ** 3)

        self.bev_in_channels = model_cfg.BEV_IN_CHANNELS
        pre_channel = conv_out_flat + self.bev_in_channels

        shared_fc_list = []
        for k in range(len(model_cfg.SHARED_FC)):
            shared_fc_list += [
                nn.Linear(pre_channel, model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True),
            ]
            pre_channel = model_cfg.SHARED_FC[k]
            if k != len(model_cfg.SHARED_FC) - 1 and model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        c_pre = pre_channel
        for k in range(len(model_cfg.CLS_FC)):
            cls_fc_list += [
                nn.Linear(c_pre, model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(model_cfg.CLS_FC[k]),
                nn.ReLU(),
            ]
            c_pre = model_cfg.CLS_FC[k]
            if k != len(model_cfg.CLS_FC) - 1 and model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(c_pre, self.num_class, bias=True)

        reg_fc_list = []
        c_pre = pre_channel
        for k in range(len(model_cfg.REG_FC)):
            reg_fc_list += [
                nn.Linear(c_pre, model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(model_cfg.REG_FC[k]),
                nn.ReLU(),
            ]
            c_pre = model_cfg.REG_FC[k]
            if k != len(model_cfg.REG_FC) - 1 and model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(c_pre, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        """Xavier-initialize the trunk, and the two prediction layers narrowly.

        The classification and regression layers start near zero so the first
        refinement steps are small corrections to the proposals rather than jumps
        away from them, which is what upstream's RoI heads do as well.
        """
        init_func = nn.init.xavier_normal_
        for module_list in [self.conv3d_layers, self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, (nn.Linear, nn.Conv3d)):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def roi_crop_pool(self, batch_dict):
        """Gather the per-proposal point grid and the bird's-eye feature at its centre.

        Returns a pair. The first is (B*M, C_in, gz, gy, gx), the binned raw points of
        every proposal in the frame of that proposal. The second is (B*M, C_bev), the
        transformer feature bilinearly sampled at the proposal's (x, y). The loop runs
        per frame because the point cloud arrives concatenated with a batch column.
        """
        rois = batch_dict['rois']                        # (B, M, 7+C)
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']                     # (sum_Np, 1 + C)
        spatial_features_2d = batch_dict['spatial_features_2d']  # (B, C, ny, nx)

        grid_feats_list, bev_feats_list = [], []
        for b in range(batch_size):
            pts_b = points[points[:, 0] == b][:, 1:]
            rois_b = rois[b]
            grid_b = crop_and_bin_points_to_roi_grid(pts_b, rois_b, self.grid_size, self.enlarge_ratio)
            grid_feats_list.append(grid_b)
            bev_b = sample_bev_features(spatial_features_2d[b], rois_b[:, 0:2], self.point_cloud_range)
            bev_feats_list.append(bev_b)

        grid_feats = torch.cat(grid_feats_list, dim=0)     # (B*M, C_in, gz, gy, gx)
        bev_feats = torch.cat(bev_feats_list, dim=0)       # (B*M, C_bev)
        return grid_feats, bev_feats

    def forward(self, batch_dict):
        """Refine the proposals of the first stage and write the second-stage outputs.

        Training also assigns targets, so `rois` and `roi_labels` are replaced by the
        sampled set before the crop. Everything after the crop, the target assignment,
        the losses and the box decoding, is RoIHeadTemplate's and unmodified.
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        grid_feats, bev_feats = self.roi_crop_pool(batch_dict)
        conv_out = self.conv3d_layers(grid_feats)
        conv_out = conv_out.view(conv_out.size(0), -1)
        pooled_features = torch.cat([conv_out, bev_feats], dim=-1)

        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if self.refine_z_only:
            # Box order is (x, y, z, dx, dy, dz, ry); keep 2 and 5, freeze the rest.
            # Multiplying by a constant mask also zeroes the gradient reaching those
            # output units, so training and inference stay consistent and the frozen
            # dimensions are never learned.
            #
            # The frozen dimensions still enter rcnn_loss_reg, as smooth L1 between a
            # hard zero and a nonzero target. That term is gradient-free but it is NOT
            # constant, it varies with the data and is typically the same order as the
            # live z and dz terms. The reported regression loss of a run using this
            # switch is therefore inflated and must not be compared against a run
            # without it. Only the metrics are comparable.
            assert rcnn_reg.shape[-1] == self.box_coder.code_size, (
                'REFINE_Z_ONLY indexes a single 7-dim box code. With more than one '
                'class the prediction is code_size * num_class wide and this mask '
                'would freeze every dimension of every class after the first. Use '
                'CLASS_AGNOSTIC: True, which is what every current config does.')
            keep = rcnn_reg.new_zeros(rcnn_reg.shape[-1])
            keep[2] = 1.0
            keep[5] = 1.0
            rcnn_reg = rcnn_reg * keep

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict
