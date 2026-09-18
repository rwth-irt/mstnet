"""Stride-1 dilated conv stack — the reference SST/GeoMAE head-adapter.

The SST exists to keep BEV resolution ("single stride"), so the reference
implementation attaches only a few stride-1 convs after it (GeoMAE's SSTOutConv:
3x conv, kernel 3, dilation 1/1/2 — dilation widens the receptive field without
ever downsampling). src MSTNet instead runs a full downsampling BaseBEVBackbone
there, which throws away exactly the resolution the SST was built to preserve.

Use this as BACKBONE_2D to get the reference behaviour; use BaseBEVBackbone to
reproduce src MSTNet. Sparse maritime scenes (~568 pts/frame, small distant boats)
are where the difference should show up.
"""
import torch.nn as nn


class SSTOutConv(nn.Module):
    """Stride-one convolutions over the transformer's bird's-eye map.

    Three 3x3 convolutions at dilations 1, 1 and 2 by default, so the receptive field
    grows without the resolution falling. Config keys NUM_FILTERS and DILATIONS must
    have the same length; each pair is one convolution with batch normalization and a
    rectifier.
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        num_filters = model_cfg.get("NUM_FILTERS", [128, 128, 128])
        dilations = model_cfg.get("DILATIONS", [1, 1, 2])
        assert len(num_filters) == len(dilations), "NUM_FILTERS and DILATIONS must align"

        layers, c_in = [], input_channels
        for c_out, d in zip(num_filters, dilations):
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.blocks = nn.Sequential(*layers)
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """Write `spatial_features_2d` from `spatial_features`, at the same resolution."""
        data_dict["spatial_features_2d"] = self.blocks(data_dict["spatial_features"])
        return data_dict
