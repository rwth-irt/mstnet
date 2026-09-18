"""z-aware variant of MultiSubVoxelDynamicVoxelNetSSL: widens the mean-pillar input
token with K sub-pillar height-bin means (DynamicZBinnedVFE), everything else
(reconstruction targets, masking, backbone, losses) is untouched -- minimal-diff
subclass so the well-tested parent class behaviour is preserved exactly except at
the one line where voxel_encoder is called.

Motivation: MSTNet-SSL's oracle-corrected z-loss is still well above
VoxelNeXt's; the earlier z-aware-tokenization test (in_channel widened,
only `linear0` re-initialised on TOP of the existing checkpoint) regressed
badly. This variant tests the previously-untried configuration: a
FULL from-scratch GeoMAE pretraining run with the widened input from the start, so
`linear0` is never mismatched with the rest of the pretrained encoder.
"""
from mmdet.models import DETECTORS
from mmdet3d.ops import Voxelization

from .multi_sub_voxel_dynamic_voxelnet_ssl import MultiSubVoxelDynamicVoxelNetSSL


@DETECTORS.register_module()
class MultiSubVoxelDynamicVoxelNetSSLZAware(MultiSubVoxelDynamicVoxelNetSSL):
    """The masked pretraining detector with a height-aware token.

    Identical to the pretraining detector it derives from except that the encoder
    sees the vertical profile of each pillar. A second, finer voxelization of the
    same points supplies the height bins; the pretext task, the mask ratio and the
    six loss terms are unchanged, so a checkpoint from here differs from an ordinary
    one only in the width of the first linear layer.

    Argument z_bin_size is the height of one bin, in metres.
    """

    def __init__(self, z_bin_size, **kwargs):
        super().__init__(**kwargs)
        # same x,y resolution and range as the top-level voxel_layer, finer z
        base_vx = kwargs["voxel_layer"]["voxel_size"][0]
        base_vy = kwargs["voxel_layer"]["voxel_size"][1]
        self.fine_voxel_layer = Voxelization(
            voxel_size=(base_vx, base_vy, z_bin_size),
            point_cloud_range=self.point_cloud_range,
            max_num_points=-1,
            max_voxels=(-1, -1),
        )

    def extract_feat(self, points, gt_bboxes_3d, gt_labels_3d, img_metas, vis=False):
        """Voxelize once more at the fine height resolution, then defer to the parent.

        The fine coordinates reach the encoder through an attribute rather than
        through the argument list, because the parent's implementation calls the
        encoder with two arguments and threading a third through would mean copying
        its seventy lines here.
        """
        # Rather than duplicating extract_feat's ~70 lines to thread a second
        # (fine_coors) argument through, stash it as a plain (non-submodule) tensor
        # attribute on self.voxel_encoder, which its forward() picks up and clears
        # immediately -- see DynamicZBinnedVFE.forward. NEVER reassign
        # self.voxel_encoder itself here: nn.Module.__setattr__ rejects assigning a
        # non-nn.Module value to a name already registered as a submodule.
        import torch
        import torch.nn.functional as F
        fine_coors = [self.fine_voxel_layer(res) for res in points]
        fine_coors_batch = torch.cat(
            [F.pad(c, (1, 0), mode="constant", value=i) for i, c in enumerate(fine_coors)], dim=0)
        self.voxel_encoder._pending_fine_coors = fine_coors_batch
        return super().extract_feat(points, gt_bboxes_3d, gt_labels_3d, img_metas, vis=vis)
