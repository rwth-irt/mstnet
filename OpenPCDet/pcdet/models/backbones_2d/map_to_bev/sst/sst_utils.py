import torch


def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]

    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype

    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    assert (check_feat == 0).all()

    return all_flat_feat


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info):
    """
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].

    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    """
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]["max_tokens"]
        num_windows = (this_inds // max_tokens).max().item() + 1
        feat_3d = torch.zeros(
            (num_windows * max_tokens, feat_dim), dtype=dtype, device=device
        )

        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict
