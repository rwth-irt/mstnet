"""GeoMAE SSL pre-training on the Thames recordings with a LEARNED pillar encoder.

Variant of ssl_london.py that replaces the mean voxel encoder (DynamicSimpleVFE) with
PointPillars' learned DynamicPillarFeatureNet. It is the Thames counterpart of
ssl_seepersea_pfn10.py and supplies the pretrained encoder for the mstnet_c rows of
experiment E.II on this dataset; without it that experiment would be measured on
SeePerSea alone. Only the voxel encoder and the backbone's input width change. The
reconstruction scheme, the schedule and the 300 epochs are those of ssl_london.py, so
the comparison against the mean-encoder arm differs in the encoder and in nothing else.

Added 2026-08-17, replacing ssl_london_pfn.py. That predecessor built its feature net
from nine per-point inputs and a single 64-wide layer, which matches no module on the
detector side, so its weights could not be loaded into mstnet_c and every learned-encoder
finetuning ran with a randomly initialised feature net under a pretrained transformer.
The two changes here, voxel_center_z and feat_channels, give shape parity with
OpenPCDet's DynPillarVFE, which is what convert_geomae_pfn_ckpt.py checks for.
"""
_base_ = [
    "../_base_/datasets/nus-3d-ssl.py",
    "../_base_/schedules/cosine_2x.py",
    "../_base_/default_runtime.py",
]
class_names = []

custom_hooks = [
    dict(type="CheckInvalidLossHook", interval=1),
]

dataset_type = "NuScenesDatasetSSL"
# Resolved relative to the GeoMAE directory the training runs from;
# GeoMAE/data is a symlink to the repository data/ tree.
data_root = "data/ssl_nuscenes/london/data/nuscenes/"

# --- Voxel / range aligned with supervised training ---
voxel_size = (0.5, 0.5, 55)

# learned pillar VFE output dim -> SST in_channel
vfe_feat_channels = (32, 64)   # -> Linear(10,32) + Linear(64,64) = OpenPCDet DynPillarVFE parity
sst_in_channel = vfe_feat_channels[-1]
window_shape = (12, 12)
point_cloud_range = [-100, -100, -20, 100, 100, 35]

sub_voxel_size_low = (0.125, 0.125, 6.875)   # voxel / (4,4,8)
sub_voxel_size_med = (0.25, 0.25, 13.75)      # voxel / (2,2,4)
sub_voxel_size_top = (0.5, 0.5, 55)
sub_voxel_ratio_low = (8, 4, 4)   # z,y,x
sub_voxel_ratio_med = (4, 2, 2)   # z,y,x
input_voxel_ratio = (1, 2, 2)
grid_size = (1, 400, 400)         # z,y,x  (200m / 0.5m = 400)
random_mask_ratio = 0.7
loss_ratio_low = 10.0
loss_ratio_med = 8.0
loss_ratio_top = 10.0
cls_loss_ratio_low = 5.0
cls_loss_ratio_med = 2.0
cls_sub_voxel = True

loss_ratio_low_nor = 4.0
loss_ratio_med_nor = 0
loss_ratio_top_nor = 0

drop_info_training = {
    0: {"max_tokens": 56, "drop_range": (0, 56)},
    1: {"max_tokens": 144, "drop_range": (56, 100000)},
}
drop_info_test = {
    0: {"max_tokens": 32, "drop_range": (0, 32)},
    1: {"max_tokens": 72, "drop_range": (32, 72)},
    2: {"max_tokens": 144, "drop_range": (72, 100000)},
}

input_modality = dict(
    use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False
)

drop_info = (drop_info_training, drop_info_test)
shifts_list = [(0, 0), (window_shape[0] // 2, window_shape[1] // 2)]

model = dict(
    type="MultiSubVoxelDynamicVoxelNetSSL",
    normalize_sub_voxel=True,
    mse_loss=True,
    loss=dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
    spatial_shape=[1, 400, 400],
    loss_ratio_low_nor=loss_ratio_low_nor,
    loss_ratio_med_nor=loss_ratio_med_nor,
    loss_ratio_top_nor=loss_ratio_top_nor,
    loss_ratio_low=loss_ratio_low,
    loss_ratio_med=loss_ratio_med,
    loss_ratio_top=loss_ratio_top,
    cls_sub_voxel=cls_sub_voxel,
    cls_loss_ratio_low=cls_loss_ratio_low,
    cls_loss_ratio_med=cls_loss_ratio_med,
    random_mask_ratio=random_mask_ratio,
    grid_size=grid_size,
    sub_voxel_ratio_low=sub_voxel_ratio_low,
    sub_voxel_ratio_med=sub_voxel_ratio_med,
    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1),
    ),
    sub_voxel_layer_low=dict(
        voxel_size=sub_voxel_size_low,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1),
    ),
    sub_voxel_layer_med=dict(
        voxel_size=sub_voxel_size_med,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1),
    ),
    hard_sub_voxel_layer_low=dict(
        voxel_size=sub_voxel_size_low,
        max_num_points=30,
        point_cloud_range=point_cloud_range,
        max_voxels=(140000, 140000),
    ),
    hard_sub_voxel_layer_med=dict(
        voxel_size=sub_voxel_size_med,
        max_num_points=50,
        point_cloud_range=point_cloud_range,
        max_voxels=(80000, 80000),
    ),
    hard_sub_voxel_layer_top=dict(
        voxel_size=voxel_size,
        max_num_points=100,
        point_cloud_range=point_cloud_range,
        max_voxels=(40000, 40000),
    ),
    # LEARNED pillar VFE (PointPillars-style) instead of the mean DynamicSimpleVFE
    voxel_encoder=dict(
        type="DynamicPillarFeatureNet",
        in_channels=4,
        feat_channels=vfe_feat_channels,
        with_distance=False,
        # voxel_center_z adds the vertical pillar-centre offset -> 10 input features,
        # matching OpenPCDet's DynPillarVFE so the pretrained VFE transfers 1:1.
        voxel_center_z=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    backbone=dict(
        type="MultiMAESSTSPChoose",
        cls_sub_voxel=cls_sub_voxel,
        window_shape=window_shape,
        shifts_list=shifts_list,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        shuffle_voxels=False,
        low=False,
        med=False,
        top=True,
        d_model=[128] * 6,
        nhead=[8] * 6,
        sub_voxel_ratio_low=sub_voxel_ratio_low,
        sub_voxel_ratio_med=sub_voxel_ratio_med,
        encoder_num_blocks=6,
        decoder_num_blocks=2,
        dim_feedforward=[256] * 6,
        output_shape=[400, 400],
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        in_channel=sst_in_channel,
    ),
)

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4,
         file_client_args=file_client_args),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points"]),
]

data = dict(
    # Raised on 2026-08-07. The pretraining is launch-bound rather than compute
    # bound: data loading is 3.6 percent of an iteration, yet the GPU sits at 30
    # percent. A larger batch puts more work behind each kernel launch. The learning
    # rate is left as it was, because the cyclic schedule already peaks at a hundred
    # times the base value and raising it as well risks divergence. The epoch counts
    # are unchanged, so each encoder still sees the same number of samples; only the
    # number of updates falls.
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_ssl_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr = 1e-5
runner = dict(type="EpochBasedRunner", max_epochs=300)
evaluation = dict(interval=-1)
