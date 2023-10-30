_base_ = [
    '../_base_/datasets/aircraft/foreign-aircraft_224px_w-meta_randaug.py',
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    type='GeoImageLateClassifier',
    use_gsd_in_backbone=True,
    backbone=dict(
        type='ScaleSwinTransformer',
        use_abs_pos_embed=True,
    ),
    meta_encoder=dict(
        type='FFN',
        embed_dims=8,
        feedforward_channels=128,
    ),
    head=dict(
        num_classes=27,
        in_channels=768 + 8,
    ))

load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth'  # noqa

# Optimizer
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

expand_ratio = 1.2
bgr_mean = _base_.data_preprocessor['mean'][::-1]
bgr_std = _base_.data_preprocessor['std'][::-1]
train_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
        include_gsd=True,
        use_object_size=True,
        use_xgsd=True,
    ),
    dict(
        type='JitterRBox',
        x_range=0.1,
        y_range=0.1,
        w_range=(0.9, 1.1),
        h_range=(0.9, 1.1),
        rad_range=(-0.174, 0.174)),
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 1.0),
        max_percentile_range=(99.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(
        type='mmpretrain.RandAugment',
        policies=_base_.rand_increasing_policies,
        # policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='mmpretrain.RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos', 'xy_gsd']),
]

test_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
        include_gsd=True,
        use_object_size=True,
        use_xgsd=True,
    ),
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos', 'xy_gsd']),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        include_date=False,
        include_gsd=True,
    ))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        include_date=False,
        include_gsd=True,
    ))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        include_date=False,
        include_gsd=True,
    ))

# runtime
default_hooks = dict(
    checkpoint=dict(
        interval=-1,
        save_best='single-label/recall',
        rule='greater',
        save_optimizer=False,
        save_param_scheduler=False), )

# train
train_cfg = dict(max_epochs=100, val_interval=5)
auto_scale_lr = dict(enable=True)

# mlflow
visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
