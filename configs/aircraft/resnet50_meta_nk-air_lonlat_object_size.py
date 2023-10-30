_base_ = [
    '../_base_/datasets/aircraft/nk-aircraft_224px_w-meta.py',
    '../_base_/models/resnet/resnet50.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', optimizer=dict(lr=0.01))

# model
model = dict(
    type='GeoImageClassifier',
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    meta_encoder=dict(
        type='PriorsNetEncoder',
        num_inputs=8,
    ),
    fuser=dict(type='DynamicMLPFuser', ),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=23,
        in_channels=2048,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa

expand_ratio = 1.2
train_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
        include_gsd=True,
        use_object_size=True,
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
        type='mmpretrain.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos']),
]
test_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
        include_gsd=True,
        use_object_size=True,
    ),
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos']),
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
train_cfg = dict(val_interval=5)

# mlflow
visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
