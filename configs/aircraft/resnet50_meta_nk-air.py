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
        num_inputs=4,
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

train_dataloader = dict(dataset=dict(
    include_date=False,
    include_gsd=False,
))
val_dataloader = dict(dataset=dict(
    include_date=False,
    include_gsd=False,
))
test_dataloader = dict(dataset=dict(
    include_date=False,
    include_gsd=False,
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
