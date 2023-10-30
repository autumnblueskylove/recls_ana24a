_base_ = [
    '../_base_/datasets/aircraft/nk-aircraft_224px_w-meta_randaug.py',
    '../_base_/models/revvit/revvit-small.py',
    '../_base_/schedules/imagenet_bs1024_adamw_revvit.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    type='GSDImageClassifier',
    use_gsd_in_backbone=True,
    backbone=dict(type='ScaleRevViT', ),
    head=dict(num_classes=23))
load_from = 'https://download.openmmlab.com/mmclassification/v0/revvit/revvit-small_3rdparty_in1k_20221213-a3a34f5c.pth'  # noqa

# Optimizer
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

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
