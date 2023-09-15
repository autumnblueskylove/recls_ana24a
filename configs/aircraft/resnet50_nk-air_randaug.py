_base_ = [
    '../_base_/datasets/aircraft/nk-aircraft-v2_224px_randaug.py',
    '../_base_/models/resnet/resnet50.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(head=dict(num_classes=23))
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', optimizer=dict(lr=0.01))

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
