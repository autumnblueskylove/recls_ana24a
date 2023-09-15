_base_ = [
    '../_base_/datasets/aircraft/nk-aircraft_256px.py',
    '../_base_/models/swin_transformer_v2/base_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        window_size=[16, 16, 16, 8],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6]),
    head=dict(num_classes=24))

load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth'  # noqa 501

# Optimizer
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# runtime
default_hooks = dict(
    checkpoint=dict(
        interval=-1,
        save_best='auto',
        save_optimizer=False,
        save_param_scheduler=False), )

# train
train_cfg = dict(val_interval=5)

# mlflow
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
