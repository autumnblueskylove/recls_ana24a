_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/aircraft/nk-aircraft-v2_224px_randaug.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', clip_grad=dict(max_norm=5.0))

model = dict(head=dict(num_classes=23))
load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth'  # noqa

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

# mlflow
visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
