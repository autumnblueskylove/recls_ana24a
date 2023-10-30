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
        embed_dims=4,
        feedforward_channels=128,
    ),
    head=dict(
        num_classes=27,
        in_channels=768 + 4,
    ))

load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth'  # noqa

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
