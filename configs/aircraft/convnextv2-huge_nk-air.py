_base_ = [
    '../_base_/datasets/aircraft/nk-aircraft_384px.py',
    '../_base_/models/convnext_v2/huge.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(head=dict(num_classes=24))

load_from = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-02a4eb35.pth'  # noqa

# train & validation
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=16)
test_dataloader = dict(batch_size=16)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-3),
    clip_grad=None,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]
auto_scale_lr = dict(enable=True)

# runtime
default_hooks = dict(
    checkpoint=dict(
        interval=-1,
        save_best='auto',
        save_optimizer=False,
        save_param_scheduler=False), )

# train
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# mlflow
visualizer = dict(
    type='mmpretrain.UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
