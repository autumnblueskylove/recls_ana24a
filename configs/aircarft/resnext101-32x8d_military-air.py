_base_ = [
    '../_base_/datasets/aircraft/military-aircraft.py',
    '../_base_/models/resnext/resnext101_32x8d.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(head=dict(num_classes=24))
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth'  # noqa 501

# runtime
default_hooks = dict(checkpoint=dict(interval=20), )

train_cfg = dict(val_interval=10)

# mlflow
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='MLflowVisBackend'),
    ])
