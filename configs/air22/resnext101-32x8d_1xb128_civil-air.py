_base_ = [
    '../_base_/datasets/collection_civil-aircraft_bs128.py',
    '../_base_/models/resnext101_32x8d.py',
    '../_base_/schedules/imagenet_bs256_140e.py',
    '../_base_/default_runtime.py',
]

model = dict(head=dict(num_classes=15))

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnext/' \
            'resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth'
fp16 = dict(loss_scale='dynamic')

# runtime
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MlflowLoggerHook',
            run_id=None,
            exp_name='Civil-Aircraft_CLS',
            run_name='resnext101-32x8d_1xb256_civil-air')
    ])
