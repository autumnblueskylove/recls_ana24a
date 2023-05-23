dataset_type = 'DataPlatformDatasetV2'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = 'jmkoo'
dp_password = 'jmkoo@sia2022'
train_dataset_id = 169
test_dataset_id = 155
categories = [
    dict(id=0, name='AN-148'),
    dict(id=1, name='AN-2'),
    dict(id=2, name='AN-24'),
    dict(id=3, name='IL-18'),
    dict(id=4, name='IL-28'),
    dict(id=5, name='IL-62'),
    dict(id=6, name='IL-76'),
    dict(id=7, name='MI-2'),
    dict(id=8, name='MI-26'),
    dict(id=9, name='MI-4'),
    dict(id=10, name='MI-8_17'),
    dict(id=11, name='MIG-15_17'),
    dict(id=12, name='MIG-19'),
    dict(id=13, name='MIG-21'),
    dict(id=14, name='MIG-23'),
    dict(id=15, name='MIG-29'),
    dict(id=16, name='SU-25'),
    dict(id=17, name='TU-134'),
    dict(id=18, name='TU-154'),
    dict(id=19, name='TU-204'),
    dict(id=20, name='YAK-18'),
    dict(id=21, name='DECOY'),
    dict(id=22, name='light-aircraft'),
    dict(id=23, name='unknown')
]
expand_ratio = 1.2
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type='RandomRBox',
        x_range=0.1,
        y_range=0.1,
        w_range=(0.9, 1.1),
        h_range=(0.9, 1.1),
        rad_range=(-0.174, 0.174)),
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.1),
        max_percentile_range=(99.0, 100.0)),
    dict(type='Resize', size=(224, 224)),
    dict(type='GaussianNoise'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=train_dataset_id,
        categories=categories,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        categories=categories,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='Accuracy', interval=1)
test_dataloader = val_dataloader
test_evaluator = val_evaluator

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=8,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=24,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)
    )
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
)

# schedule settings
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[40, 80, 120],
    gamma=0.1,
)

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=140)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20),
    logger=dict(type='LoggerHook', interval=100),
)

# set visualizer
exp_name = 'Military-Aircraft_CLS'
run_name = 'resnext101-32x8d_1xb128_military-air'

vis_backends = [
    dict(type='MLflowVisBackend', exp_name=exp_name,
         run_name=run_name,
         artifact_suffix=('.json', '.log', '.py', 'yaml', 'model_final.pth'))
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth'
resume = True

log_level = 'INFO'
fp16 = dict(loss_scale='dynamic')
work_dir = '/nas/k8s/dev/research/jhseo/ckpts/recls_migration_test'
