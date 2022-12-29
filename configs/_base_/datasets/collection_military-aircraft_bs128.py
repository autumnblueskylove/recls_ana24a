dataset_type = 'DataPlatformDatasetV2'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = None
dp_password = None

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

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=train_dataset_id,
        categories=categories),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        categories=categories),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        categories=categories))

evaluation = dict(interval=1, metric='accuracy')
