dataset_type = 'DataPlatformDatasetV2'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = None
dp_password = None

train_dataset_id = 68
test_dataset_id = 69
categories = [
    dict(id=0, name='AN-148'),
    dict(id=1, name='AN-2'),
    dict(id=2, name='AN-24'),
    dict(id=3, name='IL-18'),
    dict(id=4, name='IL-28'),
    dict(id=5, name='IL-62'),
    dict(id=6, name='IL-76'),
    dict(id=7, name='L-39'),
    dict(id=8, name='MI-2'),
    dict(id=9, name='MI-26'),
    dict(id=10, name='MI-4'),
    dict(id=11, name='MI-8·17'),
    dict(id=12, name='MIG-15·17'),
    dict(id=13, name='MIG-19'),
    dict(id=14, name='MIG-21'),
    dict(id=15, name='MIG-23'),
    dict(id=16, name='MIG-29'),
    dict(id=17, name='SU-25'),
    dict(id=18, name='TU-134'),
    dict(id=19, name='TU-154'),
    dict(id=20, name='TU-204'),
    dict(id=21, name='YAK-18'),
    dict(id=22, name='DECOY'),
    dict(id=23, name='light-aircraft'),
    dict(id=24, name='unknown'),
]

rename_class = {
    'AS-350': 'unknown',
    'H-6계열': 'unknown',
    'MI-17': 'MI-8·17',
    'MIG-15': 'MIG-15·17',
    'MIG-17': 'MIG-15·17',
    'PA-46': 'unknown',
    'an-2': 'AN-2',
    'decoy': 'DECOY',
    'mi-2': 'MI-2',
    'mi-26': 'MI-26',
    'mi-4': 'MI-4',
    'mi-8_17': 'MI-8·17',
    'mig-15_17': 'MIG-15·17',
    'mig-19': 'MIG-19',
    'mig-21': 'MIG-21',
    'mig-23': 'MIG-23',
    'mig-29': 'MIG-29',
    'su-25': 'SU-25',
    '기타 북한항공기': 'unknown',
    '기타_경비행기': 'light-aircraft',
    '민항기': 'unknown',
    '세스나-172': 'light-aircraft',
    '세스나-208': 'light-aircraft',
    '세스나-510': 'light-aircraft',
}

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
        rename_class=rename_class,
        categories=categories),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        rename_class=rename_class,
        categories=categories),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        rename_class=rename_class,
        categories=categories))

evaluation = dict(interval=1, metric='accuracy')
