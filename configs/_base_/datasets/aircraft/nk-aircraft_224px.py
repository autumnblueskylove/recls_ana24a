dataset_type = 'DataPlatformDataset'
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
data_preprocessor = dict(
    num_classes=24,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(
        type='JitterRBox',
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
    dict(type='Resize', scale=(224, 224)),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackInputs', meta_keys=['img_path', 'ilid']),
]

test_pipeline = [
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='Resize', scale=(224, 224)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=train_dataset_id,
        categories=categories),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        host=dp_host,
        user=dp_user,
        password=dp_password,
        dataset_id=test_dataset_id,
        categories=categories),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 3))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
