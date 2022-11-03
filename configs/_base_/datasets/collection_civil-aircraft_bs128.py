dataset_type = 'DataPlatformDatasetV2'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = None
dp_password = None

train_dataset_id = 61
test_dataset_id = 62
categories = [
    dict(id=0, name='Boeing707'),
    dict(id=1, name='Boeing717'),
    dict(id=2, name='Boeing727'),
    dict(id=3, name='Boeing737'),
    dict(id=4, name='Boeing747'),
    dict(id=5, name='Boeing757'),
    dict(id=6, name='Boeing767'),
    dict(id=7, name='Boeing777'),
    dict(id=8, name='Boeing787'),
    dict(id=9, name='A220'),
    dict(id=10, name='A321'),
    dict(id=11, name='A330'),
    dict(id=12, name='A340'),
    dict(id=13, name='A350'),
    dict(id=14, name='A380'),
]

rename_class = {
    'a-330': 'A330',
    'a-340': 'A340',
    'a-350': 'A350',
    'a-380': 'A380',
    'b-707': 'Boeing707',
    'b-717': 'Boeing717',
    'b-727': 'Boeing727',
    'b-737': 'Boeing737',
    'b-747': 'Boeing747',
    'b-757': 'Boeing757',
    'b-767': 'Boeing767',
    'b-777': 'Boeing777',
    'b-787': 'Boeing787',
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
