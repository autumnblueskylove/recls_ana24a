_base_ = ['../pipelines/rand_aug.py']
dataset_type = 'DataPlatformDataset'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = 'jmkoo@si-analytics.ai'
dp_password = 'jmkoo@sia2022'

train_dataset_id = 407
test_dataset_id = 408
categories = [
    dict(id=0, name='군사장비'),
    dict(id=1, name='군사차량'),
    dict(id=2, name='자주포'),
    dict(id=3, name='전차'),
]

expand_ratio = 1.2
data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    num_classes=4,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

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
        min_percentile_range=(0.0, 1.0),
        max_percentile_range=(99.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(
        type='mmpretrain.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmpretrain.RandAugment',
        policies=_base_.rand_increasing_policies,
        # policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='mmpretrain.RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='mmpretrain.PackInputs', meta_keys=['img_path', 'ilid']),
]

test_pipeline = [
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(type='mmpretrain.PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
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
    num_workers=8,
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
val_evaluator = [
    dict(type='mmpretrain.Accuracy', topk=(1, 3)),
    dict(type='mmpretrain.SingleLabelMetric')
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator
