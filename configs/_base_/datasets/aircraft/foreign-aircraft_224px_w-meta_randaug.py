_base_ = ['../pipelines/rand_aug.py']
dataset_type = 'DataPlatformDataset'
dp_host = 'http://dev-cluster.sia-service.kr:32160'
dp_user = 'jmkoo'
dp_password = 'jmkoo@sia2022'

train_dataset_id = 284
test_dataset_id = 286
categories = [
    dict(id=0, name='AN-12'),
    dict(id=1, name='AN-26'),
    dict(id=2, name='C-130'),
    dict(id=3, name='E-767'),
    dict(id=4, name='F-15'),
    dict(id=5, name='F-35'),
    dict(id=6, name='H-6'),
    dict(id=7, name='Il-78'),
    dict(id=8, name='J-7'),
    dict(id=9, name='J-10'),
    dict(id=10, name='J-11'),
    dict(id=11, name='J-16'),
    dict(id=12, name='J-20'),
    dict(id=13, name='KJ-200'),
    dict(id=14, name='KJ-500'),
    dict(id=15, name='MiG-31'),
    dict(id=16, name='Su-24'),
    dict(id=17, name='Su-25'),
    dict(id=18, name='Su-27'),
    dict(id=19, name='Su-30'),
    dict(id=20, name='Su-34'),
    dict(id=21, name='Tu-22'),
    dict(id=22, name='Tu-95'),
    dict(id=23, name='UH-60'),
    dict(id=24, name='Y-8'),
    dict(id=25, name='Y-9'),
    dict(id=26, name='Unknown'),
    dict(id=26, name='MiG-29'),
    dict(id=26, name='RQ-4'),
    dict(id=26, name='Y-7'),
    dict(id=26, name='J-15'),
    dict(id=26, name='Mi-26'),
    dict(id=26, name='MiG-29'),
    dict(id=26, name='Y-20'),
]

expand_ratio = 1.2
data_preprocessor = dict(
    type='mmpretrain.ClsDataPreprocessor',
    num_classes=27,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
    ),
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
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos', 'xy_gsd']),
]

test_pipeline = [
    dict(
        type='PreprocessMeta',
        include_longlat=True,
    ),
    dict(type='CropInstanceInScene', expand_ratio=expand_ratio),
    dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 0.0),
        max_percentile_range=(100.0, 100.0)),
    dict(type='mmpretrain.Resize', scale=(224, 224)),
    dict(
        type='mmpretrain.PackInputs',
        meta_keys=['img_path', 'ilid', 'meta_infos', 'xy_gsd']),
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
        categories=categories,
        include_longlat=True,
        include_date=True,
        include_gsd=True),
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
        categories=categories,
        include_longlat=True,
        include_date=True,
        include_gsd=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='mmpretrain.Accuracy', topk=(1, 3)),
    dict(type='mmpretrain.SingleLabelMetric')
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator
