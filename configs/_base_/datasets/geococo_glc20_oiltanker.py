# dataset settings
dataset_type = 'GeoCOCODataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(128, 64)),
    # dict(type="GaussianNoise"),
    dict(type='RandomGaussianBlur'),
    dict(type='Rotate', angle=20.0),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(128, 64)),
    dict(type='CenterCrop', crop_size=(128, 64)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=  # noqa E251
        '/nas/Dataset/Private/GLC20/유조선/geococo/GeoCOCO_CLS/v1_3/GeoCOCO_Train',
        ann_file='GeoCOCO.json',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix=  # noqa E251
        '/nas/Dataset/Private/GLC20/유조선/geococo/GeoCOCO_CLS/v1_3/GeoCOCO_Val/',
        ann_file='GeoCOCO.json',
        pipeline=test_pipeline,
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=  # noqa E251
        '/nas/Dataset/Private/GLC20/유조선/geococo/GeoCOCO_CLS/v1_3/GeoCOCO_Val/',
        ann_file='GeoCOCO.json',
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=200,
                  metric='accuracy',
                  metric_options=dict(topk=(1, 3), average_mode='macro'))
