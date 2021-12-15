# dataset settings
dataset_type = 'GeoCOCODataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=128),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(128, -1)),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/nas/Dataset/Private/GLC20/항공기/geococo/북한/GeoCOCO_CLS/v1_0/GeoCOCO_Train',
        ann_file='GeoCOCO.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/nas/Dataset/Private/GLC20/항공기/geococo/북한/GeoCOCO_CLS/v1_0/GeoCOCO_Val/',
        ann_file='GeoCOCO.json',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/nas/Dataset/Private/GLC20/항공기/geococo/북한/GeoCOCO_CLS/v1_0/GeoCOCO_Val/',
        ann_file='GeoCOCO.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(
    topk=(1,3)))
