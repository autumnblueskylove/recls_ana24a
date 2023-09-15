# model settings
# Only for evaluation
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.SwinTransformerV2',
        arch='large',
        img_size=384,
        drop_path_rate=0.2),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
