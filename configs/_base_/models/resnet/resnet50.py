# model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
