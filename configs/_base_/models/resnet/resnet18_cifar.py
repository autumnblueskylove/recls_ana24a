# model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
    ))
