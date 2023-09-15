# model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(
            type='mmpretrain.CrossEntropyLoss', loss_weight=1.0,
            use_soft=True)),
    train_cfg=dict(
        augments=dict(type='BatchCutMix', alpha=1.0, num_classes=10,
                      prob=1.0)))
