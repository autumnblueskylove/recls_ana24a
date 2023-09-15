# model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.SwinTransformerV2',
        arch='base',
        img_size=256,
        drop_path_rate=0.5),
    neck=dict(type='mmpretrain.GlobalAveragePooling'),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='mmpretrain.LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
