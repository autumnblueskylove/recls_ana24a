# model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.SwinTransformerV2',
        arch='base',
        img_size=384,
        drop_path_rate=0.2),
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
        cal_acc=False))
