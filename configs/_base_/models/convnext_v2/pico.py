# Model settings
model = dict(
    type='mmpretrain.ImageClassifier',
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='pico',
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
    ),
    head=dict(
        type='mmpretrain.LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='mmpretrain.LabelSmoothLoss', label_smooth_val=0.1),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
)
