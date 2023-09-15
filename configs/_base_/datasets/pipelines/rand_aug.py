# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
rand_increasing_policies = [
    dict(type='mmpretrain.AutoContrast'),
    dict(type='mmpretrain.Equalize'),
    dict(type='mmpretrain.Invert'),
    dict(
        type='mmpretrain.Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30)),
    dict(
        type='mmpretrain.Posterize',
        magnitude_key='bits',
        magnitude_range=(4, 0)),
    dict(
        type='mmpretrain.Solarize',
        magnitude_key='thr',
        magnitude_range=(256, 0)),
    dict(
        type='mmpretrain.SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='mmpretrain.ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmpretrain.Contrast',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmpretrain.Brightness',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmpretrain.Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='mmpretrain.Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='mmpretrain.Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='mmpretrain.Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal',
    ),
    dict(
        type='mmpretrain.Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical'),
]
