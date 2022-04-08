shufflenet_v2_1x = dict(type='ImageClassifier',
                        backbone=dict(type='ShuffleNetV2', widen_factor=1.0),
                        neck=dict(type='GlobalAveragePooling'),
                        head=dict(
                            type='LinearClsHead',
                            num_classes=3,
                            in_channels=1024,
                            loss=dict(type='CrossEntropyLoss',
                                      loss_weight=1.0),
                            topk=(1, 5),
                        ))

seresnet101 = dict(type='ImageClassifier',
                   backbone=dict(type='SEResNet',
                                 depth=101,
                                 num_stages=4,
                                 out_indices=(3, ),
                                 style='pytorch'),
                   neck=dict(type='GlobalAveragePooling'),
                   head=dict(
                       type='LinearClsHead',
                       num_classes=3,
                       in_channels=2048,
                       loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                       topk=(1, 5),
                   ))

resnext101_32x8d = dict(type='ImageClassifier',
                        backbone=dict(type='ResNeXt',
                                      depth=101,
                                      num_stages=4,
                                      out_indices=(3, ),
                                      groups=32,
                                      width_per_group=8,
                                      style='pytorch'),
                        neck=dict(type='GlobalAveragePooling'),
                        head=dict(
                            type='LinearClsHead',
                            num_classes=3,
                            in_channels=2048,
                            loss=dict(type='CrossEntropyLoss',
                                      loss_weight=1.0),
                            topk=(1, 5),
                        ))

resnetv1d152 = dict(type='ImageClassifier',
                    backbone=dict(type='ResNetV1d',
                                  depth=152,
                                  num_stages=4,
                                  out_indices=(3, ),
                                  style='pytorch'),
                    neck=dict(type='GlobalAveragePooling'),
                    head=dict(
                        type='LinearClsHead',
                        num_classes=3,
                        in_channels=2048,
                        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                        topk=(1, 5),
                    ))

resnest269 = dict(type='ImageClassifier',
                  backbone=dict(type='ResNeSt',
                                depth=269,
                                num_stages=4,
                                stem_channels=128,
                                out_indices=(3, ),
                                style='pytorch'),
                  neck=dict(type='GlobalAveragePooling'),
                  head=dict(type='LinearClsHead',
                            num_classes=3,
                            in_channels=2048,
                            loss=dict(type='LabelSmoothLoss',
                                      label_smooth_val=0.1,
                                      num_classes=1000,
                                      reduction='mean',
                                      loss_weight=1.0),
                            topk=(1, 5),
                            cal_acc=False))

res2net101_w26_s4 = dict(type='ImageClassifier',
                         backbone=dict(
                             type='Res2Net',
                             depth=101,
                             scales=4,
                             base_width=26,
                             deep_stem=False,
                             avg_down=False,
                         ),
                         neck=dict(type='GlobalAveragePooling'),
                         head=dict(
                             type='LinearClsHead',
                             num_classes=3,
                             in_channels=2048,
                             loss=dict(type='CrossEntropyLoss',
                                       loss_weight=1.0),
                             topk=(1, 5),
                         ))

mobilenet_v3_large = dict(type='ImageClassifier',
                          backbone=dict(type='MobileNetV3', arch='large'),
                          neck=dict(type='GlobalAveragePooling'),
                          head=dict(type='StackedLinearClsHead',
                                    num_classes=3,
                                    in_channels=960,
                                    mid_channels=[1280],
                                    dropout_rate=0.2,
                                    act_cfg=dict(type='HSwish'),
                                    loss=dict(type='CrossEntropyLoss',
                                              loss_weight=1.0),
                                    topk=(1, 5)))

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type='GeoCOCODataset',
        data_prefix=  # noqa E251
        '/nas/k8s/dev/mlops/dataset-artifacts/glc21a/CLS/running_car/version_2/GeoCOCO_Train',
        ann_file='GeoCOCO.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=(64, 64), scale=(1.0, 1.0)),
            dict(type='GaussianNoise'),
            dict(type='Rotate', angle=20.0),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='GeoCOCODataset',
        data_prefix=  # noqa E251
        '/nas/k8s/dev/mlops/dataset-artifacts/glc21a/CLS/running_car/version_2/GeoCOCO_Val',
        ann_file='GeoCOCO.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=(64, 64), scale=(1.0, 1.0)),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=200,
                  metric='accuracy',
                  metric_options=dict(topk=(1, 3), average_mode='macro'))

sgd_optimizer = dict(type='SGD', lr=0.005, weight_decay=1e-05, momentum=0.9)
adam_optimizer = dict(type='Adam')
rms_optimizer = dict(type='RMSprop')
adamw_optimizer = dict(type='AdamW')

optimizer_config = dict(grad_clip=None)

step_lr_config = dict(policy='step',
                      by_epoch=False,
                      warmup='linear',
                      warmup_iters=500,
                      warmup_ratio=0.3333,
                      step=[3000, 5000])
cosine_lr_config = dict(policy='CosineAnnealing', min_lr=0)
poly_lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000,
)

runner = dict(type='IterBasedRunner', max_iters=16000)
checkpoint_config = dict(interval=2000)
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
resume_from = None
load_from = None

search_space = {
    'model':
    dict(type='Choice',
         categories=[
             shufflenet_v2_1x, seresnet101, resnext101_32x8d, resnetv1d152,
             resnest269, res2net101_w26_s4, mobilenet_v3_large
         ],
         alias=[
             'shufflenet_v2_1x', 'seresnet101', 'resnext101_32x8d',
             'resnetv1d152', 'resnest269', 'res2net101_w26_s4',
             'mobilenet_v3_large'
         ]),
    'data.samples_per_gpu':
    dict(type='Randint', lower=64, upper=128),
    'optimizer':
    dict(type='Choice',
         categories=[
             sgd_optimizer, adam_optimizer, rms_optimizer, adamw_optimizer
         ],
         alias=[
             'sgd_optimizer', 'adam_optimizer', 'rms_optimizer',
             'adamw_optimizer'
         ]),
    'lr_config':
    dict(type='Choice',
         categories=[step_lr_config, cosine_lr_config, poly_lr_config],
         alias=['step', 'cosine', 'poly'])
}

search_metric = dict(
    metric='val/macro_accuracy_top-1',
    mode='max',
)

search_algorithm = dict(
    type='NevergradSearch',
    optimizer='$(nevergrad.optimizers.NGOpt)',
    **search_metric,
    budget=100,
)

search_scheduler = dict(type='AsyncHyperBandScheduler',
                        time_attr='training_iteration',
                        max_t=20,
                        grace_period=2)

tuning = dict(
    num_samples=100,
    raise_on_failed_trial=False,
    config=search_space,
    callbacks=[
        dict(type='MLflowLoggerCallback',
             save_artifact=True,
             experiment_name='CLS-HPO-TEST'),
    ],
    scheduler=search_scheduler,
    search_alg=search_algorithm,
    **search_metric,
)
