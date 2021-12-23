_base_ = [
    "../_base_/models/resnext101_32x4d.py",
    "../_base_/datasets/geococo_glc20_oiltanker.py",
]

_deprecation_ = dict(
    expected="resnext101-32x4d_8xb32_in1k.py",
    reference="https://github.com/open-mmlab/mmclassification/pull/508",
)

# optimizer
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", by_epoch=False, warmup='linear', warmup_iters=500,warmup_ratio=0.3333, step=[3000, 5000],)
runner = dict(type="IterBasedRunner", max_iters=8000)

# checkpoint saving
checkpoint_config = dict(interval=2000)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = "https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth"
resume_from = None
workflow = [("train", 1)]


# model settings
model = dict(head=dict(num_classes=42))
