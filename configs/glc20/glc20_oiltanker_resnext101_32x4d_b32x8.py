_base_ = [
    "../_base_/models/resnext101_32x4d.py",
    "../_base_/datasets/geococo_glc20_oiltanker.py",
    "../_base_/default_runtime.py",
]

# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="step", step=[200, 300])
runner = dict(type="EpochBasedRunner", max_epochs=400)

_deprecation_ = dict(
    expected="resnext101-32x4d_8xb32_in1k.py",
    reference="https://github.com/open-mmlab/mmclassification/pull/508",
)
# model settings
model = dict(head=dict(num_classes=42))
