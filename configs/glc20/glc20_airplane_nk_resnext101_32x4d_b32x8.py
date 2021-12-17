_base_ = [
    "../_base_/models/resnext101_32x4d.py",
    "../_base_/datasets/geococo_glc20_airplane_nk.py",
    "../_base_/schedules/geococo.py",
    "../_base_/default_runtime.py",
]

_deprecation_ = dict(
    expected="resnext101-32x4d_8xb32_in1k.py",
    reference="https://github.com/open-mmlab/mmclassification/pull/508",
)
# model settings
model = dict(head=dict(num_classes=27))
