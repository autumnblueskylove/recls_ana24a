# dataset settings
dataset_type = "GeoCOCODataset"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="RandomResizedCrop", size=(64, 32), scale=(1.,1.,)),
    dict(type="Resize", size=(64, 32)),
    dict(type="GaussianNoise"),
    dict(type="Rotate", angle=20),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", flip_prob=0.5, direction="vertical"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(64, 32)),
    dict(type="CenterCrop", crop_size=(64,32)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix="/nas/Dataset/RSI_CL_GLC21/geococo/facilities_car/GeoCOCO_CLS/v1_2/GeoCOCO_Train/",
        ann_file="GeoCOCO.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="/nas/Dataset/RSI_CL_GLC21/geococo/facilities_car/GeoCOCO_CLS/v1_2/GeoCOCO_Val/",
        ann_file="GeoCOCO.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix="/nas/Dataset/RSI_CL_GLC21/geococo/facilities_car/GeoCOCO_CLS/v1_2/GeoCOCO_Val/",
        ann_file="GeoCOCO.json",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=200, metric="accuracy", metric_options=dict(topk=(1, 3), average_mode="macro"))
