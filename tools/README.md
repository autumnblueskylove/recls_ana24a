# Note!

We strongly recommend to use SI-Analytics\`s [scheduler service](https://www.notion.so/sianalytics/2cd63223d8f84ec78fb80194120c39ec) to train and test models.

# Train a model

```bash
# Train with a single GPU
python3 tools/train.py ${CONFIG_FILE} [optional arguments]

# Train with multiple GPUs
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

# Test a model

```bash
# single-gpu
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]

# multi-node in slurm environment
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments] --launcher slurm

```

______________________________________________________________________

# Batch inference

Execute batch inference with pre-trained models. You can use the models uploaded in the mlflow or local config and checkpoint.

If you use with mlflow run-id, you must specify `cfg.scene_test_dataset.image_paths` in the dumped configuration file.

```bash
# with mlflow run-id
python3 tools/batch_inference.py --run-id ${RUN_ID} --save-path ${SAVE_PATH}

# with local files
python3 tools/batch_inference.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --image-paths ${IMAGE_PATHS} --save-path ${SAVE_PATH}
```

# Inference and log the predictions to MLflow

Similar to the batch inference, but log the predictions to mlflow as geojson format.

```bash
# with mlflow run-id
python3 tools/inference_and_log.py --run-id ${RUN_ID}
```

______________________________________________________________________

# Example shell script

Train with a single gpu, MLflow and DataPlatform

```bash
#!/bin/bash

# ----------------------------------------------
# Set experiment
# ----------------------------------------------
CONFIG_DIR=TO-BE-SET
CONFIG_NAME=TO-BE-SET
WORK_DIR=TO-BE-SET
EXP_NAME=TO-BE-SET
DP_USER=TO-BE-SET
DP_PASSWORD=TO-BE-SET
#------------------------------------------------

CONFIG_PATH=$CONFIG_DIR/$CONFIG_NAME.py
WORK_PATH=$WORK_DIR/$CONFIG_NAME

python3 tools/train.py $CONFIG_PATH --work-dir $WORK_PATH  \
  --cfg-options exp_name=$EXP_NAME run_name=$CONFIG_NAME dp_user=$DP_USER dp_password=$DP_PASSWORD

```

______________________________________________________________________

# Useful Tools

### Visualizing Pipeline

- Refer to [mmclassification](https://mmclassification.readthedocs.io/en/master/tools/visualization.html)
- Support DataPlatformV2
  - ```
    python3 tools/visualizations/vis_pipeline.py \
        CONFIGPATH, --output-dir OUTPUTDIR --number NUMBER \
        --cfg-options dp_user=DP_USER dp_password=DP_PASSWORD
    ```
