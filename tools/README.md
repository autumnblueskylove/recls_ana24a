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

### TODO

______________________________________________________________________

# Useful Tools

### Shell Script for train, Test and evaluation

- By single command, train, test and evaluation are run with MLflow.
- [train_script.sh](../train_script.sh)

### Dataset Visualization

- It can save processed images for input model by config file.
- Refer to [mmpretrain](https://mmpretrain.readthedocs.io/en/latest/useful_tools/dataset_visualization.html)
