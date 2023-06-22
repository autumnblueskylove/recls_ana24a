#!/bin/bash

#---------- CONFIG ------------------------
# DP
DP_USER=TO_BE_SET
DP_PASSWORD=TO_BE_SET

# Set mlflow
EXP_NAME=TO_BE_SET
RUN_NAME=TO_BE_SET

# Set config path
CONFIG_DIR=TO_BE_SET
CONFIG_NAME=TO_BE_SET_WITHOUT_PY
CONFIG_PATH=$CONFIG_DIR/$CONFIG_NAME.py

# Set work directory
WORK_DIR=TO_BE_SET
WORK_PATH=$WORK_DIR/$CONFIG_NAME
#----------------------------------------

# create mlflow run
RUN_ID=$(python3 tools/misc/create_mlflow_run.py --exp-name $EXP_NAME --run-name $RUN_NAME)

# train
python3 tools/train.py $CONFIG_PATH --work-dir $WORK_PATH  \
 --cfg-options run_id=$RUN_ID dp_user=$DP_USER dp_password=$DP_PASSWORD

# test
python3 tools/test.py 'config_dummy' 'checkpoint_dummy' \
  --run-id $RUN_ID \
  --work-dir $WORK_PATH \

# analysis
python3 tools/analysis.py --run-id=$RUN_ID
