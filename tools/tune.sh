#!/usr/bin/env bash

CONFIG=$1
WORKERS=$2
CPUS=${3:-1}
GPUS=${4:-1}

python3 -m mmray.tools.tune --num-workers $WORKERS --num-cpus-per-worker $CPUS --num-gpus-per-worker $GPUS --build-model recls.models.build_classifier --train-model recls.apis.train_model $CONFIG  ${@:5}
