# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle

import mlflow
import mmcv
import torch
from mmcv.runner import load_checkpoint

import recls  # noqa: F401
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from recls.apis import inference_single_gpu_dp_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='ClasyMM batch inference (and log) a model')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--run-id', type=str)
    parser.add_argument('--artifact-path', type=str)
    parser.add_argument('--local-path', type=str, default='ckpt')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def download_artifact():

    args = parse_args()
    os.makedirs(args.local_path, exist_ok=True)

    client = mlflow.tracking.MlflowClient()
    download_files = [
        'model_config.py',
        'model_final.pth',
    ]
    for download_file in download_files:
        client.download_artifacts(
            args.run_id, os.path.join(args.artifact_path, download_file),
            args.local_path)


def mlflow_log_artifact():

    args = parse_args()
    client = mlflow.tracking.MlflowClient()
    client.log_artifact(
        run_id=args.run_id,
        local_path=os.path.join(
            args.local_path,
            args.artifact_path,
            'result.pkl',
        ),
        artifact_path=os.path.join(
            'checkpoint',
            'inference',
        ),
    )


def main():

    args = parse_args()
    args.config = os.path.join(args.local_path, args.artifact_path,
                               'model_config.py')
    args.checkpoint = os.path.join(args.local_path, args.artifact_path,
                                   'model_final.pth')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    for pipeline in cfg.data.val.pipeline:
        if pipeline['type'] == 'Collect':
            pipeline['keys'].append('gt_label')

    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        num_gpus=1,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        round_up=True,
    )

    model = build_classifier(cfg.model)
    model = model.cuda()
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    results = inference_single_gpu_dp_model(model, dataloader)
    with open(
            os.path.join(args.local_path, args.artifact_path, 'result.pkl'),
            'wb') as f:
        pickle.dump(results, f)

    mlflow_log_artifact()


if __name__ == '__main__':
    download_artifact()
    main()
