import argparse
import os

import mmcv
import torch
import torch.distributed as dist
from mlflow.tracking import MlflowClient
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import track_iter_progress

from mmcls.utils import setup_multi_processes
from recls.apis import Classifier, inference_classifier_with_scene

ARTIFACT_PATH = 'checkpoint'
ARTIFACT_INFERENCE_PATH = os.path.join(ARTIFACT_PATH, 'inference')
MODEL_CONFIG_FILE = 'model_config.py'
MODEL_WEIGHT_FILE = 'model_final.pth'


def parse_args():
    parser = argparse.ArgumentParser(
        description='ClasyMM batch inference (and log) a model')
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default=None,
        help='test config file path')
    parser.add_argument(
        'checkpoint',
        type=str,
        nargs='?',
        default=None,
        help='checkpoint file')
    parser.add_argument(
        '--path',
        '--paths',
        '--image-paths',
        type=str,
        nargs='*',
        default=[],
        help='path to input image dirs or files. if no path is given, '
        'it will use `cfg.scene_test_dataset.image_paths` values.')
    parser.add_argument(
        '--object-path',
        type=str,
        help='path to input objects such as geojson by detection results')
    parser.add_argument(
        '--run-id', type=str, default=None, help='mlflow run id')
    parser.add_argument(
        '--save-path',
        '--show-dir',
        type=str,
        default='inference',
        help='path to store images and if not given, will not save image')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tmpdir', type=str, default='.batch_inference')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    if args.run_id:
        print('Downloading config and checkpoint from mlflow...')
        config = os.path.join(ARTIFACT_PATH, MODEL_CONFIG_FILE)
        checkpoint = os.path.join(ARTIFACT_PATH, MODEL_WEIGHT_FILE)

        rank, world_size = get_dist_info()
        if rank == 0:
            if os.path.exists(os.path.join(
                    args.tmpdir, config)) and os.path.exists(
                        os.path.join(args.tmpdir, checkpoint)):
                print(f'RUN_ID: {args.run_id} is already downloaded.')
                pass
            else:
                mmcv.mkdir_or_exist(args.tmpdir)
                cli = MlflowClient()
                cli.download_artifacts(args.run_id, config, args.tmpdir)
                cli.download_artifacts(args.run_id, checkpoint, args.tmpdir)
        if world_size > 1:
            dist.barrier()

        args.config = os.path.join(args.tmpdir, config)
        args.checkpoint = os.path.join(args.tmpdir, checkpoint)
        if not (os.path.isfile(args.config)
                and os.path.isfile(args.checkpoint)):
            RuntimeError('Cannot find config and checkpoint in mlflow run id: '
                         '{}'.format(args.run_id))
    else:
        assert args.config and args.checkpoint, \
            ('config and checkpoint should be given '
             'if mlflow run id is not given.')
        print(f'RUN ID for mlflow is not configured.'
              f'Will use config and checkpoint from {args.config} and '
              f'{args.checkpoint}')

    return args


def parse_scenes(paths):
    EXTENSIONS = ('.png', '.tif', '.tiff', '.jp2')

    if isinstance(paths, str):
        paths = [paths]

    scenes = []
    for path in paths:
        if not os.path.exists(path) or os.path.isfile(
                path) and not os.path.splitext(path)[-1].lower() in EXTENSIONS:
            raise ValueError

        if os.path.isfile(path):
            scenes.append(path)
            continue

        # if given path is directory,
        scenes.extend(
            os.path.join(root, file) for root, _, files in os.walk(path)
            for file in files
            if os.path.splitext(file)[-1].lower() in EXTENSIONS)

    if not scenes:
        raise ValueError('There is no valid scene file.')
    return sorted(scenes)


def main():
    args = parse_args()
    model = Classifier(os.path.dirname(args.config))
    cfg = model.cfg

    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)

    rank, world_size = get_dist_info()
    files = parse_scenes(
        args.path or cfg.get('scene_test_dataset').image_paths)
    for file in track_iter_progress(files[rank::world_size]):

        inference_classifier_with_scene(
            model,
            file,
            object_file=args.object_path,
            output_dir=args.save_path)

        if args.run_id and args.save_mlflow:
            cli = MlflowClient()
            cli.log_artifacts(
                args.run_id,
                args.save_path,
                ARTIFACT_INFERENCE_PATH,
            )


if __name__ == '__main__':
    main()
