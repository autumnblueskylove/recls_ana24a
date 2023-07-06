# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
from collections import defaultdict
from pathlib import Path

import mmcv
import mmengine
import torch
from mmengine import DictAction
from mmpretrain.datasets import build_dataset
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer

import recls  # noqa: F401
from recls.datasets.transforms.processing import (CropInstanceInScene,
                                                  stretch_image)
from recls.utils import download_artifacts, log_artifact


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain evaluate prediction success/fail')
    parser.add_argument(
        '--run-id',
        default=None,
        type=str,
        help='run id of mlflow to get config and results files and upload '
        'visualized images.')
    parser.add_argument(
        '--config', default=None, type=Path, help='test config file path')
    parser.add_argument(
        '--result', default=None, type=Path, help='test result json/pkl file')
    parser.add_argument(
        '--out-dir',
        default='.tmpdir/vis_eval',
        type=str,
        help='dir to store output files')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='Number of images to select for success/fail')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--descend-sort',
        action='store_true',
        help='whether to sort by descending for top-k')
    parser.add_argument(
        '--expand_ratio',
        default=1.5,
        type=float,
        help='expand ratio to crop size.')
    parser.add_argument(
        '--topk-fail',
        default=None,
        type=int,
        help='Number of images to selcet for fail and ignore top-k argument.')
    parser.add_argument(
        '--dump-info',
        action='store_true',
        help='whether not to dump information as json.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--tmpdir', type=str, default='.tmpdir')

    args = parser.parse_args()

    return args


def save_imgs(result_dir,
              folder_name,
              results,
              dataset,
              rescale_factor=None,
              expand_ratio=1.0,
              dump_info=False):
    full_dir = osp.join(result_dir, folder_name)
    vis = UniversalVisualizer()
    vis.dataset_meta = {'classes': dataset.CLASSES}

    crop_instance = CropInstanceInScene(expand_ratio)

    # save imgs
    dump_infos = []
    for data_sample in results:
        data_info = dataset.get_data_info(data_sample.sample_idx)
        if 'ilid' in data_info:
            img = stretch_image(crop_instance(data_info)['img'])
            name = f"{Path(data_info['img_path']).stem}_{data_info['ilid']}"
        elif 'img' in data_info:
            img = data_info['img']
            name = str(data_sample.sample_idx)
        elif 'img_path' in data_info:
            img = mmcv.imread(data_info['img_path'], channel_order='rgb')
            name = Path(data_info['img_path']).name
        else:
            raise ValueError('Cannot load images from the dataset infos.')
        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)
        vis.visualize_cls(
            img, data_sample, out_file=osp.join(full_dir, name + '.png'))

        if dump_info:
            dump = dict()
            for k, v in data_sample.items():
                if isinstance(v, torch.Tensor):
                    dump[k] = v.tolist()
                else:
                    dump[k] = v
                dump_infos.append(dump)

    if dump_info:
        mmengine.dump(dump_infos, osp.join(full_dir, folder_name + '.json'))


def update_args_for_mlflow(args):
    """Download model config and weight file from mlflow and update arguments
    relating paths from run id of mlflow.

    Args:
        args: Arguments that have to include run_id.

    Returns:
        Namespace: Updated arguments, "config" and "result" variables.
    """
    shutil.rmtree(args.tmpdir, ignore_errors=True)
    mmengine.mkdir_or_exist(args.tmpdir)

    download_artifacts(args.run_id, 'checkpoint/model_config.py', args.tmpdir)
    download_artifacts(args.run_id, 'inference/results.pkl', args.tmpdir)

    args.config = osp.join(args.tmpdir, 'checkpoint/model_config.py')
    args.result = osp.join(args.tmpdir, 'inference/results.pkl')


def main():
    args = parse_args()

    if args.run_id:
        update_args_for_mlflow(args)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the dataloader
    cfg.test_dataloader.dataset.pipeline = []
    dataset = build_dataset(cfg.test_dataloader.dataset)

    results = list()
    for result in mmengine.load(args.result):
        data_sample = DataSample()
        data_sample.set_metainfo({'sample_idx': result['sample_idx']})
        data_sample.set_gt_label(result['gt_label'])
        data_sample.set_pred_label(result['pred_label'])
        data_sample.set_pred_score(result['pred_score'])
        results.append(data_sample)

    # sort result
    results = sorted(
        results,
        key=lambda x: torch.max(x.pred_score),
        reverse=args.descend_sort)

    # get success and fail cases by classes.
    success = defaultdict(list)
    fail = defaultdict(list)
    for data_sample in results:
        if (data_sample.pred_label == data_sample.gt_label).all():
            success[int(data_sample.gt_label)].append(data_sample)
        else:
            fail[int(data_sample.gt_label)].append(data_sample)

    # save images of success and fail cases by classes.
    n_fail = args.topk if not args.topk_fail else args.topk_fail
    for cls_idx, cls_name in enumerate(dataset.CLASSES):
        cls_success = success[cls_idx][:args.topk]
        cls_fail = fail[cls_idx][:n_fail]

        save_imgs(args.out_dir, f'{cls_name}/success', cls_success, dataset,
                  args.rescale_factor, args.expand_ratio, args.dump_info)
        save_imgs(args.out_dir, f'{cls_name}/fail', cls_fail, dataset,
                  args.rescale_factor, args.expand_ratio, args.dump_info)

    if args.run_id:
        log_artifact(args.run_id, args.out_dir, 'visualization')


if __name__ == '__main__':
    main()
