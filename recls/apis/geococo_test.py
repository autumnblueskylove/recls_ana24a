import os
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def collect_results(result_part):

    rank, world_size = get_dist_info()
    tmpdir = '.dist_test'

    if rank == 0:
        os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    mmcv.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank != 0:
        return None
    else:
        part_list = list()
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.extend(part_result)
        return part_list


def inference_single_gpu_dp_model(model, data_loader, tmpdir=None):

    model.eval()
    results = []
    dataset = data_loader.dataset
    time.sleep(2)

    filenames = list()
    results = list()
    gt_labels = list()
    label_uuids = list()

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        filename = data['img_metas'].data[0]
        gt_label = list(data['gt_label'].cpu().numpy())
        with torch.no_grad():
            result = model(
                return_loss=False,
                img=data['img'].cuda(),
                img_metas=data['img_metas'])

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if isinstance(filename, list):
            filenames.extend(filename)
        else:
            filenames.append(filename)

        if isinstance(gt_label, list):
            gt_labels.extend(gt_label)
        else:
            gt_labels.append(gt_label)

        if 'label_uuid' in data:
            label_uuids += list(data['label_uuid'].cpu().numpy())
        else:
            label_uuids += [None] * len(gt_label)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    info_results = [
        dict(
            filename=filename,
            label=label,
            result=result,
            label_uuid=label_uuid) for result, filename, label, label_uuid in
        zip(results, filenames, gt_labels, label_uuids)
    ]
    return info_results


def inference_geococo_model(model,
                            data_loader,
                            tmpdir=None,
                            gpu_collect=False):

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None
                                  and os.path.exists(tmpdir)):
            raise OSError((
                f'The tmpdir {tmpdir} already exists.',
                ' Since tmpdir will be deleted after testing,',
                ' please make sure you specify an empty one.',
            ))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier()

    filenames = list()
    results = list()
    gt_labels = list()

    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'].data[0]
        filename = [img_meta['filename'] for img_meta in img_metas]
        gt_label = list(data['gt_label'].cpu().numpy())
        with torch.no_grad():
            result = model(
                return_loss=False,
                img=data['img'],
                img_metas=data['img_metas'])

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if isinstance(filename, list):
            filenames.extend(filename)
        else:
            filenames.append(filename)

        if isinstance(gt_label, list):
            gt_labels.extend(gt_label)
        else:
            gt_labels.append(gt_label)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    info_results = [
        dict(filename=filename, label=label, result=result)
        for result, filename, label in zip(results, filenames, gt_labels)
    ]

    results = collect_results(info_results)
    return results
