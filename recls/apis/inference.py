import os.path as osp

import geopandas as gpd
import numpy as np
import torch
from mmcv.utils import track_iter_progress
from scipy.special import logsumexp, softmax

from mmcls.datasets import build_dataloader
from recls.datasets import SceneDataset


def inference_classifier_with_scene(model,
                                    scene_path,
                                    objects=None,
                                    object_file=None,
                                    energy_score=False,
                                    output_dir='inference'):
    """inference patches with the classifier.

    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        crop_size (list): The sizes of patches.
        stride (list): The steps between two patches.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """
    cfg = model.cfg.copy()

    infer_cfg = cfg.get('scene_test_dataset')
    if infer_cfg:
        infer_pipeline = infer_cfg.get('pipeline',
                                       cfg.data.val.get('pipeline'))
    else:
        infer_pipeline = cfg.data.val.get('pipeline')

    object_file = object_file if object_file else infer_cfg.get(
        'object_file', '')

    dataset = SceneDataset(
        scene_path,
        pipeline=infer_pipeline,
        objects=objects,
        object_file=object_file)

    loader_cfg = dict(
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        samples_per_gpu=cfg.data.get('samples_per_gpu', 1))
    loader_cfg.update(cfg.data.get('test_dataloader', {}))
    loader_cfg.update(drop_last=False, shuffle=False, dist=False)
    dataloader = build_dataloader(dataset, **loader_cfg)

    pred_results = []
    for data in track_iter_progress(dataloader):

        with torch.no_grad():

            pred_result = model.process(data, return_logit=energy_score)
            pred_results.append(pred_result)

    pred_results = np.concatenate(pred_results)

    if energy_score:
        energys, softmaxs = calucate_energy_scores(pred_results)

    # appended classes of prediction to object file and save it to output dir
    if object_file and output_dir:
        dst_path = osp.join(output_dir, osp.basename(object_file))
        pred_gdf = gpd.read_file(object_file)
        pred_gdf = pred_gdf.drop(
            index=pred_gdf[pred_gdf['rbox'].isna()].index).reset_index()
        pred_gdf = pred_gdf.drop(['index'], axis='columns')

        pred_class_id = np.argmax(pred_results, axis=-1)

        categories = cfg.data.test.categories
        categories = {cat['id']: cat['name'] for cat in categories}
        if cfg.data.test.rename_class:
            categories_rename = cfg.data.test.rename_class
        else:
            categories_rename = None

        pred_class_name = []
        matched_type_cls = []
        for k, pid in enumerate(pred_class_id):
            pred_name = categories[pid]
            pred_class_name.append(pred_name)

            if pred_gdf['matched_type_det'][k] in ['TP', 'FN']:
                class_rename = categories_rename[
                    pred_gdf['class_name']
                    [k]] if categories_rename else pred_name
                if pred_name == class_rename:
                    matched_type_cls.append('TP')
                else:
                    matched_type_cls.append('FP')
            elif pred_gdf['matched_type_det'][k] in ['FP']:
                matched_type_cls.append('FP')

        pred_gdf['pred_class_id'] = pred_class_id
        pred_gdf['pred_class_name'] = pred_class_name
        pred_gdf['matched_type_cls'] = matched_type_cls
        pred_gdf['cls_score'] = np.max(pred_results, axis=-1)

        if energy_score:
            pred_gdf['energy_score'] = energys
            pred_gdf['softmax_score'] = softmaxs

        pred_gdf.to_file(dst_path)

    return pred_results


def calucate_energy_scores(pred_results, temperature=1.0):
    # In the definition of the energy-based model, the energy score is
    # defined as a negative value. To analyze it further, you can use it as
    # a negative energy score.
    # For more details, please check https://arxiv.org/abs/2010.03759
    energys = -(temperature * logsumexp(pred_results / temperature, axis=1))
    softmaxs = -np.max(softmax(pred_results, axis=1), axis=1)

    return energys, softmaxs
