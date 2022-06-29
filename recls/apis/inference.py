import numpy as np
import torch
from mmcv.utils import track_iter_progress

from mmcls.datasets import build_dataloader
from recls.datasets import SceneDataset


def inference_classifier_with_scene(model,
                                    scene_path,
                                    objects=None,
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
    infer_pipeline = infer_cfg.get('pipeline', cfg.data.val.get('pipeline'))

    object_file = infer_cfg.get('object_file', '')

    if infer_pipeline[0].type != 'ConvertSceneToPatch':
        raise RuntimeError(
            'First test pipeline should be `ConvertSceneToPatch`, '
            'and followed by pipelines such as `RandomStRetch and '
            '`CropInstance`')

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

            pred_result = model.process(data)
            pred_results.append(pred_result)

    pred_results = np.concatenate(pred_results)

    return pred_results
