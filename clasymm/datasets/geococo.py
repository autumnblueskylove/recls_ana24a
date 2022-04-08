# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

import numpy as np
from clasymm.core.evaluation import macro_accuracy
from pygeococotools.geococo import GeoCOCO

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy


class ImageInfo:
    """class to  store image info, using slots will save memory than using
    dict."""

    __slots__ = ['path', 'gt_label']

    def __init__(self, path, gt_label):
        self.path = path
        self.gt_label = gt_label


@DATASETS.register_module()
class GeoCOCODataset(BaseDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, cantains 21k+ classes
    and 1.4B files. This class has improved the following points on the
    basis of the class ``ImageNet``, in order to save memory usage and time
    required :

        - Delete the samples attribute
        - using 'slots' create a Data_item tp replace dict
        - Modify setting ``info`` dict from function ``load_annotations`` to
          function ``prepare_data``
        - using int instead of np.array(..., np.int64)

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in ``mmcls.datasets.pipelines``
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
        multi_label (bool): use multi label or not.
        recursion_subdir(bool): whether to use sub-directory pictures, which
            are meet the conditions in the folder under category directory.
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        self.coco_api = GeoCOCO(os.path.join(data_prefix, ann_file))
        super(GeoCOCODataset, self).__init__(data_prefix, pipeline, classes,
                                             ann_file, test_mode)
        self.CLASSES = [i['name'] for i in self.coco_api.cats.values()]

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [self.data_infos[idx].gt_label]

    def load_annotations(self):
        """load dataset annotations."""

        data_infos = []
        for img_id in self.coco_api.imgs:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {
                'filename': self.coco_api.loadImgs(img_id)[0]['file_name']
            }
            ann_ids = self.coco_api.getAnnIds(imgIds=img_id)

            info['gt_label'] = np.array(self.coco_api.loadAnns(ann_ids)[0]
                                        ['properties']['category_id'],
                                        dtype=np.int64)
            data_infos.append(info)

        if len(data_infos) == 0:
            msg = 'Found no valid file in '
            msg += f'{self.ann_file}. ' if self.ann_file else f'{self.data_prefix}. '
            msg += 'Supported extensions are: ' + ', '.join(
                self.IMG_EXTENSIONS)
            raise RuntimeError(msg)

        return data_infos

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, (
            'dataset testing results should '
            'be of the same length as gt_labels.')

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if average_mode == 'macro':
                accuracy_metric = macro_accuracy
            else:
                accuracy_metric = accuracy

            if thrs is not None:
                acc = accuracy_metric(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy_metric(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'{average_mode}_accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {f'{average_mode}_accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(results,
                                    gt_labels,
                                    average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
