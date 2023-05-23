# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

import numpy as np
from pygeococotools.geococo import GeoCOCO
from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.datasets.builder import DATASETS


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

            info['gt_label'] = np.array(
                self.coco_api.loadAnns(ann_ids)[0]['properties']
                ['category_id'],
                dtype=np.int64)
            data_infos.append(info)

        if len(data_infos) == 0:
            msg = 'Found no valid file in '
            if self.ann_file:
                msg += f'{self.ann_file}. '
            else:
                msg += f'{self.data_prefix}. '
            msg += 'Supported extensions are: ' + ', '.join(
                self.IMG_EXTENSIONS)
            raise RuntimeError(msg)

        return data_infos
