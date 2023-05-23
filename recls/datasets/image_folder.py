# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from glob import glob
from typing import List

import numpy as np
from torch.utils.data import Dataset

from mmpretrain.datasets.builder import DATASETS
from mmcv.transforms import Compose


@DATASETS.register_module(force=True)
class ImageFolder(Dataset):

    CLASSES = None

    def __init__(self,
                 root_dir,
                 pipeline,
                 suffix='.jpg',
                 test_mode=False,
                 **kwargs):
        super(ImageFolder, self).__init__()

        self.root_dir = root_dir
        self.suffix = suffix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(self.root_dir)
        self.data_infos = self.load_annotations()

    @classmethod
    def get_classes(self, root_dir):
        classes = os.listdir(root_dir)
        classes = sorted(classes)
        return classes

    def load_annotations(self):

        data_infos = list()
        for cls in self.CLASSES:
            path = os.path.join(self.root_dir, cls)
            images = glob(os.path.join(path, '*' + self.suffix))
            idxes = [self.CLASSES.index(cls) for _ in images]
            for image, idx in zip(images, idxes):
                data_info = {'img_prefix': path}
                data_info['img_info'] = {'filename': image}
                data_info['gt_label'] = np.array(idx)
                data_infos.append(data_info)

        return data_infos

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.
        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
