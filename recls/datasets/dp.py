import csv
import os

import numpy as np

from mmpretrain.datasets.builder import DATASETS
from mmcv.transforms import Compose
from recls.datasets.api_wrappers import DataPlatformReader
from .mongo import MongoDataset


@DATASETS.register_module()
class DataPlatformDataset(MongoDataset):

    def __init__(self,
                 pipeline,
                 object_list,
                 categories,
                 classes=None,
                 test_mode=False,
                 host=None,
                 dbname=None,
                 port=None,
                 user=None,
                 password=None):
        assert isinstance(object_list, list) or os.path.isfile(object_list)

        self.pipeline = Compose(pipeline)
        self.dbname = dbname
        self.categories = categories
        if isinstance(object_list, str) and os.path.isfile(object_list):
            with open(object_list) as f:
                objects = list(csv.reader(f))
                object_list = list([map(float, i) for i in objects])

        self.object_list = object_list
        self.CLASSES = self.get_classes(classes)
        self.test_mode = test_mode
        self.cursor = self.get_connection(host, port, user, password, dbname)
        self.data_infos = self.load_annotations()

    def get_connection(self, host, port, user, password, dbname):
        import psycopg2
        db = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port)
        cursor = db.cursor()
        return cursor

    def load_annotations(self):

        data_infos = list()
        dataset = DataPlatformReader(
            cursor=self.cursor,
            categories=self.categories,
            object_list=self.object_list)

        for data in dataset.dataset:
            info = {'img_prefix': data['scene_path']}
            info['img_info'] = {
                'filename':
                data['scene_path'],
                'coordinate':
                [data['x'], data['y'], data['w'], data['h'], data['rad']],
            }
            info['gt_label'] = np.array(data['category_id'], dtype=np.int64)

            data_infos.append(info)

        return data_infos
