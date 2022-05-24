import numpy as np
from clasymm.datasets.api_wrappers import MongoGeoCOCO

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose
from .geococo import GeoCOCODataset


@DATASETS.register_module()
class MongoDataset(GeoCOCODataset):
    def __init__(self,
                 pipeline,
                 dbname,
                 scene_list,
                 categories,
                 classes=None,
                 test_mode=False,
                 url='dev-cluster.sia-service.kr',
                 port=32000,
                 user='readonly-user',
                 password='readonly-user'):

        self.pipeline = Compose(pipeline)
        self.dbname = dbname
        self.categories = categories
        self.scene_list = scene_list
        self.CLASSES = self.get_classes(classes)
        self.test_mode = test_mode
        self.client = self.get_connection(url, port, user, password)
        self.data_infos = self.load_annotations()

    def get_connection(self, url, port, user, password):
        from pymongo import MongoClient
        connection_query = f'mongodb://{user}:{password}@{url}:{port}'
        client = MongoClient(connection_query)
        return client

    def load_annotations(self):

        data_infos = list()
        dataset = MongoGeoCOCO(client=self.client,
                               dbname=self.dbname,
                               categories=self.categories,
                               scene_list=self.scene_list)

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
