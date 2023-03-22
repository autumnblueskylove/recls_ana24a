from typing import List

import numpy as np

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS


def get_connection(host: str, user: str, password: str):
    """Connect to DataPlatform.

    Args:
        host (str): DataPlatform URL with port
        user (str): user in DataPlatform
        password (str): password of user

    Returns:
        connected client of DataPlatform
    """
    from dpc import DpClient

    dp_client = DpClient(host, user, password)
    return dp_client


@DATASETS.register_module()
class DataPlatformDatasetV2(BaseDataset):
    """DataPlatform dataset using Dataset API. Dataset API:
    https://github.com/SIAnalytics/DP-CLIENT.

    Args:
        dataset_id (int): 'dsid' of dataset
        categories (List[dict]): Mapping class name to index
            example:
                [dict(id=0, name='class1'), dict(id=1, name='class2'), ...]
        pipeline (List[dict]): A list of dict, where each element
            represents an operation defined in :mod:`recls.dataset.pipelines`
        host (str): DataPlatform URL with port
        user (str): user in DataPlatform
        password (str): password of user
        rename_class (List[str], optional): Remapping class names to specified
            names
        **kwargs (dict): for BaseDataset
    """

    def __init__(self,
                 dataset_id: int,
                 categories: List[dict],
                 pipeline: List[dict],
                 host: str,
                 user: str,
                 password: str,
                 rename_class=None,
                 **kwargs: dict):
        if rename_class is None:
            rename_class = {}

        self.dataset_id = dataset_id
        self.rename_class = rename_class
        self.dp_client = get_connection(host, user, password)
        self.class_name_to_id = {cat['name']: cat['id'] for cat in categories}
        classes = list(self.class_name_to_id.keys())

        super(DataPlatformDatasetV2, self).__init__(
            '', pipeline, classes=classes, **kwargs)

    def load_annotations(self):
        """Get image paths and labels from DataPlatform dataset."""
        dataset = self.dp_client.dataset.get(
            self.dataset_id, rename_class=self.rename_class)
        dataset = dataset['data'][0]['dataset']['scenes']

        data_infos = list()
        for data in dataset:
            filename = data['filepath']
            for label in data['labels']:
                if label['width'] > 1 and label[
                        'height'] > 1:  # temporarily avoid label noise
                    info = {
                        'img_info': {
                            'filename':
                            filename,
                            'coordinate': [
                                label['x'], label['y'], label['width'],
                                label['height'], label['rotate_angle']
                            ]
                        },
                        'gt_label':
                        np.array(
                            self.class_name_to_id[label['class_name']],
                            dtype=np.int64),
                        'label_uuid':
                        label['ilid']
                    }
                    data_infos.append(info)
        return data_infos
