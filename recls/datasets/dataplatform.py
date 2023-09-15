from typing import List

from mmpretrain.datasets.base_dataset import BaseDataset
from osgeo import gdal, osr

from recls.registry import DATASETS


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
class DataPlatformDataset(BaseDataset):
    """DataPlatform dataset using Dataset API. Dataset API:
    https://github.com/SIAnalytics/DP-CLIENT.

    Args:
        dataset_id (int): 'dsid' of dataset
        categories (List[dict]): Mapping class name to index
            example:
                [dict(id=0, name='class1'), dict(id=1, name='class2'), ...]
        pipeline (List[dict]): A list of dict, where each element
            represents an operation defined in :mod:`recls.dataset.transforms`
        host (str): DataPlatform URL with port
        user (str): user in DataPlatform
        password (str): password of user
        rename_class (List[str], optional): Remapping class names to specified
            names
        include_longlat (bool): whether to include a geolocation(long, lat) of
            an object.
        include_date (bool): whether to include date of an image.
        include_gsd (bool): whether to include GSD(x, y) of an image.
        **kwargs (dict): for BaseDataset
    """

    WGS84_CS = osr.SpatialReference()
    WGS84_CS.SetWellKnownGeogCS('WGS84')

    def __init__(self,
                 dataset_id: int,
                 categories: List[dict],
                 pipeline: List[dict],
                 host: str,
                 user: str,
                 password: str,
                 rename_class=None,
                 include_longlat: bool = False,
                 include_date: bool = False,
                 include_gsd: bool = False,
                 **kwargs: dict):
        if rename_class is None:
            rename_class = {}

        self.dataset_id = dataset_id
        self.rename_class = rename_class
        self.dp_client = get_connection(host, user, password)
        self.class_name_to_id = {cat['name']: cat['id'] for cat in categories}
        classes = list(self.class_name_to_id.keys())

        self.include_longlat = include_longlat
        self.include_date = include_date
        self.include_gsd = include_gsd

        super(DataPlatformDataset, self).__init__(
            '', pipeline=pipeline, classes=classes, **kwargs)

    def load_data_list(self):
        """Get image paths and labels from DataPlatform dataset."""
        dataset = self.dp_client.dataset.get(
            self.dataset_id, rename_class=self.rename_class)
        dataset = dataset['data'][0]['dataset']['scenes']

        data_list = list()
        for data in dataset:
            filename = data['filepath']
            if self.include_longlat:
                raster = gdal.Open(filename)
                geo_transform = raster.GetGeoTransform(can_return_null=True)
                coord_transform = osr.CoordinateTransformation(
                    osr.SpatialReference(raster.GetProjectionRef()),
                    self.WGS84_CS)

            for label in data['labels']:
                # temporarily avoid label noise
                if label['width'] > 1 and label['height'] > 1:
                    info = {
                        'img_path':
                        filename,
                        'rbox': [
                            label['x'], label['y'], label['width'],
                            label['height'], label['rotate_angle']
                        ],
                        'gt_label':
                        int(self.class_name_to_id[label['class_name']]),
                        'ilid':
                        label['ilid']
                    }
                    if self.include_longlat:
                        if geo_transform:
                            longlat = gdal.ApplyGeoTransform(
                                geo_transform, info['rbox'][0],
                                info['rbox'][1])
                            longlat = coord_transform.TransformPoint(
                                longlat[0], longlat[1])
                            info['longlat'] = longlat[:2]
                        else:
                            info['longlat'] = None
                    if self.include_date:
                        date = info['img_path'].split('_')[1]
                        info['date'] = date
                    if self.include_gsd:
                        xy_gsd = [data['x_gsd'], data['y_gsd']]
                        info['xy_gsd'] = xy_gsd

                    data_list.append(info)
        return data_list
