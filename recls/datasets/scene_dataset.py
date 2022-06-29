import copy
import csv
import os

from torch.utils.data import Dataset

from mmcls.datasets import DATASETS
from mmcls.datasets.pipelines import Compose


@DATASETS.register_module()
class SceneDataset(Dataset):
    """Dataset to inference a large image.

    The dataset loads scene image, crop the image to smaller patches, and
    applies specified transforms and finally returns a dict containing a patch
    and other information.

    Args:
        path (str | :obj:`Path`): Target path.
        pipeline (list[dict | callable]): Sequence of data transformations.
        stride (tuple[int]):
        crop_size (tuple[int]):
    """

    def __init__(
        self,
        path,
        pipeline,
        objects=None,
        object_file='',
    ):
        if not (os.path.exists(object_file) ^ isinstance(objects, list)):
            raise RuntimeError(
                'Either object file or objects should be exist!')
        elif os.path.exists(object_file):
            with open(object_file) as f:
                objects = list(csv.reader(f))
                self.objects = list([map(float, i) for i in objects])
        else:
            self.objects = objects

        from osgeo import gdal
        os.environ['JP2KAK_THREADS'] = '5'

        self.path = str(path)
        self.scene = gdal.Open(self.path)

        self.data_infos = self.load_annotations()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    @property
    def shape(self):
        return (self.scene.RasterYSize, self.scene.RasterXSize,
                self.scene.RasterCount)

    @property
    def rows(self):
        return max(self.shape[0] - self.crop_size[0] + self.stride[0] - 1,
                   0) // self.stride[0] + 1

    @property
    def cols(self):
        return max(self.shape[1] - self.crop_size[1] + self.stride[1] - 1,
                   0) // self.stride[1] + 1

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self):
        data_infos = []
        for object in self.objects:
            info = {'img_prefix': self.path}
            info['img_info'] = {
                'filename': self.path,
                'coordinate': object,
            }
            data_infos.append(info)

        return data_infos

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])

        return self.pipeline(results)
