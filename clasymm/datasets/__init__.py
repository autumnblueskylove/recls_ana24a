from mmcls.datasets import build_dataloader, build_dataset
from . import pipelines
from .dp import DataPlatformDataset
from .geococo import GeoCOCODataset
from .image_folder import ImageFolder
from .mongo import MongoDataset

__all__ = ['build_dataloader', 'build_dataset']
