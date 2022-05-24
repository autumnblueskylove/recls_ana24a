from unittest.mock import patch

import clasymm  # noqa: F401
import numpy as np
from mmcv.utils import build_from_cfg

from mmcls.datasets.builder import PIPELINES


def get_toy_data():

    results = {
        'img_prefix': 'sample.tif',
        'img_info': {
            'filename': 'sample.tif',
            'coordinate': [1000, 1000, 20, 8, 0.0],
        },
        'gt_label': np.array(1)
    }

    return results


def test_random_rbox():

    results = get_toy_data()
    transforms = dict(
        type='RandomRBox',
        x_range=0.4,
        y_range=0.4,
        w_range=(0.8, 1.2),
        h_range=(0.8, 1.2),
        rad_range=(-0.785, 0.785),
    )
    transforms_module = build_from_cfg(transforms, PIPELINES)
    transforms_module(results)


@patch('osgeo.gdal.Open')
def test_convert_scene_to_patch(mock_gdal):

    mock_gdal.return_value.RasterXSize = 2048
    mock_gdal.return_value.RasterYSize = 2048
    mock_gdal.return_value.ReadAsArray.return_value = np.ones((3, 512, 512))

    results = get_toy_data()
    transforms = dict(
        type='ConvertSceneToPatch',
        patch_size=(512, 512),
    )
    transforms_module = build_from_cfg(transforms, PIPELINES)
    transforms_module(results)


def test_crop_instance():

    results = get_toy_data()
    results['img'] = np.ones((512, 512, 3))

    transforms = dict(
        type='CropInstance',
        expand_ratio=2.0,
    )
    transforms_module = build_from_cfg(transforms, PIPELINES)
    transforms_module(results)
