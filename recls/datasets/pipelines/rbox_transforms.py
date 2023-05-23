import copy
import math
import os
from typing import List

import mmcv
import numpy as np

from mmpretrain.registry import TRANSFORMS


def read_as_array(scene,
                  xoff: int,
                  yoff: int,
                  xsize: int,
                  ysize: int,
                  channels: List[int] = None) -> np.ndarray:
    """Read array from gdal.Dataset with specified channels.

    If not channels, set default values [0, 1, 2]
    Args:
        scene (gdal.Dataset): raster of gdal.Dataset
        xoff (int): start x
        yoff (int): start y
        xsize (int): width
        ysize (int): height
        channels (List[int]): channels to read array
    """
    if channels is None:
        channels = [0, 1, 2]

    img = scene.ReadAsArray(xoff, yoff, xsize, ysize)
    if img.ndim == 3:
        img = np.array([img[ch] for ch in channels])
    else:  # to support a single band image
        img = np.array([img for _ in range(len(channels))])

    # change shape from ch, h, w to h, w, ch
    img = img.transpose((1, 2, 0))
    return img


@TRANSFORMS.register_module()
class CropInstance:

    def __init__(self, expand_ratio=1.0):
        assert isinstance(expand_ratio, float)

        self.expand_ratio = expand_ratio

    def rotate(self, img, rad, center):
        degree = -rad * (180.0 / math.pi)
        img = mmcv.imrotate(
            img,
            angle=degree,
            center=center,
        )
        return img

    def crop(self, img, x, y, w, h):

        xmin = int(x - self.expand_ratio * w / 2)
        xmax = int(x + self.expand_ratio * w / 2)
        ymin = int(y - self.expand_ratio * h / 2)
        ymax = int(y + self.expand_ratio * h / 2)

        img = img[ymin:ymax, xmin:xmax, ]
        return img

    def __call__(self, results):

        coordinate = results['img_info']['coordinate']
        img = results['img']
        img = self.rotate(img, coordinate[-1], (coordinate[0], coordinate[1]))
        img = self.crop(img, coordinate[0], coordinate[1], coordinate[2],
                        coordinate[3])

        results['img'] = img
        return results


@TRANSFORMS.register_module()
class CropInstanceInScene(CropInstance):
    """Load, crop and align instance(label such as rbox) in scene(image).
    Args:
        expand_ratio (float): Crop size is width * expand_ratio,
            height * expand_ratio
            defaults to 1.0
        rot_dir (str): Rotation direction of instance
            defaults to 'ccw'
    """

    def __init__(self, expand_ratio: float = 1.0, rot_dir: str = 'cw'):
        assert rot_dir in ['ccw', 'cw'],\
            f'rot_dir({rot_dir}) should be ccw or cw'

        self.rot_dir = rot_dir
        super(CropInstanceInScene, self).__init__(expand_ratio)

    def __call__(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline and to be used
                below.
                - results['image_info']['filename'] to open image
                - results['image_info']['coordinate'] to get label
        Return:
            dict: Added or update results to 'img'
        """
        from osgeo import gdal

        scene = gdal.Open(results['img_info']['filename'])
        rbox = results['img_info']['coordinate']
        crop_bbox = self.get_crop_bbox(rbox)

        # check and fix crop bbox that is in scene.
        valid_crop_bbox = [
            crop_bbox[0] if crop_bbox[0] > 0 else 0,
            crop_bbox[1] if crop_bbox[1] > 0 else 0, crop_bbox[2]
            if crop_bbox[2] < scene.RasterXSize else scene.RasterXSize,
            crop_bbox[3]
            if crop_bbox[3] < scene.RasterYSize else scene.RasterYSize
        ]

        img = read_as_array(scene, valid_crop_bbox[0], valid_crop_bbox[1],
                            valid_crop_bbox[2] - valid_crop_bbox[0],
                            valid_crop_bbox[3] - valid_crop_bbox[1])

        # if crop bbox is not in scene, fix image by zero padding
        if valid_crop_bbox != crop_bbox:
            img = np.pad(
                img, ((valid_crop_bbox[1] - crop_bbox[1],
                       crop_bbox[3] - valid_crop_bbox[3]),
                      (valid_crop_bbox[0] - crop_bbox[0],
                       crop_bbox[2] - valid_crop_bbox[2]), (0, 0)),
                mode='constant')

        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        rad = rbox[-1] if self.rot_dir == 'cw' else -rbox[-1]
        img = self.rotate(img, rad, center=(center_x, center_y))
        img = self.crop(img, center_x, center_y, rbox[2], rbox[3])

        results['img'] = img
        return results

    def get_crop_bbox(self, rbox: List[float]):
        """Get crop region for rotated instance.

        Args:
            rbox (List[float)]: cx, cy, width, height, radian
        """
        MARGIN_RATIO = 0.1  # TODO why? for rotation?

        x, y, w, h, rad = rbox
        cosa = math.cos(rad)
        sina = math.sin(rad)
        bbox_w = abs(cosa * w) + abs(sina * h)
        bbox_h = abs(sina * w) + abs(cosa * h)
        bbox_w *= (self.expand_ratio + MARGIN_RATIO)
        bbox_h *= (self.expand_ratio + MARGIN_RATIO)

        # for handling variable size by rotation
        crop_size = bbox_w if bbox_w > bbox_h else bbox_h

        xmin = round(x - crop_size / 2)
        ymin = round(y - crop_size / 2)
        xmax = round(x + crop_size / 2)
        ymax = round(y + crop_size / 2)

        return [xmin, ymin, xmax, ymax]


@TRANSFORMS.register_module()
class ConvertSceneToPatch:

    def __init__(self, patch_size=(512, 512)):
        os.environ['JP2KAK_THREADS'] = '5'

        assert isinstance(patch_size, tuple)
        assert len(patch_size) == 2
        self.patch_size = patch_size

    def convert_patch_size(self, patch_size, scene_size):
        patch_height = min(patch_size[0], scene_size[0])
        patch_width = min(patch_size[1], scene_size[1])
        return (patch_height, patch_width)

    def make_patch(self, filename, coordinate):
        from osgeo import gdal
        src = gdal.Open(filename)
        scene_width, scene_height = src.RasterXSize, src.RasterYSize

        pad_img = np.zeros((self.patch_size[0], self.patch_size[1], 3))

        x, y, _, _, _ = coordinate
        crop_w_start = max(0, x - self.patch_size[1] / 2)
        crop_w_end = min(scene_width, x + self.patch_size[1] / 2)
        crop_h_start = max(0, y - self.patch_size[0] / 2)
        crop_h_end = min(scene_height, y + self.patch_size[0] / 2)

        patch = read_as_array(src, int(crop_w_start), int(crop_h_start),
                              int(crop_w_end) - int(crop_w_start),
                              int(crop_h_end) - int(crop_h_start))

        insert_h = int(self.patch_size[0] - (crop_h_end - crop_h_start))
        insert_w = int(self.patch_size[1] - (crop_w_end - crop_w_start))
        h, w, c = patch.shape

        pad_img[insert_h:h + insert_h, insert_w:w + insert_w, :] = patch

        return pad_img

    def __call__(self, results):
        coordinate = results['img_info']['coordinate']
        scene_coordinate = copy.deepcopy(coordinate)
        results['img_info']['scene_coordinate'] = scene_coordinate
        filename = results['img_info']['filename']
        patch = self.make_patch(filename, coordinate)

        results['img'] = patch
        results['img_info']['coordinate'][0] = int(self.patch_size[1] / 2)
        results['img_info']['coordinate'][1] = int(self.patch_size[0] / 2)

        return results


@TRANSFORMS.register_module()
class RandomRBox:

    def __init__(
            self,
            x_range=0.4,
            y_range=0.4,
            w_range=(0.8, 1.2),
            h_range=(0.8, 1.2),
            rad_range=(-0.785, 0.785),
    ):
        assert isinstance(x_range, float)
        assert isinstance(y_range, float)
        assert isinstance(w_range, tuple)
        assert len(w_range) == 2
        assert isinstance(h_range, tuple)
        assert len(h_range) == 2
        assert isinstance(rad_range, tuple)
        assert len(rad_range) == 2

        self.x_range = x_range
        self.y_range = y_range
        self.w_range = w_range
        self.h_range = h_range
        self.rad_range = rad_range

    def jitter_point(self, value, value_range):

        rand = np.random.uniform()
        rand_value = rand * value * value_range
        if np.random.uniform() > 0.5:
            return rand_value
        else:
            return -rand_value

    def jitter_window(self, value, value_range):

        rand = np.random.uniform()
        rand = rand * (value_range[1] - value_range[0]) + value_range[0]

        return rand * value

    def jitter_rad(self, rad, rad_range):

        rand = np.random.uniform()
        rand = rand * (rad_range[1] - rad_range[0]) + rad_range[0]

        return rand + rad

    def __call__(self, results):

        coordinate = copy.deepcopy(results['img_info']['coordinate'])
        x, y, w, h, rad = coordinate
        new_x = x + self.jitter_point(w, self.x_range)
        new_y = y + self.jitter_point(h, self.y_range)
        new_w = self.jitter_window(w, self.w_range)
        new_h = self.jitter_window(h, self.h_range)
        new_rad = self.jitter_rad(rad, self.rad_range)

        new_coordinate = [int(new_x), int(new_y), new_w, new_h, new_rad]

        results['img_info']['coordinate'] = new_coordinate
        return results
