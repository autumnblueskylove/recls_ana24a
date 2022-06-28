import copy
import math
import os

import mmcv
import numpy as np

from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
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


@PIPELINES.register_module()
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

        patch = src.ReadAsArray(int(crop_w_start), int(crop_h_start),
                                int(crop_w_end) - int(crop_w_start),
                                int(crop_h_end) - int(crop_h_start))

        if patch.ndim == 2:
            patch = np.array([patch for _ in range(3)])
        patch = patch.transpose((1, 2, 0))
        # TODO: temporary slicing sensor
        patch = patch[:, :, :3]

        insert_h = int(self.patch_size[0] - (crop_h_end - crop_h_start))
        insert_w = int(self.patch_size[1] - (crop_w_end - crop_w_start))
        h, w, c = patch.shape

        pad_img[insert_h:h + insert_h, insert_w:w + insert_w, :] = patch

        return pad_img

    def __call__(self, results):

        coordinate = results['img_info']['coordinate']
        filename = results['img_info']['filename']
        patch = self.make_patch(filename, coordinate)

        results['img'] = patch
        results['img_info']['coordinate'][0] = int(self.patch_size[1] / 2)
        results['img_info']['coordinate'][1] = int(self.patch_size[0] / 2)

        return results


@PIPELINES.register_module()
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
