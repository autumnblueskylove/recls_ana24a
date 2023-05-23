import os.path as osp

try:
    from osgeo import gdal
except ImportError:
    IMPORT_GDAL = False
else:
    IMPORT_GDAL = True
import numpy as np

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromMSFile(object):
    """Load an image from multispectral file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        channel_order (list): Arguments to extract which channel what you want
    """

    def __init__(
        self,
        to_float32=False,
        color_type='color',
        channel_order=[0, 1, 2],
    ):
        assert IMPORT_GDAL, 'Install gdal to use `LoadImageFromMSFile`'
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        assert len(
            channel_order) == 3, 'The length of channel_order has to be 3'

    def __call__(self, results):

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = self.get_image(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def get_image(self, filename):
        assert osp.exists(filename), f'The {filename} cannot be founded'
        src = gdal.Open(filename)
        array = src.ReadAsArray()
        array = [array[i] for i in self.channel_order]
        array = np.stack(array, axis=2)
        return array

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'channel_order={self.channel_order})')
        return repr_str
