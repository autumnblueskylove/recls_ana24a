import random

import mmcv
import numpy as np
from PIL import Image, ImageFilter

from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines.auto_augment import Rotate, random_negative
from .transform_utils import stretch_image


@PIPELINES.register_module()
class RandomRotate(Rotate):
    """Rotate images from range of argument(-angle, angle).

    Args:
        angle (float): The angle used for rotate. Positive values stand for
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If None, the center of the image will be used.
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing Rotate therefore should be
            in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the angle
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        angle = np.random.uniform(self.angle)
        angle = random_negative(angle, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_rotated = mmcv.imrotate(
                img,
                angle,
                center=self.center,
                scale=self.scale,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_rotated.astype(img.dtype)
        return results


@PIPELINES.register_module()
class RandomStretch:

    def __init__(self,
                 min_percentile_range=(0.0, 3.0),
                 max_percentile_range=(97.0, 100.0),
                 new_max=255.0):

        assert isinstance(min_percentile_range, tuple)
        assert len(min_percentile_range) == 2
        assert isinstance(max_percentile_range, tuple)
        assert len(max_percentile_range) == 2
        assert isinstance(new_max, float)

        self.min_percentile_range = min_percentile_range
        self.max_percentile_range = max_percentile_range
        self.new_max = new_max

    def __call__(self, results):
        min_percentile = np.random.uniform(self.min_percentile_range[0],
                                           self.min_percentile_range[1])
        max_percentile = np.random.uniform(self.max_percentile_range[0],
                                           self.max_percentile_range[1])
        for key in results.get('img_fields', ['img']):
            results[key] = stretch_image(
                results[key],
                new_max=self.new_max,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
            ).astype('uint8')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_percentile_range={self.min_percentile_range}, '
        repr_str += f'max_percentile_range={self.max_percentile_range}, '
        repr_str += f'new_max={self.new_max})'
        return repr_str


@PIPELINES.register_module()
class GaussianNoise:
    """Apply Gaussian Noising to image.

    The bboxes, masks and
    segmentations are not modified.
    Args:
        mean (float): mean of the noise distribution.
        std (float): std of the noise distribution.
        prob (float): The probability for performing Gaussian Noising.
    """

    def __init__(self, mean=0.0, std=1.0, prob=0.5):
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1].'
        self.mean = mean
        self.std = std
        self.prob = prob

    def _add_noise_to_img(self, img, mean, std):
        """Add the noise to image."""
        noise = np.random.normal(mean, std)
        noised_img = img + noise

        return noised_img

    def __call__(self, results):
        """Call function for Gaussian Noising.

        Args:
            results (dict): Results dict from loading pipeline.
        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self._add_noise_to_img(img, self.mean, self.std)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, '
        repr_str += f'std={self.std}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class RandomGaussianBlur:
    """Customized RandomGaussianBlur Module.

    Args:
        p (float): probability of the image being blurred. Default value is 0.5
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        """
        Args:
            img (PIL Image): Image to be blurred.

        Returns:
            PIL Image: Randomly blurred image.
        """
        if np.random.rand() > self.prob:
            return results

        for key in results.get('img_fields', ['img']):
            img = Image.fromarray(results[key])
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            results[key] = np.asarray(img)

        return results


@PIPELINES.register_module()
class Identity:
    """Placeholder pipeline."""

    def __call__(self, results):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Output image.
        """

        return results
