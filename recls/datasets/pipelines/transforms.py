import numpy as np

from mmpretrain.registry import TRANSFORMS
from .transform_utils import stretch_image


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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


@TRANSFORMS.register_module()
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
