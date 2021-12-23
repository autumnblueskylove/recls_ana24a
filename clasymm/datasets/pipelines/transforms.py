from mmcls.datasets.builder import PIPELINES
import random
import numpy as np
from PIL import Image, ImageFilter
from mmcls.datasets.pipelines.auto_augment import random_negative

@PIPELINES.register_module()
class GaussianNoise:
    """Apply Gaussian Noising to image. The bboxes, masks and
    segmentations are not modified.
    Args:
        mean (float): mean of the noise distribution.
        std (float): std of the noise distribution.
        prob (float): The probability for performing Gaussian Noising.
    """

    def __init__(self, mean=0.0, std=1.0, prob=0.5):
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]."
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
        repr_str += f"(mean={self.mean}, "
        repr_str += f"std={self.std}, "
        repr_str += f"prob={self.prob})"
        return repr_str

@PIPELINES.register_module()
class RandomGaussianBlur:
    """Customized RandomGaussianBlur Module

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

