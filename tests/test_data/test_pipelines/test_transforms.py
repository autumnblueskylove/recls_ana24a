import clasymm  # noqa: F401
import numpy as np
from mmcv.utils import build_from_cfg

from mmcls.datasets.builder import PIPELINES


def test_random_stretch():

    results = {'img': np.ones((512, 512, 3))}

    transforms = dict(
        type='RandomStretch',
        min_percentile_range=(0.0, 3.0),
        max_percentile_range=(97.0, 100.0),
        new_max=255.0,
    )
    transforms_module = build_from_cfg(transforms, PIPELINES)
    transforms_module(results)
