from typing import Tuple

import numpy as np


def stretch_image(
    image: np.ndarray,
    new_max=255.0,
    min_percentile=0,
    max_percentile=100,
    clipped_min_val=10,
) -> np.ndarray:
    """특정 범위로 normalization을 진행합니다.

    @param image: 이미지
    @param new_max: 범위 최댓값
    @param min_percent: 범위 최솟값 %
    @param max_percent: 범위 최댓값 %
    @param dtype: 변환할 이미지 데이터 타입
    @return: 변환된 이미지
    """

    def _scale_range(min_val: int,
                     max_val: int,
                     to_range=255.0) -> Tuple[int, int]:
        """최소 범위에 해당하는 최솟값과 최댓값을 반환합니다.

        @param min_val: 범위를 축소시킬 값 중 최솟값
        @param max_val: 범위를 축소시킬 값 중 최댓값
        @param min_range: 축소시킬 범위
        @return: 최솟값과 최댓값
        """
        intensity_gap = max_val - min_val
        if intensity_gap < to_range:
            margin = (to_range - intensity_gap) / 2
            min_val -= margin
            max_val += margin

            if min_val < 0:
                max_val -= min_val
                min_val = 0
            if max_val > 2**16:
                min_val -= 2**16 - max_val
                max_val = 2**16 - 1
        return min_val, max_val

    for idx in range(image.shape[2]):
        band = image[:, :, idx]
        filtered_band = band[band > clipped_min_val]

        if filtered_band.any():
            min_val = np.percentile(filtered_band, min_percentile)
            max_val = np.percentile(filtered_band, max_percentile)
        else:
            min_val, max_val = 0, 255

        min_val, max_val = _scale_range(min_val, max_val)

        cvt_range = max_val - min_val
        band = (band - min_val) / cvt_range * new_max
        band = np.clip(band, 0, new_max)
        image[:, :, idx] = band

    return image
