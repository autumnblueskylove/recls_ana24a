import datetime

import numpy as np
from mmcv import BaseTransform

from recls.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PreprocessMeta(BaseTransform):
    """Crop instance in image by coordinates.

    Args:
        expand_ratio (float): Expand ratio when cropping. In other word, the
            background is included.
    """

    def __init__(self,
                 include_longlat: bool = True,
                 include_date: bool = False,
                 include_gsd: bool = False,
                 use_object_size: bool = False,
                 use_xgsd: bool = False,
                 object_norm=50.,
                 num_metas: int = 2):
        self.include_longlat = include_longlat
        self.include_date = include_date
        self.include_gsd = include_gsd
        self.use_object_size = use_object_size
        self.num_metas = num_metas
        self.use_xgsd = use_xgsd
        self.object_norm = object_norm

    def transform(self, results):
        """Transform function to crop instance in an image.

         Args:
            results (dict): Result dict from loading pipline.

        Returns:
            dict: Cropped results, 'img' keys is updated in result dict.
        """
        longlat = results['longlat']
        date = results.get('date')
        if (longlat is not None) or (date is not None):
            if date is not None:
                date_time = datetime.datetime.strptime(date[:10], '%Y%m%d')
                date = self.get_scaled_date_ratio(date_time)

            long, lat = longlat

            long = float(long) / 90
            lat = float(lat) / 180
            meta_infos = []
            if self.include_longlat:
                meta_infos += [long, lat]
            if self.include_date:
                meta_infos += [date]
            if self.include_gsd:
                x, y = results.get('xy_gsd')
                if self.use_object_size:
                    coordinate = results['rbox']
                    # print(x, y, x*coordinate[2], y*coordinate[2], coordinate)
                    x = x * coordinate[2] / 50.
                    if self.use_xgsd:
                        y = x * coordinate[3] / self.object_norm
                    else:
                        y = y * coordinate[3] / self.object_norm
                meta_infos += [x, y]
            meta_infos = np.array(meta_infos, dtype=np.float32)
            meta_infos = self.encode_sinusoidal(meta_infos)
        else:
            meta_infos = np.zeros(self.num_metas, float)
        results['meta_infos'] = meta_infos
        # if self.include_gsd:
        #     results['xy_gsd'] = results.get('xy_gsd')[0]
        results['xy_gsd'] = results.get('xy_gsd')[0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def _is_leap_year(self, year):
        if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
            return False
        return True

    def get_scaled_date_ratio(self, date_time):
        r'''
        scale date to [-1,1]
        '''
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total_days = 365
        year = date_time.year
        month = date_time.month
        day = date_time.day
        if self._is_leap_year(year):
            days[1] += 1
            total_days += 1

        assert day <= days[month - 1]
        sum_days = sum(days[:month - 1]) + day
        assert sum_days > 0 and sum_days <= total_days

        return (sum_days / total_days) * 2 - 1

    def encode_sinusoidal(self, loc_time):
        # assumes inputs location and date features are in range -1 to 1
        # location is lon, lat
        feats = np.concatenate(
            (np.sin(np.pi * loc_time), np.cos(np.pi * loc_time)))
        return feats


@TRANSFORMS.register_module()
class PreprocessLongLat(BaseTransform):
    """Crop instance in image by coordinates.

    Args:
        expand_ratio (float): Expand ratio when cropping. In other word, the
            background is included.
    """

    def __init__(self, include_gsd: bool = False):
        self.include_gsd = include_gsd

    def transform(self, results):
        """Transform function to crop instance in an image.

         Args:
            results (dict): Result dict from loading pipline.

        Returns:
            dict: Cropped results, 'img' keys is updated in result dict.
        """
        longlat = results['longlat']

        assert longlat is not None

        long, lat = longlat
        lat = np.radians(lat)
        long = np.radians(long)

        x = np.cos(lat) * np.cos(long)
        y = np.cos(lat) * np.sin(long)
        z = np.sin(lat)

        meta_infos = np.array([x, y, z], dtype=np.float32)

        if self.include_gsd:
            gsd_array = np.array(results.get('xy_gsd'), dtype=np.float32)
            gsd_array = self.encode_sinusoidal(gsd_array)
            meta_infos = np.concatenate([meta_infos, gsd_array])

        results['meta_infos'] = meta_infos

        return results
