# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

from tqdm import tqdm

from mmpretrain.evaluation.metrics import SingleLabelMetric
from mmpretrain.registry import METRICS


@METRICS.register_module()
class PerSensorSingleLabelMetric(SingleLabelMetric):
    def __init__(self,
                 sensors: Sequence[str] = ('K3A', 'K3I', 'WV1', 'WV2', 'WV3'),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(thrs, items, average, num_classes, collect_device=collect_device, prefix=prefix)
        assert 'etc' not in sensors
        self.sensors = sensors

    def compute_metrics(self, results: List):
        def filter_by_key(keys, value):
            for k in keys:
                if k in value:
                    return True, k
            else:
                return False, None

        sensor_map = {sensor: list() for sensor in self.sensors}
        sensor_map.update({'etc': list()})

        for result in tqdm(results):
            filename = result['filename']
            include, key = filter_by_key(list(sensor_map.keys()), filename)
            if include:
                sensor_map[key].append(result)
            else:
                sensor_map['etc'].append(result)


        per_sensor_result_metrics = {}
        for sensor, results in sensor_map.items():
            if len(results) == 0:
                continue

            result_metrics = super().compute_metrics(results)
            for result_metric_name, result_metric in result_metrics.items():
                per_sensor_result_metrics[f'{sensor}/{result_metric_name}'] = result_metric

        return per_sensor_result_metrics
