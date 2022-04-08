import os

import numpy as np
from clasymm.datasets.geococo import GeoCOCODataset
from tqdm import tqdm


class EvalMetrics(GeoCOCODataset):
    def __init__(
        self,
        infos,
        topk,
        add_key,
        metric=['accuracy', 'precision', 'recall', 'f1_score', 'support'],
    ):
        self.data_infos = infos
        self.logits = [info['logit'] for info in self.data_infos]
        self.topk = dict(topk=(1, topk))
        self.metric = metric
        self.add_key = add_key

    def get_metric(self):
        metric = self.evaluate(self.logits,
                               metric_options=self.topk,
                               metric=self.metric)
        add_metric = {
            os.path.join(self.add_key, k): v
            for k, v in metric.items()
        }
        return add_metric


def get_topk(num_classes, default_topk=3):
    return default_topk if num_classes > default_topk else num_classes


def filter_by_key(keys, value):
    for k in keys:
        if k in value:
            return True, k
    else:
        return False, None


def evaluate_per_class(results):

    logits = np.array([result['result'] for result in results])
    assert len(logits.shape) == 2
    num_classes = logits.shape[1]
    topk = get_topk(num_classes)

    class_map = {str(i): list() for i in range(num_classes)}

    for result in tqdm(results):
        logit = result['result']
        label = result['label']
        include, key = filter_by_key(list(class_map.keys()), str(label))
        if include:
            class_map[key].append(dict(
                gt_label=label,
                logit=logit,
            ))

    metrics = list()
    for key in class_map.keys():
        if len(class_map[key]) == 0:
            continue
        metric = EvalMetrics(
            infos=class_map[key],
            topk=topk,
            add_key=os.path.join('class', key),
        ).get_metric()
        metrics.append(metric)
    return metrics


def evaluate_per_sensor(results):

    logits = np.array([result['result'] for result in results])
    assert len(logits.shape) == 2
    num_classes = logits.shape[1]
    topk = get_topk(num_classes)

    sensor_map = {
        'K3A': list(),
        'K3I': list(),
        'WV1': list(),
        'WV2': list(),
        'WV3': list(),
    }

    etc_map = {
        'etc': list(),
    }

    for result in tqdm(results):
        filename = result['filename']
        logit = result['result']
        label = result['label']
        include, key = filter_by_key(list(sensor_map.keys()), filename)
        if include:
            sensor_map[key].append(dict(
                gt_label=label,
                logit=logit,
            ))
        else:
            etc_map['etc'].append(dict(
                gt_label=label,
                logit=logit,
            ))

    metrics = list()
    for key in sensor_map.keys():
        if len(sensor_map[key]) == 0:
            continue
        metric = EvalMetrics(
            infos=sensor_map[key],
            topk=topk,
            add_key=os.path.join('sensor', key),
        ).get_metric()
        metrics.append(metric)

    for key in etc_map.keys():
        if len(etc_map[key]) == 0:
            continue
        metric = EvalMetrics(
            infos=etc_map[key],
            topk=topk,
            add_key=os.path.join('sensor', key),
        ).get_metric()
        metrics.append(metric)

    return metrics
