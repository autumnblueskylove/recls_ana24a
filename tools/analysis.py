import argparse
import os
import pickle

import mlflow
from clasymm.utils import evaluate_per_class, evaluate_per_sensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--artifact-path')
    parser.add_argument('--local-path', default='ckpt')
    args = parser.parse_args()
    return args


def download_artifacts():

    args = parse_args()
    os.makedirs(args.local_path, exist_ok=True)

    client = mlflow.tracking.MlflowClient()
    download_files = [
        'inference/result.pkl',
    ]
    for download_file in download_files:
        client.download_artifacts(
            args.run_id, os.path.join(args.artifact_path, download_file),
            args.local_path)


def main():

    args = parse_args()
    result_path = os.path.join(args.local_path, args.artifact_path,
                               'inference/result.pkl')
    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    sensor_metrics = evaluate_per_sensor(result)
    class_metrics = evaluate_per_class(result)

    client = mlflow.tracking.MlflowClient()
    for metric in sensor_metrics:
        for k, v in metric.items():
            client.log_metric(args.run_id, k, v)

    for metric in class_metrics:
        for k, v in metric.items():
            client.log_metric(args.run_id, k, v)


if __name__ == '__main__':
    download_artifacts()
    main()
