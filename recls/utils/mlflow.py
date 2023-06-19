import os

import mlflow


def download_artifacts(run_id, artifact_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path)


def log_artifact(run_id, local_path, artifact_path):
    client = mlflow.tracking.MlflowClient()
    client.log_artifact(run_id, local_path, artifact_path)


def log_metrics(run_id, metrics):
    client = mlflow.tracking.MlflowClient()
    for metric in metrics:
        for k, v in metric.items():
            client.log_metric(run_id, k, v)
