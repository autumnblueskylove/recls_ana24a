import os

import mlflow


def download_artifacts(run_id, artifact_path, dst_path):
    """Download artifacts from mlflow.

    Args:
        run_id (str): Run ID of mlflow.
        artifact_path (str): Artifact path to be downloaded
        dst_path (str): Destined path to be saved.

    Returns:
        None
    """
    os.makedirs(dst_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path)


def log_artifact(run_id, local_path, artifact_path):
    """Log(upload) artifact to mlflow.

    Args:
        run_id (str): Run ID of mlflow for logging.
        local_path (str): Local path to be logged.
        artifact_path (str): Artifact path of mlflow

    Returns:
        None
    """
    client = mlflow.tracking.MlflowClient()
    client.log_artifact(run_id, local_path, artifact_path)


def log_metrics(run_id, metrics):
    """Log metrics to mlflow.

    Args:
        run_id (str): Run ID of mlflow.
        metrics (Dict[str, float]): A dictionary of metrics for logging.

    Returns:
        None
    """
    client = mlflow.tracking.MlflowClient()
    for metric in metrics:
        for k, v in metric.items():
            client.log_metric(run_id, k, v)
