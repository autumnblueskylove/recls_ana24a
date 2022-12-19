import argparse
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config

from recls.utils import evaluate_per_class

INFER_DIR = 'inference'
PRED_PATH = os.path.join(INFER_DIR, 'result.pkl')
CONF_PATH = os.path.join(INFER_DIR, 'confusion_matrix.png')
CONFIG_PATH = 'model_config.py'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--artifact-path', default='checkpoint')
    parser.add_argument('--local-path', default='ckpt')
    args = parser.parse_args()
    return args


def download_artifacts():

    args = parse_args()
    os.makedirs(args.local_path, exist_ok=True)

    client = mlflow.tracking.MlflowClient()
    download_files = [
        PRED_PATH,
        CONFIG_PATH,
    ]
    for download_file in download_files:
        client.download_artifacts(
            args.run_id, os.path.join(args.artifact_path, download_file),
            args.local_path)


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show_counts=False,
                          title='Accuracy Confusion Matrix',
                          color_theme='viridis'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """

    if show_counts:
        counts_matrix = confusion_matrix.copy()
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)
            if show_counts:
                ax.text(
                    j,
                    i + 0.3,
                    '({})'.format(int(counts_matrix[i, j])),
                    ha='center',
                    va='center',
                    color='w',
                    size=5)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')


def main():

    args = parse_args()
    result_path = os.path.join(args.local_path, args.artifact_path, PRED_PATH)
    config_path = os.path.join(args.local_path, args.artifact_path,
                               CONFIG_PATH)
    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    cfg = Config.fromfile(config_path)
    categories = cfg.get('categories')

    if categories:
        categories = [i['id'] for i in categories]
        categories = list(set(categories))

    class_metrics, confusion_mat = evaluate_per_class(result, categories)
    os.makedirs(INFER_DIR, exist_ok=True)

    plot_confusion_matrix(
        confusion_mat,
        categories,
        os.path.join(args.local_path, args.artifact_path, CONF_PATH),
        show_counts=True)

    client = mlflow.tracking.MlflowClient()

    for metric in class_metrics:
        for k, v in metric.items():
            client.log_metric(args.run_id, k, v)

    client.log_artifact(
        args.run_id,
        os.path.join(args.local_path, args.artifact_path, CONF_PATH),
        os.path.join(args.artifact_path, INFER_DIR),
    )

    # Evaluation according to the sensor type is not required currently.
    # sensor_metrics = evaluate_per_sensor(result)
    # for metric in sensor_metrics:
    #     for k, v in metric.items():
    #         client.log_metric(args.run_id, k, v)


if __name__ == '__main__':
    download_artifacts()
    main()
