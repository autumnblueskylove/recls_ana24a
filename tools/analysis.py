import argparse
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import mmengine
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmpretrain.evaluation import ConfusionMatrix

from recls.utils import download_artifacts, log_artifact, log_metrics

INFER_DIR = 'inference'
PRED_PATH = os.path.join(INFER_DIR, 'results.pkl')
CONF_MAT_PATH = os.path.join(INFER_DIR, 'confusion_matrix.png')
CONFIG_PATH = 'checkpoint/model_config.py'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--tmpdir', default='.tmpdir')
    args = parser.parse_args()
    return args


def get_preds_w_gt_from_mlflow(args):
    download_artifacts(args.run_id, PRED_PATH, args.tmpdir)
    preds_w_gt = mmengine.load(osp.join(args.tmpdir, PRED_PATH))
    return preds_w_gt


def get_categories_from_mlflow(args):
    download_artifacts(args.run_id, CONFIG_PATH, args.tmpdir)
    cfg = Config.fromfile(osp.join(args.tmpdir, CONFIG_PATH))

    category_map = cfg.get('categories')
    if category_map:
        categories = [''] * len(category_map)
        for category in reversed(category_map):
            categories[category['id']] = category['name']
    else:
        categories = None

    return categories


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
        save_path (str|optional): If set, save the confusion matrix plot to the
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

    # draw confusion matrix value
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


def evaluate(preds_w_gt, categories=None):
    evaluator = Evaluator(ConfusionMatrix())
    metrics = evaluator.offline_evaluate(preds_w_gt, None)

    cm = metrics['confusion_matrix/result'].numpy()
    acc_per_cls = cm.diagonal() / cm.sum(axis=1) * 100.
    metrics = [{'evaluation/00.Top-1 Acc': acc_per_cls.mean()}]
    metrics += [{
        f'evaluation/{str(i + 1).zfill(2)}.{categories[i]}': acc_per_cls[i]
        for i in range(len(categories))
    }]

    return metrics, cm


def main():
    args = parse_args()

    # clear tmpdir
    shutil.rmtree(args.tmpdir, ignore_errors=True)

    # get preds with gt and categories
    preds_w_gt = get_preds_w_gt_from_mlflow(args)
    categories = get_categories_from_mlflow(args)

    # metrics
    metrics, cm = evaluate(preds_w_gt, categories)
    log_metrics(args.run_id, metrics)

    # plot confusion matrix
    cm_plot_path = os.path.join(args.tmpdir, CONF_MAT_PATH)
    plot_confusion_matrix(
        cm,
        categories,
        os.path.join(args.tmpdir, CONF_MAT_PATH),
        show_counts=True)
    log_artifact(args.run_id, cm_plot_path, INFER_DIR)


if __name__ == '__main__':
    main()
