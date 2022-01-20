# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch


def macro_accuracy_torch(pred, target, topk=(1,), thrs=0.0):
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(f"thrs should be a number or tuple, but got {type(thrs)}.")

    maxk = max(topk)
    num_classes = pred.size(1)

    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()

    res = []
    for k in topk:
        res_thr = []
        for thr in thrs:
            res_class = []
            for class_id in range(num_classes):
                masking = target == class_id

                pred_class_label = pred_label[:, masking]
                pred_class_score = pred_score[masking]
                class_target = target[masking]

                num = class_target.size(0)
                if num == 0:
                    continue

                correct = pred_class_label.eq(class_target.view(1, -1).expand_as(pred_class_label))

                # Only prediction values larger than thr are counted as correct
                _correct = correct & (pred_class_score.t() > thr)
                correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res_class.append((correct_k.mul_(100.0 / num)))
            res_thr.append(sum(res_class) / num_classes)
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def macro_accuracy(pred, target, topk=1, thrs=0.0):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.
    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), (
        f"The pred should be torch.Tensor or np.ndarray " f"instead of {type(pred)}."
    )
    assert isinstance(target, (torch.Tensor, np.ndarray)), (
        f"The target should be torch.Tensor or np.ndarray " f"instead of {type(target)}."
    )

    # torch version is faster in most situations.
    to_tensor = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x  # noqa: E731
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = macro_accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res
