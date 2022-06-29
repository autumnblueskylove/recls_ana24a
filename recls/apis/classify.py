import os
from typing import List, Tuple

import numpy as np
import torch
from mmcv.parallel import DataContainer, MMDataParallel, collate

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose


class Classifier:

    def __init__(self, model_path: str, device='cuda:0'):
        """Base class for detector.

        Args:
            model_path (str): path containing the config and weight.
            device (str, optional): device to allocate the model.
                Defaults to 'cuda:0'.
        """
        super().__init__()
        self._model_root = model_path
        self.device = device

        self._setup()
        self._aux_setup()

    def _setup(self):
        """Setup the model."""
        self.model = self.load_model(self.config_path, self.weight_path)

    def _aux_setup(self):
        """Setup the test pipeline."""
        cfg = self.cfg.copy()

        # get the test_pipeline
        infer_cfg = cfg.get('scene_test_dataset', dict())
        test_pipeline = infer_cfg.get('pipeline', cfg.data.val.get('pipeline'))

        if test_pipeline[0].type != 'ConvertSceneToPatch':
            raise RuntimeError(
                'First test pipeline should be `ConvertSceneToPatch`, '
                'and followed by pipelines such as `RandomStRetch and '
                '`CropInstance`')

        self.test_pipeline = Compose(test_pipeline)
        self.is_cuda = next(self.model.parameters()).is_cuda

    def load_model(self, cfg_path: str, weight_path: str):
        """Load model from config and weight and initialize it.

        Args:
            cfg_path (str): configuration file path.
            weight_path (str): weight file path.
        Returns:
            torch.nn.Module: initialized detector model
        """
        model = init_model(cfg_path, weight_path)
        self.cfg = model.cfg
        model = MMDataParallel(model, [self.device])
        return model

    @property
    def config_path(self) -> str:
        return os.path.join(self._model_root, 'model_config.py')

    @property
    def weight_path(self) -> str:
        return os.path.join(self._model_root, 'model_final.pth')

    def preprocess(self, imgs) -> dict:
        """Preprocess the image.

        Args:
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
                Either image files or loaded images.
        Returns:
            List[DataContainer]: List of preprocessed images.
        """

        if isinstance(imgs, (list, tuple)):
            self.is_batch = True
        else:
            imgs = [imgs]
            self.is_batch = False

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = self.test_pipeline(data)
            datas.append(data)

        return datas

    def make_batch(self, imgs) -> DataContainer:
        """Make list of data to batch.

        Args:
            imgs (List[DataContainer]): List of images.
        Returns:
            data (dict of str: DataContainer): batched images.
        """
        data = collate(imgs, samples_per_gpu=len(imgs))
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]

        return data

    def process(self, data: DataContainer) -> List[List[np.ndarray]]:
        """Feed images to the detector.

        Args:
            data (DataContainer): input data.
        Returns:
            List[List[np.ndarray]]: [batch_size, num_classes, pred_rbbox]
                where pred_rbbox is [num_pred, 6] with
                (x_center, y_center, w, h, angle_radian, score).
        """

        with torch.no_grad():
            results = self.model(return_loss=False, **data)
        return results

    def postprocess(
        self,
        results: List[List[np.ndarray]],
    ) -> List[List[np.ndarray]]:
        return results

    def infer(
        self,
        imgs,
        *,
        batch_size=1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Infer the image.

        Args:
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
                Either image files or loaded images.
            batch_size (int, optional): Defaults to 1.
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                (polygons, classes, scores).
        """

        preprocessed_imgs = self.preprocess(imgs)
        pred_results = []

        start = 0
        while True:
            data = preprocessed_imgs[start:start + batch_size]
            data = self.make_batch(data)

            pred_results.extend(self.process(data))

            if start + batch_size >= len(preprocessed_imgs):
                break
            start += batch_size

        pred_results = self.postprocess(pred_results)

        return pred_results
