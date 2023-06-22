import os
import os.path as osp
from pathlib import Path

from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import VISBACKENDS
from mmengine.visualization import MLflowVisBackend as _MLflowVisBackend
from mmengine.visualization.vis_backend import force_init_env


@VISBACKENDS.register_module(force=True)
class MLflowVisBackend(_MLflowVisBackend):
    """Modified MLflow visualization backend class based on mmengine.

    - Support to set existed run id of mlflow
    - Log artifact of "model_config.py" instead of config.py
    - Log artifact of "model_final.pth" and "###.log" when training get done.

    Args:
        run_id (str): Run id of mlflow to be set.
        **kwargs (dict): For parent class.
    """

    def __init__(self, run_id=None, **kwargs):
        """Class to log metrics and (optionally) a trained model to MLflow.

        It requires `MLflow`_ to be installed.
        """
        super(MLflowVisBackend, self).__init__(**kwargs)
        self.run_id = run_id

    def _init_env(self):
        """Setup env for MLflow."""
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow'
            )  # type: ignore
        self._mlflow = mlflow

        # when mlflow is imported, a default logger is created.
        # at this time, the default logger's stream is None
        # so the stream is reopened only when the stream is None
        # or the stream is closed
        logger = MMLogger.get_current_instance()
        for handler in logger.handlers:
            if handler.stream is None or handler.stream.closed:
                handler.stream = open(handler.baseFilename, 'a')

        if self.run_id:
            self._mlflow.start_run(run_id=self.run_id)
        else:
            self._exp_name = self._exp_name or 'Default'

            if self._mlflow.get_experiment_by_name(self._exp_name) is None:
                self._mlflow.create_experiment(self._exp_name)

            self._mlflow.set_experiment(self._exp_name)

            if self._run_name is not None:
                self._mlflow.set_tag('mlflow.runName', self._run_name)
            if self._tags is not None:
                self._mlflow.set_tags(self._tags)
            if self._params is not None:
                self._mlflow.log_params(self._params)

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to mlflow.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        if self._tracked_config_keys is None:
            self._mlflow.log_params(self._flatten(self.cfg))
        else:
            tracked_cfg = dict()
            for k in self._tracked_config_keys:
                tracked_cfg[k] = self.cfg[k]
            self._mlflow.log_params(self._flatten(tracked_cfg))
        self._mlflow.log_text(self.cfg.pretty_text,
                              'checkpoint/model_config.py')

    def close(self) -> None:
        """Close the mlflow."""
        if not hasattr(self, '_mlflow'):
            return

        # log artifact
        log_dir = osp.dirname(self._save_dir)
        log_path = osp.join(log_dir, f'{Path(log_dir).stem}.log')
        artifact_lists = ['model_final.pth', log_path]
        for artifact_list in artifact_lists:
            self._mlflow.log_artifact(
                os.path.join(self.cfg.work_dir, artifact_list),
                artifact_path='checkpoint')

        self._mlflow.end_run()
