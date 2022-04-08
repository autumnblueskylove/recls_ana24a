import os

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import LoggerHook


@HOOKS.register_module(force=True)
class MlflowLoggerHook(LoggerHook):
    def __init__(
        self,
        exp_name=None,
        run_name=None,
        tags=None,
        log_model=True,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        run_id=None,
        by_epoch=True,
    ):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            exp_name (str, optional): Name of the experiment to be used.
                Default None.
                If not None, set the active experiment.
                If experiment does not exist, an experiment with provided name
                will be created.
            tags (dict of str: str, optional): Tags for the current run.
                Default None.
                If not None, set tags for the current run.
            log_model (bool, optional): Whether to log an MLflow artifact.
                Default True.
                If True, log runner.model as an MLflow artifact
                for the current run.
            interval (int): Logging interval (every k iterations).
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`.
            reset_flag (bool): Whether to clear the output buffer after logging
            by_epoch (bool): Whether EpochBasedRunner is used.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """
        super(MlflowLoggerHook, self).__init__(interval, ignore_last,
                                               reset_flag, by_epoch)
        self.import_mlflow()
        self.exp_name = exp_name
        self.run_name = run_name
        self.tags = tags
        self.log_model = log_model
        self.run_id = run_id

    def _has_tracking_uri(self):
        return 'file://' not in self.mlflow.get_tracking_uri()

    def import_mlflow(self):
        try:
            import mlflow
            import mlflow.pytorch as mlflow_pytorch
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_pytorch = mlflow_pytorch

    @master_only
    def before_run(self, runner):
        super(MlflowLoggerHook, self).before_run(runner)
        if self.mlflow.active_run() or not self._has_tracking_uri():
            return

        if self.run_id is not None:
            self.mlflow.start_run(run_id=self.run_id)
        else:
            if self.exp_name is not None:
                self.mlflow.set_experiment(self.exp_name)
            if self.tags is not None:
                self.mlflow.set_tags(self.tags)
            self.mlflow.start_run(run_name=self.exp_name
                                  if self.run_name is None else self.run_name)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags and self._has_tracking_uri():
            max_tries = 5
            for i in range(1, max_tries + 1):
                try:
                    self.mlflow.log_metrics(tags, step=self.get_iter(runner))
                    break
                except Exception as e:
                    print(e)
                    print(f'Retrying ... : {i}')

    @master_only
    def after_run(self, runner):
        if not bool(runner.work_dir) or not self._has_tracking_uri():
            return

        artifact_lists = ['model_config.py', 'model_final.pth']
        for artifact_list in artifact_lists:
            self.mlflow.log_artifact(os.path.join(runner.work_dir,
                                                  artifact_list),
                                     artifact_path='checkpoint')

        self.mlflow.end_run()
