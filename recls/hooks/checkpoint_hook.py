import os.path as osp
import shutil

from mmengine.hooks import CheckpointHook as _CheckpointHook
from mmengine.registry import HOOKS


@HOOKS.register_module(force=True)
class CheckpointHook(_CheckpointHook):
    """Support to save checkpoint whose name is 'model_final.pth'."""

    def after_train(self, runner) -> None:
        super().after_train(runner)

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        # Open the file A and read the first line
        with open(save_file, 'r') as save_file_p:
            old_model_file_path = save_file_p.readline().strip()

        model_final_path = osp.join(self.out_dir, 'model_final.pth')
        shutil.copy2(old_model_file_path, model_final_path)
