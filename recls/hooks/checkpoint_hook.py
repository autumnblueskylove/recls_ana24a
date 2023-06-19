import os.path as osp
import shutil

from mmengine.dist import is_main_process
from mmengine.hooks import CheckpointHook as _CheckpointHook
from mmengine.registry import HOOKS


@HOOKS.register_module(force=True)
class CheckpointHook(_CheckpointHook):

    def _save_checkpoint(self, runner) -> None:
        super()._save_checkpoint(runner)

        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        # Open the file A and read the first line
        with open(save_file, 'r') as save_file_p:
            old_model_file_path = save_file_p.readline().strip()

        model_final_path = osp.join(self.out_dir, 'model_final.pth')
        shutil.copy2(old_model_file_path, model_final_path)
