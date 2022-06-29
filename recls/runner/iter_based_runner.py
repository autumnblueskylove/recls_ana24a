import os.path as osp
import platform
import shutil

import mmcv
from mmcv.runner import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.iter_based_runner import IterBasedRunner as _IterBasedRunner


@RUNNERS.register_module(force=True)
class IterBasedRunner(_IterBasedRunner):

    @property
    def max_epochs(self):
        """Return dummy data for ProfilerHook."""
        return -1

    def save_checkpoint(
        self,
        out_dir,
        filename_tmpl='iter_{}.pth',
        meta=None,
        save_optimizer=True,
        create_symlink=True,
    ):
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'model_final.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
