import os.path as osp
import platform
import shutil

import mmcv
from mmcv.runner import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.epoch_based_runner import EpochBasedRunner as _EpochBasedRunner


@RUNNERS.register_module(force=True)
class EpochBasedRunner(_EpochBasedRunner):
    def save_checkpoint(
        self,
        out_dir,
        filename_tmpl="epoch_{}.pth",
        save_optimizer=True,
        meta=None,
        create_symlink=True,
    ):
        """Save the checkpoint.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, "model_final.pth")
            if platform.system() != "Windows":
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
