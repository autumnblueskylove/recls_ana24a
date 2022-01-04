import siavision
import torch.nn as nn
from mmcls.models import BACKBONES
from mmcls.utils import get_root_logger
from mmcv.runner import load_checkpoint
from siavision.models import (
    beit,
    dla,
    effnet,
    hrt,
    resnest,
    resnet,
    rexnetv1,
    swin_transformer,
    volo,
)

SIAVISION_ARCHITECTURE = [dla, resnet, resnest, rexnetv1, swin_transformer, effnet, beit, hrt, volo]


@BACKBONES.register_module()
class SiavisionModel(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7",
        stage=4,
        use_pretrained=True,
        out_index=None,
        **kwargs,
    ):
        super(SiavisionModel, self).__init__()

        attr = [hasattr(m, model_name) for m in SIAVISION_ARCHITECTURE]
        assert sum(attr), "model is not available"

        self.stage = stage
        self.out_index = out_index
        self.backbone = getattr(siavision.models, model_name)(pretrained=use_pretrained)

    def forward(self, x):
        if isinstance(self.out_index, list):
            return [
                feat
                for idx, feat in enumerate(self.backbone.forward_features(x))
                if idx in self.out_index
            ]
        outputs = self.backbone.forward_features(x)[-self.stage :]
        return tuple(outputs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError("pretrained must be a str or None")
