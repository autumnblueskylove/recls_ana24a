import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule

from recls.registry import MODELS


@MODELS.register_module()
class PriorsNetEncoder(BaseModule):

    def __init__(self, num_inputs, embed_dims=256, init_cfg=None):
        super(PriorsNetEncoder, self).__init__(init_cfg=init_cfg)
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, embed_dims),
            nn.ReLU(inplace=True),
            FFN(embed_dims),
            FFN(embed_dims),
            FFN(embed_dims),
            FFN(embed_dims),
        )
        self.class_emb = nn.Linear(embed_dims, embed_dims, bias=False)

    def init_weights(self):
        super(PriorsNetEncoder, self).init_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, torch.Tensor):
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, return_feats=False, class_of_interest=None):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb  # [b, num_filts]
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return class_pred  # [b, num_class]

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[
                class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])
