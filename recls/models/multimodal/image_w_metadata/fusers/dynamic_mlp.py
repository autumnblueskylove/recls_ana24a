# Modified from https://github.com/ylingfeng/DynamicMLP/blob/master/models/dynamic_mlp.py  # noqa: E501
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from recls.registry import MODELS


class Basic1d(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        layers = nn.Linear(in_channels, out_channels, bias)
        self.layers = nn.Sequential(layers, )
        if not bias:
            self.layers.add_module('ln', nn.LayerNorm(out_channels))
        self.layers.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.layers(x)
        return out


class Dynamic_MLP_A(BaseModule):

    def __init__(self, inplanes, planes, loc_planes, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.inplanes = inplanes
        self.planes = planes

        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        weight = self.get_weight(loc_fea)
        weight = weight.view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(img_fea.unsqueeze(1), weight).squeeze(1)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea


class Dynamic_MLP_B(BaseModule):

    def __init__(self, inplanes, planes, loc_planes, init_cfg=None):
        super().__init__(init_cfg)
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(loc_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class Dynamic_MLP_C(BaseModule):

    def __init__(self, inplanes, planes, loc_planes, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        cat_fea = torch.cat([img_fea, loc_fea], 1)

        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class RecursiveBlock(BaseModule):

    def __init__(self,
                 inplanes,
                 planes,
                 loc_planes,
                 mlp_type='c',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def init_weights(self):
        super(RecursiveBlock, self).init_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea


@MODELS.register_module()
class DynamicMLPFuser(BaseModule):

    def __init__(self,
                 inplanes=2048,
                 planes=256,
                 hidden=64,
                 num_layers=2,
                 mlp_type='c',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)
        conv2 = []
        if num_layers == 1:
            conv2.append(
                RecursiveBlock(
                    planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(
                RecursiveBlock(
                    planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(
                    RecursiveBlock(
                        hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(
                RecursiveBlock(
                    hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)
        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        super(DynamicMLPFuser, self).init_weights()
        # conv1
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias, std=1e-6)
        # conv3
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.normal_(self.conv3.bias, std=1e-6)

    def forward(self, img_feats: Tuple[torch.Tensor],
                metadata_feats: Union[torch.Tensor, Tuple[torch.Tensor]]):
        """Fusion module for image features and metadata features.

        Args:
            img_feats (Tuple[torch.Tensor]): The features extracted from the
                image encoder backbone.
            loc_feats (Union[torch.Tensor, Tuple[torch.Tensor]]): The features
                extracted from the metadata encoder.

        Returns:
            tuple | Tensor: The output of fused features
        """
        if isinstance(metadata_feats, torch.Tensor):
            metadata_feats = [metadata_feats] * len(img_feats)
        res = []
        for i, feat in enumerate(img_feats):
            identity = feat
            loc_feat = metadata_feats[i]

            feat = self.conv1(feat)

            for m in self.conv2:
                feat, loc_feat = m(feat, loc_feat)

            feat = self.conv3(feat)
            feat = self.norm3(feat)
            feat += identity

            res.append(feat)

        return res
