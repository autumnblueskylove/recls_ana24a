# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmpretrain.models import SwinTransformer, VisionTransformer
from mmpretrain.models.backbones.revvit import (RevBackProp,
                                                RevVisionTransformer)

from recls.models.utils import build_2d_gsd_sincos_position_embedding
from recls.registry import MODELS


@MODELS.register_module()
class ScaleViT(VisionTransformer):

    def __init__(self, *args, **kwags) -> None:
        super().__init__(*args, **kwags)

        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def forward(self, x, relative_scale):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + build_2d_gsd_sincos_position_embedding(
            self.patch_resolution,
            self.pos_embed.shape[-1],
            relative_scale,
            cls_token=True)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)


@MODELS.register_module()
class ScaleRevViT(RevVisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # position embedding is not learnable during pretraining
        self.pos_embed.requires_grad = False

    def forward(self, x, relative_scale):
        x, patch_resolution = self.patch_embed(x)

        pos_embed = build_2d_gsd_sincos_position_embedding(
            self.patch_resolution,
            self.pos_embed.shape[-1],
            relative_scale,
            with_cls_token=True,
            device=x.device)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]
        x = self.drop_after_pos(x)

        x = torch.cat([x, x], dim=-1)

        # forward with different conditions
        if not self.training or self.no_custom_backward:
            # in eval/inference model
            executing_fn = RevVisionTransformer._forward_vanilla_bp
        else:
            # use custom backward when self.training=True.
            executing_fn = RevBackProp.apply

        x = executing_fn(x, self.layers, [])

        if self.final_norm:
            x = self.ln1(x)
        x = self.fusion_layer(x)

        res = (self._format_output(x, patch_resolution), )
        return res


@MODELS.register_module()
class ScaleSwinTransformer(SwinTransformer):

    def __init__(self, *args, use_abs_pos_embed=True, **kwargs):
        super().__init__(*args, use_abs_pos_embed=use_abs_pos_embed, **kwargs)
        self.absolute_pos_embed.requires_grad = False

    def forward(self, x, relative_scale):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            pos_embed = build_2d_gsd_sincos_position_embedding(
                self.patch_resolution,
                self.absolute_pos_embed.shape[-1],
                relative_scale,
                with_cls_token=False,
                device=x.device)
            # add pos embed w/o cls token
            x = x + pos_embed

        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)
