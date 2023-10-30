from typing import Optional, Sequence, Union

import torch
from mmpretrain.models.utils.position_encoding import torch_meshgrid


def build_2d_gsd_sincos_position_embedding(
        patches_resolution: Union[int, Sequence[int]],
        embed_dims: int,
        relative_scale: float,
        temperature: Optional[int] = 10000.,
        with_cls_token: Optional[bool] = False,
        device: torch.device = 'cpu') -> torch.Tensor:
    """The function is to build position embedding for model to obtain the
    position information of the image patches.

    Args:
        patches_resolution (Union[int, Sequence[int]]): The resolution of each
            patch.
        embed_dims (int): The dimension of the embedding vector.
        relative_scale (float): The relative scale to the original scale.
        temperature (int, optional): The temperature parameter. Defaults to
            10000.
        cls_token (bool, optional): Whether to concatenate class token.
            Defaults to False.

    Returns:
        torch.Tensor: The position embedding vector.
    """

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w, grid_h = torch_meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    n = relative_scale.shape[0]

    # multiply relative scale
    grid_w = torch.einsum('hw,n->nhw', grid_w, relative_scale)
    grid_h = torch.einsum('hw,n->nhw', grid_h, relative_scale)

    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('nm,d->nmd', [grid_w.flatten(1), omega])
    out_h = torch.einsum('nm,d->nmd', [grid_h.flatten(1), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=2,
    )

    if with_cls_token:
        cls_token_pe = torch.zeros([n, 1, embed_dims],
                                   dtype=torch.float32,
                                   device=device)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb
