# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure.pytorch.py_arch.base import LayerScaler, LayerNorm2d
from torchvision.ops.misc import SqueezeExcitation
from torchvision.ops import StochasticDepth
# %%
class PatchConvBlock(nn.Module):
    def __init__(self, 
                hidden_dim:int, 
                drop_rate:float=0.3, 
                layer_scale_init:float=1e-6,
                norm_type='bn'):
        super().__init__()
        self.layers = nn.Sequential(
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            # nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.BatchNorm2d(hidden_dim) if norm_type == 'bn' else LayerNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            SqueezeExcitation(hidden_dim, hidden_dim // 4),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            LayerScaler(hidden_dim, init_scale=layer_scale_init),
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )

    def forward(self, x):
        return self.layers(x)