import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from sample.pytorch.py_arch.base import ConvMixerLayer, LayerScaler

class Squeeze(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LayerScaler(hidden_dim, init_scale),
        )

    def forward(self, x):
        # x shape (batch_size, hidden_dim, height, weight)
        squeezed = self.squeeze(x)
        return squeezed


class Attn(nn.Module):
    def __init__(
        self,
        num_heads: int,
        temperature: float = 1.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature

    def forward(self, query, keys):
        # query: (batch_size, hidden_dim) -> (batch_size heads head_dim)
        query = Rearrange('b (heads head_dim) -> b heads head_dim', heads=self.num_heads)(query)
        # keys: (current_depth, batch_size, hidden_dim) -> (current_depth, batch_size, heads, head_dim)
        keys = Rearrange('d b (heads head_dim) -> d b heads head_dim', heads=self.num_heads)(keys)

        attn = torch.einsum('b e h, d b e h -> d b e', query, keys)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)
        return attn


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.squeeze = Squeeze(hidden_dim, init_scale)
        self.attn = Attn(num_heads, temperature)
        self.num_heads = num_heads
        self.query_idx = query_idx

    def forward(self, xs, all_squeezed):
        # xs shape (current_depth, batch_size, hidden_dim, height, width)
        squeezed = self.squeeze(xs[-1]).unsqueeze(0)
        all_squeezed = torch.cat([all_squeezed, squeezed])
        # all_squeezed shape (current_depth, batch_size, hidden_dim)
        query = all_squeezed[self.query_idx,:,:]
        keys = all_squeezed
        attended = self.attn(query, keys) # * xs.shape[0]
        # attended shape (current_depth, batch_size, num_heads)
        xs = Rearrange('d b (heads head_dim) v w -> d b heads head_dim v w', heads=self.num_heads)(xs)
        x_new = torch.einsum('d b e h v w, d b e -> b e h v w', xs, attended)
        x_new = Rearrange('b heads head_dim v w -> b (heads head_dim) v w')(x_new)
        # x_new shape (batch_size, hidden_dim, height, width)
        return x_new, all_squeezed
    

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x
    

class Block2(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            )
        
    def forward(self, x):
        skip = x
        x = self.layers(x)
        x = x + skip
        return x
            
        