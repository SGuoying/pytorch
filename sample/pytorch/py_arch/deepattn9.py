"""
Based on deepattn7, add squeeze with a shift.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# from sunyata.pytorch.arch.base import BaseCfg, ConvMixerLayer, LayerScaler
# from sunyata.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.base import BaseCfg, ConvMixerLayer, LKALayer, LayerScaler, BaseModule
# from sample.pytorch_lightning.base import BaseModule


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
        self.shift = nn.Parameter(init_scale * torch.rand(hidden_dim))

    def forward(self, x):
        # x shape (batch_size, hidden_dim, height, weight)
        squeezed = self.squeeze(x)
        # squeezed shape (batch_size, hidden_dim)
        squeezed = squeezed + self.shift
        return squeezed
class SE(nn.Module):
    def __init__(self, hidden_dim: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = hidden_dim // squeeze_factor
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
			nn.Conv2d(hidden_dim, squeeze_c, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(squeeze_c , hidden_dim, 1),
			nn.Sigmoid())
        
    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

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
        squeeze_factor: int = 4
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.squeeze = Squeeze(hidden_dim, init_scale)
        self.squeeze_se = SE(hidden_dim, squeeze_factor)
        self.attn = Attn(num_heads, temperature)
        self.num_heads = num_heads
        self.query_idx = query_idx

    def forward(self, xs, all_squeezed):
        # xs shape (current_depth, batch_size, hidden_dim, height, width)
        # squeezed = self.squeeze(xs[-1]).unsqueeze(0)
        squeezed = self.squeeze_se(xs(-1)).unsqueeze(0)
        all_squeezed = torch.cat([all_squeezed, squeezed])
        # all_squeezed shape (current_depth, batch_size, hidden_dim)
        query = all_squeezed[self.query_idx,:,:]
        keys = all_squeezed
        attended = self.attn(query, keys)
        # attended shape (current_depth, batch_size, num_heads)
        xs = Rearrange('d b (heads head_dim) v w -> d b heads head_dim v w', heads=self.num_heads)(xs)
        x_new = torch.einsum('d b e h v w, d b e -> b e h v w', xs, attended)
        x_new = Rearrange('b heads head_dim v w -> b (heads head_dim) v w')(x_new)
        # x_new shape (batch_size, hidden_dim, height, width)
        return x_new, all_squeezed


class Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.attn_layer = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)
        self.block = ConvMixerLayer(hidden_dim, kernel_size)
        # self.block = LKALayer(hidden_dim, kernel_size)

    def forward(self, xs, all_squeezed):
        x_new, all_squeezed = self.attn_layer(xs, all_squeezed)
        # x_new shape (batch_size, hidden_dim, height, width)
        x_next = self.block(x_new)
        x_next = x_next.unsqueeze(0)
        return torch.cat((xs, x_next), dim=0), all_squeezed


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    num_heads: int = 1
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0. 
    query_idx: int = -1   
    temperature: float = 1.
    init_scale: float = 1.

    squeeze_factor: int = 4


class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            Layer(cfg.hidden_dim, cfg.num_heads, cfg.kernel_size, cfg.query_idx, cfg.temperature, cfg.init_scale)
            for _ in range(cfg.num_layers)
        ])

        self.final_attn = AttnLayer(cfg.hidden_dim, cfg.num_heads, cfg.query_idx, cfg.temperature, cfg.init_scale)

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),  # eps>6.1e-5 to avoid nan in half precision
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg
        
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        xs = x.unsqueeze(0)
        all_squeezed = torch.zeros(0, device=x.device)
        for layer in self.layers:
            xs, all_squeezed = layer(xs, all_squeezed)
        x, all_squeezed = self.final_attn(xs, all_squeezed)
        x = self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss    