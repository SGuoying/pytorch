from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
import pytorch_lightning as pl
from sample.pytorch.py_arch.base import BaseCfg, Residual, ConvMixerLayer, BaseModule
# from sample.pytorch_lightning.base import BaseModule

from sample.pytorch.py_arch.convnext import BottleNeckBlock
from sample.pytorch.py_arch.patchconvnet import PatchConvBlock


@dataclass
class FoldNetCfg(BaseCfg):
    block: nn.Module = None
    num_heads: int = 8
    qkv_bias: bool = True
    qk_scale: float = None
    attn_drop_ratio: float=0.
    proj_drop_ratio: float=0.

    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    fold_num: int = 1
    drop_rate: float = 0.
    expansion: int = 4
    layer_scaler_init_value: float = 1e-6

    squeeze_factor: int = 4
@dataclass
class AttnCfg(BaseCfg):
    block: nn.Module = None
    num_heads: int = 8
    qkv_bias: bool = True
    qk_scale: float = None
    attn_drop_ratio: float=0.
    proj_drop_ratio: float=0.

    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    fold_num: int = 1
    # drop_rate: float = 0.
    expansion: int = 1
    layer_scaler_init_value: float = 1e-6

    squeeze_factor: int = 4

class ResConvMixer(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            Residual(nn.Sequential(
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            )
            ) for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn * u
class Atten(nn.Sequential):
    def __init__(self, dim, drop_rate):
        super().__init__(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(), 
            LKA(dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(dim, dim, 1),
            # nn.Dropout(drop_rate),
        )

class AttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches+1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @ :multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: ->[batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1,total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block2(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
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


class Block3(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0., squeeze_factor:int = 4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),)
        self.se = SE(hidden_dim, squeeze_factor)

    def forward(self, x):
        x = self.layer(x)
        x = self.se(x)
        return x 
            
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

class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, fold_num:int, Unit:nn.Module, *args, **kwargs):  # , hidden_dim: int, kernel_size: int, drop_rate:float=0.
        super(FoldBlock, self).__init__()
        self.fold_num = fold_num
        units = []
        for i in range(max(1, fold_num - 1)):
            units += [Unit(*args, **kwargs)]
        self.units = nn.ModuleList(units)
        
    def forward(self, *xs):
        xs = list(xs)
        if self.fold_num == 1:
            xs[0] = xs[0] + self.units[0](xs[0])
            # xs[0] = xs[0] * self.units[0](xs[0])
            return xs
        for i in range(self.fold_num - 1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs


class FoldNet(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)
        
        if cfg.block == ConvMixerLayer or cfg.block == Block3:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate, cfg.squeeze_factor)
                for _ in range(cfg.num_layers)
            ])
            
        elif cfg.block == BottleNeckBlock:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, in_features = cfg.hidden_dim, out_features = cfg.hidden_dim,
                                kernel_size = cfg.kernel_size, expansion = cfg.expansion, drop_p = cfg.drop_rate, layer_scaler_init_value = cfg.layer_scaler_init_value
                )
                for _ in range(cfg.num_layers)
            ])
        elif cfg.block == PatchConvBlock:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.drop_rate, cfg.layer_scaler_init_value)
                for _ in range(cfg.num_layers)
            ])
        elif cfg.block == Atten:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.drop_rate)
                for _ in range(cfg.num_layers)
            ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )
        # self.se = SE(cfg.hidden_dim, squeeze_factor=cfg.expansion)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        xs = [x for _ in range(self.cfg.fold_num)]
        for layer in self.layers:
            xs= layer(*xs)
        x = xs[-1]
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


class FoldNetRepeat(FoldNet):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim * cfg.fold_num, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        xs = x.repeat(1, self.cfg.fold_num, 1, 1)
        xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
        xs = torch.cat(xs, dim = 1)
        x = self.digup(xs)
        return x


class FoldNetRepeat2(FoldNet):
    def forward(self, x):
        x = self.embed(x)
        xs = x.repeat(1, self.cfg.fold_num, 1, 1)
        xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
        x = xs[-1]
        x = self.digup(x)
        return x