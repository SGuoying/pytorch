from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
import pytorch_lightning as pl
from sample.pytorch.py_arch.base import BaseCfg, Residual, ConvMixerLayer, BaseModule
# from sample.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.bayes.core import log_bayesian_iteration

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
        
        if cfg.block == ConvMixerLayer or cfg.block == Block2:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate )   #)
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
       
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg
        self.log_prior = nn.Parameter(torch.zeros(1, cfg.num_classes))

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)
        
        x = self.embed(x)
        xs = [x for _ in range(self.cfg.fold_num)]
        for layer in self.layers:
            xs= layer(*xs)
            x = xs[-1]
            logits = self.digup(x)
            log_prior = log_bayesian_iteration(log_prior, logits)
        return log_prior

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
        self.log_prior = nn.Parameter(torch.zeros(1, cfg.num_classes))

    # def forward(self, x):
    #     x = self.embed(x)
    #     xs = x.repeat(1, self.cfg.fold_num, 1, 1)
    #     xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
    #     for layer in self.layers:
    #         xs = layer(*xs)
    #     xs = torch.cat(xs, dim = 1)
    #     x = self.digup(xs)
    #     return x
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        xs = x.repeat(1, self.cfg.fold_num, 1, 1)
        xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
            xs = torch.cat(xs, dim = 1)
            logits = self.digup(xs)
            log_prior = log_bayesian_iteration(log_prior, logits)
        return log_prior


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
    
