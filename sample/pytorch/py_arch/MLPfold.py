from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

from sample.pytorch.py_arch.base import BaseModule
from sample.pytorch.py_arch.convnext import BottleNeckBlock
from sample.pytorch.py_arch.patchconvnet import PatchConvBlock
from sample.pytorch.py_arch.base import BaseCfg, Residual, ConvMixerLayer, BaseModule

@dataclass
class FoldNetCfg(BaseCfg):
    block: nn.Module = None
    image_size: int = 224

    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    fold_num: int = 1
    drop_rate: float = 0.
    expansion: int = 1
    expansion_factor_token: float = 0.5
    layer_scaler_init_value: float = 1e-6
    batch_size: int = 16

    num_layers: int = 8
    
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = None
    optimizer_method: str = "Adam"
    learning_rate_scheduler: str = "CosineAnnealing"
    warmup_epochs: int = None
    warmup_start_lr: float = None
    steps_per_epoch: int = None
    last_epoch: int = -1

    
class MixerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_patch,
                #  token_dim,
                #  channel_dim,
                 expansion_factor = 4,
                 expansion_factor_token = 0.5,
                 dropout=0.):
        super().__init__(
        # token_dim = dim * expansion_factor
        # token_mix
        nn.LayerNorm(dim),
        Rearrange('b n d -> b d n'),
        nn.Linear(num_patch, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expansion_factor, num_patch),
        nn.Dropout(dropout),
        Rearrange('b d n -> b n d'),
        # channel_mix
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * expansion_factor_token),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expansion_factor_token, dim),
        nn.Dropout(dropout),)


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
            return xs
        for i in range(self.fold_num - 1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs


class FoldNet(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)
        
        if cfg.block == ConvMixerLayer :
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
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
        elif cfg.block == BottleNeckBlock:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, in_features = cfg.hidden_dim, out_features = cfg.hidden_dim,
                                kernel_size = cfg.kernel_size, expansion = cfg.expansion, drop_p = cfg.drop_rate, layer_scaler_init_value = cfg.layer_scaler_init_value
                )
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
        
        elif cfg.block == MixerBlock:
            num_patch =  (cfg.image_size// cfg.patch_size) ** 2
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, num_patch, expansion_factor=cfg.expansion,
                          expansion_factor_token=cfg.expansion_factor_token, dropout=cfg.drop_rate
                          )
                          for _ in range(cfg.num_layers)
            ])

            self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, cfg.patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
            self.digup = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

        # self.embed = nn.Sequential(
        #     nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
        #     nn.GELU(),
        #     nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        # )

        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Flatten(),
        #     nn.Linear(cfg.hidden_dim, cfg.num_classes)
        # )

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

