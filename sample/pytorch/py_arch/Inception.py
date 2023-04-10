import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import repeat
import pytorch_lightning as pl
# from sample.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.base import BaseCfg, Residual, BaseModule
from torchvision.ops.misc import SqueezeExcitation
from torchvision.ops import StochasticDepth
from sample.pytorch.py_arch.base import LayerScaler


@dataclass
class ConvMixerCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    squeeze_factor: int = 4
    drop_rate: float = 0.
    num_layers: int = 8
    # layer_scale_init: float = 1e-6

class ConvMixerBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, squeeze_factor, drop_rate,):
        super().__init__()
        self.layer1 = nn.Sequential(
            Residual(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
        )),
        nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            ) )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // squeeze_factor, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // squeeze_factor, hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate)
        )
        self.layer4 = nn.Sequential(
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            # nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.BatchNorm2d(hidden_dim) ,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            SqueezeExcitation(hidden_dim, hidden_dim // 4),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            # LayerScaler(hidden_dim, init_scale=layer_scale_init),
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(x)
        # layer = [layer1, layer2, layer3, layer4]
        # return torch.cat(layer, 1)
        layer = layer1 + layer2 + layer3 + layer4
        return layer
    

class IncNet(BaseModule):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        
        self.block = ConvMixerBlock(cfg.hidden_dim, cfg.kernel_size, cfg.squeeze_factor,
                                    cfg.drop_rate)
        
        self.blocks = nn.ModuleList([])

        # for _ in range(cfg.num_layers):
        #     self.blocks.append(ConvMixerBlock())

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
        x = self.block(x)
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
    
class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate,):
        super().__init__()
        self.layer1 = nn.Sequential(
            Residual(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
        )),
        nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            ) )
        self.layer2 = nn.Sequential(
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
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate)
        )
        self.layer4 = nn.Sequential(
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            # nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.BatchNorm2d(hidden_dim) ,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            SqueezeExcitation(hidden_dim, hidden_dim // 4),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            # LayerScaler(hidden_dim, init_scale=layer_scale_init),
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )

    def forward(self, x):
        layer1 = self.layer2(x)
        layer2 = self.layer2(x)
        layer3 = self.layer2(x)
        layer4 = self.layer2(x)
        layer = [layer1, layer2, layer3, layer4]
        return torch.cat(layer, 1)
    

class IncNet2(BaseModule):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        
        self.block = ConvMixerBlock(cfg.hidden_dim, cfg.kernel_size, cfg.squeeze_factor,
                                    cfg.drop_rate)
        
        self.blocks = nn.ModuleList([])

        for _ in range(cfg.num_layers):
            self.blocks.append(ConvBlock(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate))

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim * 4 , cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x = self.block(x)
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
