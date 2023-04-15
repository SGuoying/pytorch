from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from sample.pytorch.py_arch.base import BaseCfg, ConvMixerLayer, LayerScaler, BaseModule, Residual
# from sample.pytorch_lightning.base import BaseModule


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
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                                   nn.GELU(),
                                   nn.BatchNorm2d(dim))
        self.conv_spatial =  nn.Sequential(nn.Conv2d(dim, dim, 5, stride=1, padding="same", groups=dim, dilation=2),
                                           nn.GELU(),
                                           nn.BatchNorm2d(dim))
                                           
        self.conv1 = nn.Sequential( nn.Conv2d(dim, dim, 1),
                                    nn.GELU(),
                                    nn.BatchNorm2d(dim))

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn + u
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block2(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            # nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            # nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout(drop_rate),
            )
        
    def forward(self, x):
        skip = x
        x = self.layers(x)
        x = x + skip
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
   
@dataclass
class AttnCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 200

    drop_rate: float = 0.1 
    squeeze_factor: int = 4


class Attn(BaseModule):
    def __init__(self, cfg: AttnCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            nn.Sequential(
            Residual(nn.Sequential(
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, padding="same", groups=cfg.hidden_dim),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim)
            )),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, stride=1, padding="same", groups=cfg.hidden_dim, dilation=2),         
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1),
            nn.Dropout(cfg.drop_rate),
            nn.BatchNorm2d(cfg.hidden_dim),
            
            ) for _ in range(cfg.num_layers)
        ])

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
        # self.pooling = AttentionPooling(cfg.hidden_dim)
        # self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        # for layer in self.layers:
        #     x = layer(x)
        x = self.layers(x)
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
