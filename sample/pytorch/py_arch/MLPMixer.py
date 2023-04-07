from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from sample.pytorch.py_arch.base import BaseCfg, Residual, ConvMixerLayer, BaseModule

@ dataclass
class MLPcfg(BaseCfg):
    batch_size: int = 16
    image_size: int = 64
    hidden_dim: int = 256
    token_dim: int = 256
    channel_dim: int = 1024

    # in_channels: int = 3
    patch_size: int = 2
    num_classes: int = 200

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

    
   
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
    
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
    

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(BaseModule):

    def __init__(self, cfg:MLPcfg):
        super().__init__(cfg)

        assert cfg.image_size % cfg.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (cfg.image_size// cfg.patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, cfg.patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(cfg.num_layers):
            self.mixer_blocks.append(MixerBlock(cfg.hidden_dim, self.num_patch, cfg.token_dim, cfg.channel_dim))

        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):


        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss




if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]