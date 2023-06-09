from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl
# from sample.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.base import AvgAttnPooling2d, BaseCfg, Residual, BaseModule

from sample.pytorch.py_arch.bayes.core import log_bayesian_iteration
trainer = pl.Trainer()

@dataclass
class ConvMixerCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    squeeze_factor: int = 4
    drop_rate: float = 0.
    eca_kernel_size: int = 3
    

class eca_layer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        self.attn_pool = AvgAttnPooling2d(dim=dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        #  (batch_size, channels, 1, 1)
        # y = self.avg_pool(x)
        y = self.attn_pool(x)
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        y = y.transpose(-1, -2).squeeze(-1)
        return y

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

class ConvMixer(BaseModule):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(cfg.hidden_dim),
                )),
                # SE(cfg.hidden_dim),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim), 
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


class BayesConvMixer(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 

        self.cfg = cfg

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x) 
            log_prior = log_bayesian_iteration(log_prior, logits)
        
        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
      
class BayesConvMixer2(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)
        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)

        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )
        self.digup = AvgAttnPooling2d(cfg.hidden_dim)
        # self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.cross_entropy(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
        
class NormConvMixer(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = layer(x)
            logits = logits + self.digup(x) 
            # log_prior = log_bayesian_iteration(log_prior, logits)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss