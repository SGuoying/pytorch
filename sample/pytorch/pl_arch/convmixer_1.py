from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl
# from sample.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.base import BaseCfg, Residual, BaseModule

from sample.pytorch.py_arch.bayes.core import log_bayesian_iteration


@dataclass
class ConvMixerCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10


class ConvMixer(BaseModule):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(cfg.hidden_dim)
                )),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
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


class SumConvMixer(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x1 = x
        for layer in self.layers:
            x1 = layer(x1)
            x = x + x1
        x = self.digup(x)
        return x


class BayesConvMixer(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 

        self.cfg = cfg

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)
        # log_prior = self.log_prior.repeat(batch_size, 1)

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

        # log_prior = torch.zeros(1, cfg.num_classes)
        # self.register_buffer('log_prior', log_prior) 
        self.log_prior = nn.Parameter(torch.zeros(1, cfg.num_classes))
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
    
    
class BayesConvMixer_1(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 

        self.cfg = cfg

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)
        # log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x) 
            # log_prior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_prior + logits
            log_prior = F.log_softmax(log_prior, dim=-1)
            log_prior = log_prior + math.log(self.cfg.num_classes)
        
        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss

        
class BayesConvMixer2_1(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 
        self.logits_bias = nn.Parameter(torch.zeros(1, cfg.num_classes))
        # self.cfg = cfg
        self.digup = None
        num_stages = 4
        self.digups = nn.ModuleList([
            nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
            ) for _ in range(num_stages)
        ])

        self.stage_depth = cfg.num_layers // num_stages if cfg.num_layers % num_stages == 0 else cfg.num_layers // num_stages + 1

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        for i,layer in enumerate(self.layers):
            x = layer(x)
            logits = self.digups[i // self.stage_depth](x) 
            # log_prior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_prior + logits
            log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
            log_prior = F.log_softmax(log_prior, dim=-1)
        
        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss