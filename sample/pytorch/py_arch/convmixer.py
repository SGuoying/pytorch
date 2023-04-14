from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl
# from sample.pytorch_lightning.base import BaseModule
from sample.pytorch.py_arch.base import BaseCfg, ConvMixerLayer, Layer, Residual, BaseModule

from sample.pytorch.py_arch.bayes.core import log_bayesian_iteration
from sample.pytorch.py_arch.foldnet import LKA
from torch.distributions import Normal


@dataclass
class ConvMixerCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    squeeze_factor: int = 4
    drop_rate: float = 0.1

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

class AttnMixer(BaseModule):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1),
                    nn.GELU(),
                    # LKA(cfg.hidden_dim),
                    DilatedCV(cfg.hidden_dim),
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1),
                )),
                nn.Dropout(cfg.drop_rate),
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

@dataclass
class IsotropicCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0. 
   
class Isotropic(BaseModule):
    def __init__(self, cfg: IsotropicCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            # Layer(cfg.hidden_dim, cfg.drop_rate)
            for _ in range(cfg.num_layers)
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

        self.cfg = cfg
        
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.digup(x)
        return x
    
    def variational_inference(self, logits, target, num_samples):
        # B = logits.size(0)
        mean = self.forward(input)
        log_var = torch.randn_like(mean)
        # log_var = torch.zeros_like(mean)
        var = torch.exp(log_var)
        total_loss = 0
        for i in range(num_samples):
            eps = torch.randn_like(mean)
            sample = mean + eps * torch.sqrt(var)
            # 计算似然概率
            likelihood_probs = F.cross_entropy(logits, target)
            # 计算先验概率
            prior_probs = Normal(torch.zeros_like(mean), torch.ones_like(var))
            prior_log_probs = prior_probs.log_prob(sample).sum(-1)
            # 计算后验概率
            posterior_probs = Normal(mean, torch.sqrt(var))
            posterior_log_probs = posterior_probs.log_prob(sample).sum(-1)
             # 计算 KL 散度
            kl_divergence = (posterior_log_probs - prior_log_probs).mean()
            # 计算损失
            loss = -likelihood_probs + kl_divergence
            total_loss += loss
        return total_loss / num_samples

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        # loss = F.cross_entropy(logits, target)
        loss = self.variational_inference(logits, target, 10)
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
        
class NormConvMixer(ConvMixer):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 
        # self.log_prior = nn.Parameter(torch.zeros(1, cfg.num_classes))
        # self.cfg = cfg
        self.logits_layer_norm = nn.LayerNorm(cfg.num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x) 
            # log_prior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_prior + logits
            log_prior = self.logits_layer_norm(log_prior)
        
        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss