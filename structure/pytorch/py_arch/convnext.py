import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import pytorch_lightning as pl

from torchvision.ops import stochastic_depth

from structure.pytorch.py_arch.bayes.core import log_bayesian_iteration 
from structure.pytorch.py_arch.base import LayerScaler

class ConvNextForImageClassification(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
        num_classes: int = 200,
        learning_rate: float = 4e-3
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)
        self.encoder = ConvNextEncoder(stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)
        self.heads = nn.ModuleList(
            ClassificationHead(out_features, num_classes)
            for out_features in widths
        )
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        log_prior = torch.zeros(batch_size, self.num_classes, device=x.device)
        x = self.stem(x)
        for stage, head in zip(self.encoder.stages, self.heads):
            x = stage(x)
            # for i, block in enumerate(stage):
            #     x = block(x)
            #     if i > 0:
            logits = self.head(x)
            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def training_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("train_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("val_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 200):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes)
        )


class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNextStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNextStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        ),


class ConvNextStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size, padding="same", bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(out_features, layer_scaler_init_value)
        self.drop_p = drop_p

    def forward(self, x: Tensor) -> Tensor:
        # res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = stochastic_depth(x, self.drop_p, mode="row")
        # x = x + res
        return x