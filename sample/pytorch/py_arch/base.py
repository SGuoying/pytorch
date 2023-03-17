import math
from typing import Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.ops import StochasticDepth
import pytorch_lightning as pl


class RevSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
    ):
        # self.params = params
        # self.lr = lr
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Callable=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data * group["lr"])

        return loss

@dataclass
class BaseCfg:
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



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def set_requires_grad(model:nn.Module, val: bool):
    for p in model.parameters():
        p.requires_grad = val

            
class LayerScaler(nn.Module):
    def __init__(self, dim: int, init_scale: float):
        super().__init__()
        self.gamma = nn.Parameter(init_scale * torch.ones(dim))

    def forward(self, x):
        return self.gamma[None,...] * x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvMixerLayer(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate)
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )


class BaseModule(pl.LightningModule):
    def __init__(self, cfg:BaseCfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "RevSGD":
            optimizer = RevSGD(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr, last_epoch=self.cfg.last_epoch)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]   


class BYOL_EMA(pl.Callback):
    """
    Exponential Moving Average of BYOL 
    """
    def __init__(
        self, 
        student_name: str,
        teacher_name: str,
        initial_tau: float=0.999, 
        do_tau_update: bool=True):
        super().__init__()
        self.student_name = student_name
        self.teacher_name = teacher_name
        self.initial_tau = initial_tau
        self.current_tau = initial_tau
        self.do_tau_update = do_tau_update

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        student = getattr(pl_module, self.student_name) # pl_module.online_encoder
        teacher = getattr(pl_module, self.teacher_name) # pl_module.target_encoder

        self.update_weights(student, teacher)

        if self.do_tau_update:
            self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, student: nn.Module, teacher: nn.Module):
        for student_params, teacher_params in zip(student.parameters(), teacher.parameters()):
            teacher_params.data = self.current_tau * teacher_params.data + (1 - self.current_tau) * student_params