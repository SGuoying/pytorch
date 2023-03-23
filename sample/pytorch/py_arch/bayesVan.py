import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from einops import repeat
from timm .models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import List, Callable, Optional


@dataclass
class VanCfg:
    image_size: int = 224
    in_chans: int = 3
    num_classes: int = 1000
    embed_dims = [64, 128, 256, 512]
    mlp_ratio = [4, 4, 4, 4]
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm
    depths = [3, 4, 6, 3]
    num_stages: int = 4
    flag: bool = False
    batch_size = 128

    num_epochs: int = 10
    learning_rate: float = 1e-3
    optimizer_method: str = "Adam"  # or "AdamW"
    learning_rate_scheduler: str = "CosineAnnealing"
    weight_decay: float = None  # of "AdamW"
    warmup_epochs: int = None
    warmup_start_lr: float = None
    steps_per_epoch: int = None
    last_epoch: int = -1


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


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


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layeer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layeer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, image_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class Van(pl.LightningModule):
    def __init__(self, cfg:VanCfg
                 # image_size=224,
                 # in_chans=3,
                 # num_classes=1000,
                 # embed_dims=[64, 128, 256, 512],
                 # mlp_ratio=[4, 4, 4, 4],
                 # drop_rate=0.,
                 # drop_path_rate=0.,
                 # norm_layer=nn.LayerNorm,
                 # depths=[3, 4, 6, 3],
                 # num_stages=4,
                 # flag=False,
                 ):
        super().__init__()
        if cfg.flag == False:
            self.num_classes = cfg.num_classes

        self.depths = cfg.depths
        self.num_stages = cfg.num_stages

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths))]   #  随机深度衰减法则   stochastic depth decay rule
        cur = 0

        for i in range(cfg.num_stages):
            patch_embed = OverlapPatchEmbed(image_size=cfg.image_size if i == 0 else cfg.image_size // (2 ** (i+1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=cfg.in_chans if i == 0 else cfg.embed_dims[i - 1],
                                            embed_dim=cfg.embed_dims[i])

            block = nn.ModuleList([
                Block(
                    dim=cfg.embed_dims[i], mlp_ratio=cfg.mlp_ratio[i], drop=cfg.drop_rate, drop_path=dpr[cur + j]
                ) for j in range(cfg.depths[i])
            ])
            norm = cfg.norm_layer(cfg.embed_dims[i])
            cur += cfg.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)   # self.patch_embed1 ....
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            # self.head = nn.Linear(embed_dims[i], num_classes) if num_classes > 0 else nn.Identity()
        # classification head
        self.head = nn.Linear(cfg.embed_dims[3], cfg.num_classes) if cfg.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.cfg = cfg

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # print("B=%f" % B)
        # log_prior = repeat(self.log_prior, '1 n -> b n ', b=B)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            # print("patch_embed=%f" % x)
            for blk in block:
                x = blk(x)
                # print("blk=%f" % x)
            x = x.flatten(2).transpose(1, 2)
            # print("flatten=%f" % x)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # print("reshape=%f" % x)

        x = x.mean(dim=1)
        x = self.head(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate, 
                                          weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only support Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs,
                                                                      last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                                                               steps_per_epoch=self.cfg.steps_per_epoch,
                                                               epochs=self.cfg.num_epochs,
                                                               last_epoch=self.cfg.last_epoch)
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

        # if self.cfg.learning_rate_scheduler == "CosineAnnealing":
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        # else:
        #     lr_scheduler = None
        #
        # if lr_scheduler is None:
        #     return optimizer
        # else:
        #     return [optimizer], [lr_scheduler]


class bayes_Van(Van):
    def __init__(self, cfg: VanCfg):
        super().__init__(cfg)

        # embed_dims=[64, 128, 256, 512],
        head1 = nn.Linear(cfg.embed_dims[0], cfg.embed_dims[1]) if cfg.num_classes > 0 else nn.Identity()
        head2 = nn.Linear(cfg.embed_dims[1], cfg.embed_dims[2]) if cfg.num_classes > 0 else nn.Identity()
        head3 = nn.Linear(cfg.embed_dims[2], cfg.embed_dims[3]) if cfg.num_classes > 0 else nn.Identity()
        self.heads = nn.ModuleList([
            head1,
            head2,
            head3,
            self.head
        ])

        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior)
        #         self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))
        embed_dim = [128, 256, 512, cfg.num_classes]
        self.embed = embed_dim
        for i in range(cfg.num_stages):
            logits_layer_norm = nn.LayerNorm(self.embed[i])
            # log_prior = torch.zeros(1, self.embed[i])
            # setattr(self, f"log_prior{i + 1}", log_prior)
            setattr(self, f"logits_layer_norm{i + 1}", logits_layer_norm)
        # embed_dims=[128, 256, 512, num_classes]
        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        # self.norm = None
        self.apply(self._init_weights)
        self.cfg = cfg

    def forward_features(self, x):
        B = x.shape[0]
        # log_prior = repeat(self.log_prior, '1 n -> b n ', b=B)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            logits_layer_norm = getattr(self, f"logits_layer_norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
                logits = x.flatten(2).transpose(1, 2)
                logits = logits.mean(dim=1)
                logits = self.heads[i](logits)
                log_prior = logits_layer_norm(logits)
                log_prior = log_prior + logits

            # x = x.flatten(2).transpose(1, 2)
            # x = norm(x)
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # logits = x.mean(dim=1)
            # logits = self.head(logits)

        # x = x.mean(dim=1)
        # x = self.head(x)

        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only support Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs,
                                                                      last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                                                               steps_per_epoch=self.cfg.steps_per_epoch,
                                                               epochs=self.cfg.num_epochs,
                                                               last_epoch=self.cfg.last_epoch)
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


class bayes_Van2(Van):
    def __init__(self, cfg: VanCfg
                 # image_size=224,
                 # in_chans=3,
                 # num_classes=1000,
                 # embed_dims=[64, 128, 256, 512],
                 # mlp_ratio=[4, 4, 4, 4],
                 # drop_rate=0,
                 # drop_path_rate=0,
                 # norm_layer=nn.LayerNorm,
                 # depths=[3, 4, 6, 3],
                 # num_stages=4,
                 # flag=False
                 ):
        super().__init__(cfg)

        # embed_dims=[64, 128, 256, 512],
        head1 = nn.Linear(cfg.embed_dims[0], cfg.embed_dims[1]) if cfg.num_classes > 0 else nn.Identity()
        head2 = nn.Linear(cfg.embed_dims[1], cfg.embed_dims[2]) if cfg.num_classes > 0 else nn.Identity()
        head3 = nn.Linear(cfg.embed_dims[2], cfg.embed_dims[3]) if cfg.num_classes > 0 else nn.Identity()
        self.heads = nn.ModuleList([
            head1,
            head2,
            head3,
            self.head
        ])
        embed_dim = [128, 256, 512, cfg.num_classes]
        self.embed = embed_dim
        for i in range(cfg.num_stages):
            log_prior = torch.zeros(1, self.embed[i])
            setattr(self, f"log_prior{i + 1}", log_prior)

        # log_prior = torch.zeros(1, num_classes)
        # self.register_buffer('log_prior', log_prior)
        #         self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))

        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        # self.norm = None
        self.cfg = cfg
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            log_prior1 = getattr(self, f"log_prior{i + 1}")
            # log_prior = repeat(self.log_prior, '1 n -> b n ', b=B)
            print(log_prior1.shape)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)

            for blk in block:
                x = blk(x)
                logits = x.flatten(2).transpose(1, 2)
                logits = logits.mean(dim=1)
                logits = self.heads[i](logits)
                # log_prior = self.logits_layer_norm(logits)
                log_prior = logits + log_prior1
                log_prior = F.log_softmax(log_prior, dim=-1)
                log_prior = log_prior + math.log(self.embed[i])

        #     x = x.flatten(2).transpose(1, 2)
        #     x = norm(x)
        #     if i != self.num_stages - 1:
        #         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # x = x.mean(dim=1)
        # x = self.head(x)

        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only support Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs,
                                                                      last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                                                               steps_per_epoch=self.cfg.steps_per_epoch,
                                                               epochs=self.cfg.num_epochs,
                                                               last_epoch=self.cfg.last_epoch)
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

