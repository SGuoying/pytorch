from dataclasses import dataclass
from pandas import DataFrame
from tokenizers import NormalizedString
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from structure.pytorch.py_arch.base import Residual
from structure.pytorch.py_arch.deepattn import AttnLayer, Squeeze


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x:torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim:int, drop_path:float=0., layer_scale_init_value:float=1e-6,
                        kernel_size:int=7, expansion:int=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                    requires_grad=True) if layer_scale_init_value > 0. else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = self.drop_path(x)
        return x


class ConvStage(nn.Sequential):
    def __init__(self, num_blocks:int, dim:int, drop_paths:tuple, layer_scale_init_value:float=1e-6):
        super().__init__(
            *[
                Residual(Block(dim, drop_paths[i], layer_scale_init_value))
                for i in range(num_blocks)
            ]
        )


class ConvNext(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = ConvStage(depths[i], dims[i], drop_rates[cur:cur+depths[i]], layer_scale_init_value)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self,m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
        drop_path:float=0., 
        layer_scale_init_value:float=1e-6,
    ):
        super().__init__()
        self.attn_layer = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)
        self.block = Block(hidden_dim, drop_path, layer_scale_init_value)

    def forward(self, xs, all_squeezed):
        x_new, all_squeezed = self.attn_layer(xs, all_squeezed)
        # x_new shape (batch_size, hidden_dim, height, width)
        x_next = self.block(x_new)
        x_next = x_next.unsqueeze(0)
        return torch.cat((xs, x_next), dim=0), all_squeezed


class Stage(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
        drop_paths:tuple = None, 
        layer_scale_init_value:float=1e-6,
    ):
        super().__init__()
        assert num_layers > 1
        self.first_block = Block(hidden_dim, drop_paths[0], layer_scale_init_value)
        self.first_squeeze = Squeeze(hidden_dim, init_scale)
        self.layers = nn.ModuleList([
            Layer(hidden_dim, num_heads, query_idx, temperature, init_scale, drop_paths[i + 1], layer_scale_init_value)
            for i in range(num_layers - 1)
        ])
        self.final_attn = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)

    def forward(self, x):
        xs = torch.stack([x, self.first_block(x)], dim=0)
        all_squeezed = self.first_squeeze(x).unsqueeze(0)
        for layer in self.layers:
            xs, all_squeezed = layer(xs, all_squeezed)
        x, all_squeezed = self.final_attn(xs, all_squeezed)
        return x


class AttnConvNext(ConvNext):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
        num_heads: int = 1,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,

    ):
        super().__init__(in_chans,num_classes, depths, dims, drop_path_rate, layer_scale_init_value, head_init_scale)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = Stage(depths[i], dims[i], 
                num_heads, query_idx, temperature, init_scale, drop_rates[cur:cur+depths[i]], layer_scale_init_value)
            self.stages.append(stage)
            cur += depths[i]


@register_model
def attnconvnext_tiny(pretrained=False, pretrained_cfg=None, **kwargs):
    model = AttnConvNext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


@register_model
def convnext_tiny(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ConvNext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def convnext_small(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ConvNext(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@dataclass
class ConvNextCfg:
    batch_size: int = 64 # Per GPU batch size
    epochs: int = 300
    update_freq: int = 1  # gradient accumulation steps

    drop_path: float = 0.  # drop path rate
    input_size: int = 224  # image input size
    layer_scale_init_value: float = 1e-6

    # EMA related parameters
    model_ema: bool = False
    model_ema_decay: float = 0.9999
    model_ema_force_cpu: bool = False
    model_ema_eval: bool = False  # using ema to eval during training

    # Optimization parameters
    opt: str = "adamw"  # Optimizer
    opt_eps: float = 1e-8  # Optimizer Epsilon
    opt_betas: float = None  # Optimizer Betas (default: None, use opt default)
    clip_grad: float = None  # Clip gradient norm (default: None, no clipping)
    momentum: float = 0.9  # SGD momentum (default: 0.9)
    weight_decay: float = 0.05  #
    weight_decay_end: float = None # Final value of the weight decay. We use a cosine schedule for WD

    lr: float = 4e-3  # learning rate (default: 4e-3), with total batch size 4096
    layer_decay: float = 1.0
    min_lr: float = 1e-6  # lower lr bound for cyclic schedulers that hit 0 (1e-6)
    warmup_epochs: int = 20 # epochs to warmup LR, if scheduler supports
    warmup_steps: int = -1 # num of steps to warmup LR, will overload warmup_epochs if set > 0

    # Augmentation parameters
    color_jitter: float = 0.4 # Color jitter factor
    aa: str = "rand-m9-mstd0.5-incl" # use AutoAugment policy. "v0" or "original"
    smoothing: float = 0.1 # label smoothing
    train_interpolation: str = 'bicubic' # training interpolation (random, bilinear, bicubic)

    # Evaluation parameters
    crop_pct: float = None

    # Random Erase params
    reprob: float = 0.25 # random erase prob
    remode: str = 'pixel' # random erase mode
    recount: int = 1 # random erase count
    resplit: bool = False # do not random erase first (clean) augmentation split

    # Mixup params
    mixup: float = 0.8 # mixup alpha, mixup enabled if > 0
    cutmix: float = 1.0 # cutmix alpha, cutmix enabled if > 0
    cutmix_minmax: float = None # cutmix min/max ratio, overrides alpha and enables cutmix if set
    mixup_prob: float = 1.0 # probability of performing mixup or cutmix when either/both is enabled
    mixup_switch_prob: float = 0.5 # probability of switching to cutmix when both mixup and cutmix enabled
    mixup_mode: str = 'batch' # how to apply mixup/cutmix params, Per 'batch', 'pair' or 'elem'

    # Finetuning params
    finetune: str = '' # finetune from checkpoint
    head_init_scale: float = 1.0 # classifier head initial scale, typically adjusted in fine-tuning
    # model_key, model_prefix

    # Dataset params
    data_path: str = None # dataset path
    eval_data_path: str = None # evaluation dataset path
    nb_classes: int = 1000 # number of the classification types
    imagenet_default_mean_and_std: bool = True
    data_set: str = 'IMNET'
    output_dir: str = '' # where to save, empty for no saving
    log_dir: str = None # where to tensorboard log
    device: str = 'cuda' # device to use for training / testing
    seed: int = 0

    resume: str = '' # resume from checkpoint
    auto_resume: bool = True
    save_ckpt: bool = True
    save_ckpt_freq: int = 1
    save_ckpt_num: int = 3

    start_epoch: int = 0
    eval: bool = False # perform evaluation only
    dist_eval: bool = True # enabling distributed evaluation
    disable_eval: bool = False # disabling eval during training
    num_workers: int = 10
    pin_mem: bool = True