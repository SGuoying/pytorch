# %%
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# %%
class BayesResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        avgpool1 = nn.AdaptiveAvgPool2d((2, 4))
        avgpool2 = nn.AdaptiveAvgPool2d((2, 2))
        avgpool3 = nn.AdaptiveAvgPool2d((2, 1))
        self.avgpools = nn.ModuleList([
            avgpool1,
            avgpool2, 
            avgpool3,
            self.avgpool,
        ])
        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        self.logits_bias = Parameter(torch.zeros(1, num_classes))

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.avgpools[i](x)
                logits = torch.flatten(logits, start_dim=1)
                logits = self.fc(logits)
                log_prior = log_prior + logits
                log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior

# %%
class BayesResNet2(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64 * i * expansion, num_classes)
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                self.avgpool,
                nn.Flatten(),
                self.fc,
            )
        ])

        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.digups[i](x)
                log_prior = log_prior + logits
                # log_prior = self.logits_layer_norm(log_prior)
                log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                # log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior