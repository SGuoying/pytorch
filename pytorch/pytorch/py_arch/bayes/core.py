"""Deep Bayesian Inference Architecture
A deep Bayesian inference architecture includes four parts:
- Deep Bayesian Inference composed with a chain of Bayesian iterations.
- Neural Network composed with a chain of layers.
- Embed Block mapping from the observable space to the hidden space.
- Dig-up Block mapping from the hidden space to the probability space.
Classes:
Functions:
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DeepBayesInferCfg:
    hidden_dim: int = 128  # 1024

    # attention
    is_attn: bool = True
    num_heads: int = 2  # 16
    attn_scale: float = None
    attn_dropout: float = 0.
    is_mask: bool = True
    is_softmax: bool = True

    # feed forward
    is_ff: bool = True
    expanded_dim: int = 256  # 2048
    ff_dropout: float = 0.
    ff_act_nn: nn.Module = nn.GELU()

    # layernorm
    is_attn_layernorm: bool = True
    is_ff_layernorm: bool = True

    # shortcut
    is_attn_shortcut: bool = True
    is_ff_shortcut: bool = True

    # bayes
    is_layernorm_before_digup: bool = True
    is_sharing_weight: bool = False
    is_prior_as_params: bool = False

    # training
    num_layers: int = 8
    batch_size: int = None
    num_epochs: int = 1
    optimizer_method: str = "Adam"  # or "AdamW"
    learning_rate: float = None
    learning_rate_scheduler: str = "Step"  # or "OneCycle" or "ReduceLROnPlateau"

    gamma: float = 0.7
    factor: float = 0.1
    patience: int = 3



def log_bayesian_iteration(log_prior: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    log_total_evidence = torch.logsumexp(log_prior + logits, dim=-1, keepdim=True)
    log_posterior = log_prior + logits - log_total_evidence
    return log_posterior