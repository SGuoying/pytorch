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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration, DeepBayesInferCfg

@dataclass
class DeepBayesInferLMCfg(DeepBayesInferCfg):
    vocab_size: int = None
    seq_len: int = None


class DeepBayesInferLM(pl.LightningModule):
    """
    Deep Bayesian Inference Architecture for Language Models.
    Parameters
    ----------
    layers: ModuleList
        a chain of neural network layers
    Attributes
    ----------
    layers: ModuleList
        a chain of neural network layers, that transform hidden feature vectors
    embed: Module
        map word sequence to hidden feature vectors
    digup: Module
        map hidden feature vectors to logits, i.e. unscaled log probabilities
    
    """
    def __init__(self, layers: nn.ModuleList, cfg: DeepBayesInferLMCfg):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.init_weights()

        if cfg.is_sharing_weight:
            self.digup.weight = self.embed.weight
        
        if cfg.is_layernorm_before_digup:
            self.digup = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim, eps=1e-12),
                self.digup
            )

        log_prior = torch.zeros(1, cfg.vocab_size)  # sequence len of prior should be substracted by 1
        if cfg.is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior)

        self.vocab_size, self.hidden_dim, self.learning_rate = cfg.vocab_size, cfg.hidden_dim, cfg.learning_rate
        
    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input.shape
        log_prior = repeat(self.log_prior, '1 v -> b v', b=batch_size)        
        # log_prior = torch.zeros_like(input).unsqueeze(-1).repeat((1, 1, self.vocab_size))

        hidden_features = self.embed(input)

        for layer in self.layers:
            hidden_features = layer(hidden_features)
            chosen_hidden_features = hidden_features[:, -1]
            logits = self.digup(chosen_hidden_features)

            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def training_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        # log_posterior = log_posterior.permute((0, 2, 1))
        loss = F.nll_loss(log_posterior, target[:, -1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        # log_posterior = log_posterior.permute((0, 2, 1))
        loss = F.nll_loss(log_posterior, target[:, -1])
        self.log("val_loss", loss)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    cfg = DeepBayesInferLMCfg(vocab_size=16, hidden_dim=8, seq_len=8, batch_size=2, learning_rate=1e-3)
    batch = torch.randint(0, cfg.vocab_size-1, (cfg.batch_size, cfg.seq_len+1))
    input = batch[:, :-1]

    layers = [nn.Identity()]
    deep_bayesian_net = DeepBayesInferLM(layers, cfg)

    log_posterior = deep_bayesian_net(input)
    posterior = torch.exp(log_posterior)

    assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 