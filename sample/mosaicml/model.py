import torch
from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from sample.pytorch.py_arch.convmixer_ml import BayesConvMixer, ConvMixer, ConvMixerCfg


def build_composer_convmixer(model_name: str = 'convmixer',
                             num_layers: int = 8,
                             hidden_dim: int = 256,
                             patch_size: int = 7,
                             kernel_size: int = 5,
                             num_classes: int = 100,
                             layer_norm_zero_init: bool = True,
                             ):
    
    cfg = ConvMixerCfg(
        num_layers = num_layers,
        hidden_dim = hidden_dim,
        patch_size = patch_size,
        kernel_size = kernel_size,
        num_classes = num_classes,
        layer_norm_zero_init = layer_norm_zero_init,
    )

    if model_name == "convmixer":
        model = ConvMixer(cfg)
    elif model_name == "bayes_convmixer":
        model = BayesConvMixer(cfg)
    else:
        raise ValueError(f"model_name='{model_name}' but only 'convmixer' and 'bayes_convmixer' are supported now.")
    
    # Performance metrics to log other than training loss
    train_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    val_metrics = MetricCollection([
        CrossEntropy(),
        MulticlassAccuracy(num_classes=num_classes, average='micro')
    ])

    # Wrapper function to convert a image classification Pytorch model into a Composer model
    composer_model = ComposerClassifier(
        model,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        loss_fn=soft_cross_entropy,
    )
    return composer_model
# %%
# composer_model = build_composer_convmixer()
# input = [torch.randn(2, 3, 224, 224), torch.randint(0,100, (2,))]
# output = composer_model(input)
# output.shape