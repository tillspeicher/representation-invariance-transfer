from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import torch
import torchvision
from torch.nn import Module

from .cifar import densenet, resnet, vgg, vit


# ModelType = Literal["resnet-18", "densenet-121", "vgg-11"]
# ModelDomain = Literal["cifar", "imagenet"]


@dataclass
class ModelConfig:
    type: str  # ModelType
    domain: str  # ModelDomain
    num_classes: int
    init_seed: int


cifar_models = {
    "resnet-18": resnet.ResNet18,
    "resnet-34": resnet.ResNet34,
    "resnet-50": resnet.ResNet50,
    "densenet-121": densenet.densenet_cifar,
    "vgg-11": vgg.VGG11,
    "vit": vit.DefaultViT,
}
imagenet_models = {
    "resnet-18": (
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.DEFAULT,
    ),
    "densenet-121": (
        torchvision.models.densenet121,
        torchvision.models.DenseNet121_Weights.DEFAULT,
    ),
    "vgg-11": (
        torchvision.models.vgg11,
        torchvision.models.VGG11_Weights.DEFAULT,
    ),
}


def create_model(
    config: ModelConfig,
    pretrained: Union[bool, str, Path] = False,
) -> Module:
    load_pytorch_pretrained = pretrained is True
    if config.domain == "cifar":
        if load_pytorch_pretrained:
            raise ValueError("Cannot load PyTorch pretrained models for CIFAR")
        model_constructor = cifar_models[config.type]
        model_args = {"num_classes": config.num_classes}
    elif config.domain == "imagenet":
        model_constructor, model_weights = imagenet_models[config.type]
        model_args = {
            "num_classes": config.num_classes,
            "weights": model_weights if load_pytorch_pretrained else None,
        }
    else:
        raise ValueError(f"Invalid domain {config.domain}")

    torch.manual_seed(config.init_seed)
    model = model_constructor(**model_args)

    if isinstance(pretrained, (str, Path)):
        state_dict = torch.load(pretrained)["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in state_dict.items()
        }  # remove 'model.' prefix
        model.load_state_dict(state_dict)
    return model
