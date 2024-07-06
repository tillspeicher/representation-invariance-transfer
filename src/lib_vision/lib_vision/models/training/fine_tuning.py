from dataclasses import dataclass
from typing import cast

from lora_pytorch import LoRA
from torch import nn

from ..architectures.access import get_last_layer, set_last_layer
from ..architectures.create import ModelConfig, create_model
from .utils import disable_gradients


@dataclass
class FineTuningFreezeConfig:
    type: str  # linear_probe, full
    reset_head: bool = True
    lora_rank: int = 4


# Based on https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html # noqa: W505
def prepare_torchvision_finetuning(
    model: nn.Module,
    model_config: ModelConfig,
    freeze_config: FineTuningFreezeConfig,
) -> nn.Module:
    """Prepare a torchvision model for finetuning or feature extraction.

    Parameters
    ---------
    model_name: TorchvisionModelName
        The id of the model family, e.g. 'resnet', 'alexnet', 'vgg', etc.
    model: torch.nn.Module
        The model instance that should be finetuned.
        This must be a model from torchvision.models corresponding to
        the provided model_name.
    num_classes: int
        The number of classes of the new fully-connected final layer
    feature_extract: bool
        If True, all layers except the new last layer will be frozen.
        If False, the entire model has gradients enabled for finetuning.

    Returns
    -------
    model: torch.nn.Module
        The model with a new final layer and disabled gradients except for
        the final layer if feature_extract is set to True

    Raises
    ------
    ValueError if model_name is invalid.
    """

    if freeze_config.type == "linear_probe":
        model_ft = disable_gradients(model)
        return _reinit_last_layer(model_ft, model_config)
    elif freeze_config.type == "full":
        model_ft = model
        if freeze_config.reset_head:
            model_ft = _reinit_last_layer(model_ft, model_config)
        return model_ft
    elif freeze_config.type == "lora":
        last_layer = get_last_layer(model_config, model)
        model_ft = LoRA.from_module(model, rank=freeze_config.lora_rank).module
        if freeze_config.reset_head:
            set_last_layer(model_config, model_ft, layer=last_layer)
            _reinit_last_layer(model_ft, model_config)
        return model_ft
    else:
        raise ValueError(f"Invalid fine-tuning type: {freeze_config.type}")


def _reinit_last_layer(
    model: nn.Module,
    model_config: ModelConfig,
) -> nn.Module:
    last_layer = get_last_layer(model_config, model)
    num_features = cast(int, last_layer.in_features)
    new_last_layer = nn.Linear(num_features, model_config.num_classes)
    return set_last_layer(model_config, model, layer=new_last_layer)


def copy_model(
    model: nn.Module,
    model_config: ModelConfig,
) -> nn.Module:
    copied_model = create_model(model_config)
    copied_model.load_state_dict(model.state_dict())
    return copied_model
