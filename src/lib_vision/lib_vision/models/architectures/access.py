import re
from typing import Callable, Iterator, Optional, Union, cast

from torch.nn import Module, Sequential

from .cifar import densenet as cifar_densenet
from .cifar import resnet as cifar_resnet  # vit as cifar_vit,
from .cifar import vgg as cifar_vgg
from .cifar import vit as cifar_vit
from .create import ModelConfig


CIFAR_LAST_LAYER_NAMES = {
    "resnet-18": cifar_resnet.LAST_LAYER_NAME,
    "resnet-34": cifar_resnet.LAST_LAYER_NAME,
    "resnet-50": cifar_resnet.LAST_LAYER_NAME,
    "densenet-121": cifar_densenet.LAST_LAYER_NAME,
    "vgg-11": cifar_vgg.LAST_LAYER_NAME,
    "vit": cifar_vit.LAST_LAYER_NAME,
}
CIFAR_PENULTIMATE_LAYER_NAMES = {
    "resnet-18": cifar_resnet.PENULTIMATE_LAYER_NAME,
    "densenet-121": cifar_densenet.PENULTIMATE_LAYER_NAME,
    "vgg-11": cifar_vgg.PENULTIMATE_LAYER_NAME,
    "vit": cifar_vit.PENULTIMATE_LAYER_NAME,
}

IMAGENET_LAST_LAYER_NAMES = {
    "resnet-18": "fc",
    "densenet-121": "classifier",
    "vgg-11": "classifier[6]",
}
IMAGENET_PENULTIMATE_LAYER_NAMES = {
    "resnet-18": "avgpool",
    "densenet-121": "features.norm5",
    "vgg-11": "classifier[5]",
}


def get_last_layer(model_config: ModelConfig, model: Module) -> Module:
    layer_name = _get_layer_name(model_config, is_last=True)
    return get_layer(model, layer_name)


def get_penultimate_layer(model_config: ModelConfig, model: Module) -> Module:
    layer_name = get_penultimate_layer_name(model_config)
    return get_layer(model, layer_name)


def get_penultimate_layer_name(model_config: ModelConfig) -> str:
    return _get_layer_name(model_config, is_last=False)


def set_last_layer(
    model_config: ModelConfig, model: Module, layer: Module
) -> Module:
    layer_name = _get_layer_name(model_config, is_last=True)
    return set_layer(model, layer_name, layer)


def _get_layer_name(
    model_config: ModelConfig,
    is_last: bool,
) -> str:
    if model_config.domain == "cifar":
        if is_last:
            layer_names = CIFAR_LAST_LAYER_NAMES
        else:
            layer_names = CIFAR_PENULTIMATE_LAYER_NAMES
    else:
        if is_last:
            layer_names = IMAGENET_LAST_LAYER_NAMES
        else:
            layer_names = IMAGENET_PENULTIMATE_LAYER_NAMES
    model_type = model_config.type
    layer_name = layer_names[model_type]
    return layer_name


NameComponent = Union[str, int]


def get_layer(
    model: Module,
    *layer_name: NameComponent,
) -> Module:
    name_components = [
        parsed_component
        for name_component in layer_name
        for parsed_component in _get_name_components(name_component)
    ]

    cur_module = model
    for name_component in name_components:
        if isinstance(name_component, str):
            cur_module = getattr(cur_module, name_component)
        else:
            # Index access
            assert isinstance(cur_module, Sequential)
            cur_module = cast(Sequential, cur_module)[name_component]
    return cur_module


def set_layer(
    model: Module,
    layer_name: Union[NameComponent, list[NameComponent]],
    layer: Module,
) -> Module:
    if isinstance(layer_name, list):
        name_components = [
            parsed_component
            for name_component in layer_name
            for parsed_component in _get_name_components(name_component)
        ]
    else:
        name_components = _get_name_components(layer_name)
    parent_module = get_layer(model, *name_components[:-1])
    last_component = name_components[-1]
    if isinstance(last_component, str):
        setattr(parent_module, last_component, layer)
    else:
        # Index access
        assert isinstance(parent_module, Sequential)
        parent_module[last_component] = layer
    return model


INDEX_PATTERN = re.compile(r"^\w+\[\d+\]$")


def _get_name_components(layer_name: NameComponent) -> list[NameComponent]:
    if isinstance(layer_name, int):
        return [layer_name]

    # The name component is a string, parse it
    name_components = []
    layer_dot_components = layer_name.split(".")
    for component in layer_dot_components:
        if INDEX_PATTERN.match(component) is not None:
            # This is an index access
            component_parts = component.split("[")
            name_components.append(component_parts[0])
            # Convert and add the numeric index
            name_components.append(int(component_parts[1][:-1]))
        else:
            name_components.append(component)
    return name_components
