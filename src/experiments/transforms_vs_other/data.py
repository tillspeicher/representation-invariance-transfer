from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

import transforms_2d as t2d
from lib_vision.data.lib.multiclass_datamodule import MulticlassDataModule
from utils.data import create_t2d_like_rw_dataset


ARCHITECTURES = [
    "resnet-18",
    "resnet-50",
    "densenet-121",
    "vgg-11",
    "vit",
]
CLASS_RELATIONSHIPS = [
    "subset",
    "disjoint",
    "same",
    "superset",
]
SAMPLE_COUNTS_PT = {
    "t2d": [1000, 10000, 50000, 100000, 500000],
    # For CIFAR, the sample counts are counts per class
    "cifar10": [10, 50, 200, 1000, 4500],
    "cifar100": [10, 50, 100, 250, 500],
}
SAMPLE_COUNTS_FT = {
    "t2d": [50, 200, 500, 2000, 5000],
    # For CIFAR, the sample counts are percentages per class
    "cifar10": [0.01, 0.05, 0.1, 0.5, 1.0],
    "cifar100": [0.01, 0.05, 0.1, 0.5, 1.0],
}


@dataclass
class TvODataConfig(t2d.Transforms2DConfig):
    data_type: str
    # Used to determine the structure of the dataset, i.e. which transforms
    # and classes to use
    config_seed: int
    num_target_classes: int
    transform_type: str = "mixed"
    classes_rel_range: list[str] = field(
        default_factory=lambda: CLASS_RELATIONSHIPS
    )
    arch_range: list[str] = field(default_factory=lambda: ARCHITECTURES)


@dataclass
class SampledDataConfig:
    target_transforms: list[str]
    disjoint_transforms: list[str]
    target_objects: list[int]
    training_objects: list[int]


@dataclass
class TvOData:
    # Maps (transform setting relative to the target task, e.g.
    # subset, disjoint) -> (value of the variation parameter, e.g.
    # number of samples or classes) -> dataset
    same_transforms_datasets: dict[str, MulticlassDataModule]
    disjoint_transforms_datasets: dict[str, MulticlassDataModule]
    target_datasets: dict[str, MulticlassDataModule]
    sampled_config: SampledDataConfig


def create_tvo_datasets(
    config: TvODataConfig,
    variation_category: str,
    num_ft_samples: int | float,
    normalize: bool = True,
) -> TvOData:
    rng = np.random.default_rng(config.config_seed)
    target_transforms, disjoint_transforms = _get_transforms(
        config.transform_type,
        rng,
    )
    print("target transforms:", target_transforms)
    print("disjoint transforms:", disjoint_transforms)
    training_classes, target_classes = _get_classes(
        "disjoint",
        num_target_classes=config.num_target_classes,
        is_t2d=config.data_type == "t2d",
        rng=rng,
    )

    def create_datasets(
        same_transforms: bool,
        use_training_classes: bool,
        variation_values: dict[str, dict[str, Any]],
    ) -> dict[str, MulticlassDataModule]:
        """Creates the set of datasets using default parameters, but
        allows overwriting parameters as necessary"""
        if same_transforms:
            transforms = target_transforms
        else:
            transforms = disjoint_transforms
        if use_training_classes:
            classes = training_classes
        else:
            classes = target_classes
        return {
            value_name: _create_dataset(
                data_type=config.data_type,
                config=config,
                transforms=transforms,
                classes=classes,
                values=value,
                normalize=normalize,
            )
            for value_name, value in variation_values.items()
        }

    if config.data_type == "t2d":
        # Create two categories of transfer datasets, one with
        # multiple object prototypes per class and one with a
        # single opject prototype per class
        target_variation_values = {
            "single_object": {
                "use_single_object": True,
                "config": _get_num_samples_config(
                    config, cast(Any, num_ft_samples)
                ),
            },
            # "multi_object": {
            #     "use_single_object": False,
            # },
        }
    else:
        target_variation_values = {
            "default": {
                "config": _get_num_samples_config(
                    config,
                    cast(Any, num_ft_samples),
                )
            }
        }
    target_datasets = create_datasets(
        same_transforms=True,
        use_training_classes=False,
        variation_values=target_variation_values,
    )

    if variation_category == "none":
        variation_values = {
            "default": {
                "config": _get_num_samples_config(
                    config,
                    cast(int, num_ft_samples),
                )
            }
        }
        same_transforms_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=True,
            variation_values=variation_values,
        )
        disjoint_transforms_datasets = create_datasets(
            same_transforms=False,
            use_training_classes=True,
            variation_values=variation_values,
        )
    elif variation_category == "class_relationship":
        foreground_values = {
            class_rel: {
                "foregrounds": _get_classes(
                    class_rel,
                    num_target_classes=config.num_target_classes,
                    is_t2d=config.data_type == "t2d",
                    rng=rng,
                    existing_target_classes=target_classes,
                    existing_training_classes=training_classes,
                )[0]
            }
            for class_rel in CLASS_RELATIONSHIPS
        }
        same_transforms_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=True,
            variation_values=foreground_values,
        )
        disjoint_transforms_datasets = create_datasets(
            same_transforms=False,
            use_training_classes=True,
            variation_values=foreground_values,
        )
    elif variation_category == "architecture":
        architecture_values = {
            architecture: {} for architecture in ARCHITECTURES
        }
        same_transforms_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=True,
            variation_values=architecture_values,
        )
        disjoint_transforms_datasets = create_datasets(
            same_transforms=False,
            use_training_classes=True,
            variation_values=architecture_values,
        )
    elif variation_category == "pretraining_samples":
        n_samples_range = SAMPLE_COUNTS_PT[config.data_type]
        num_samples_values = {
            str(num_samples): {
                "config": _get_num_samples_config(
                    config,
                    num_samples,
                )
            }
            for num_samples in n_samples_range
        }
        same_transforms_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=True,
            variation_values=num_samples_values,
        )
        disjoint_transforms_datasets = create_datasets(
            same_transforms=False,
            use_training_classes=True,
            variation_values=num_samples_values,
        )
    elif variation_category == "finetuning_samples":
        pt_values = {"default": {"config": deepcopy(config)}}
        same_transforms_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=False,
            variation_values=pt_values,
        )
        disjoint_transforms_datasets = create_datasets(
            same_transforms=False,
            use_training_classes=False,
            variation_values=pt_values,
        )

        n_samples_range = SAMPLE_COUNTS_FT[config.data_type]
        num_samples_values = {
            str(num_samples): {
                "config": _get_num_samples_config(
                    config,
                    num_samples,
                )
            }
            for num_samples in n_samples_range
        }
        target_datasets = create_datasets(
            same_transforms=True,
            use_training_classes=False,
            variation_values=num_samples_values,
        )
    else:
        raise ValueError(f"Invalid variation category: {variation_category}")

    sampled_config = SampledDataConfig(
        target_transforms=target_transforms,
        disjoint_transforms=disjoint_transforms,
        target_objects=target_classes,
        training_objects=training_classes,
    )
    return TvOData(
        same_transforms_datasets=same_transforms_datasets,
        disjoint_transforms_datasets=disjoint_transforms_datasets,
        target_datasets=target_datasets,
        sampled_config=sampled_config,
    )


TRANSFORM_NAME_CANDIDATES = {
    "geometric": list(t2d.GEOMETRIC_TRANSFORMS),
    "photometric": list(t2d.PHOTOMETRIC_TRANSFORMS),
    "corruption": list(t2d.CORRUPTION_TRANSFORMS),
}
N_TRANSFORMS = 3


def _get_transforms(
    transforms_type: str, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    if transforms_type == "mixed":
        transforms_1: list[str] = []
        transforms_2: list[str] = []
        for transform_type_names in TRANSFORM_NAME_CANDIDATES.values():
            type_transforms = rng.choice(
                transform_type_names,
                2,
                replace=False,
            )
            transforms_1.append(type_transforms[0])
            transforms_2.append(type_transforms[1])
    elif transforms_type == "adversarial":
        raise NotImplementedError()
    elif transforms_type in TRANSFORM_NAME_CANDIDATES:
        transform_names = list(
            rng.choice(
                TRANSFORM_NAME_CANDIDATES[transforms_type],
                2 * N_TRANSFORMS,
                replace=False,
            )
        )
        transforms_1 = transform_names[:N_TRANSFORMS]
        transforms_2 = transform_names[N_TRANSFORMS:]
    else:
        raise ValueError(f"Invalid transform type: {transforms_type}")

    # Converting from numpy strings to python strings
    transforms_1 = [str(t) for t in transforms_1]
    transforms_2 = [str(t) for t in transforms_2]
    return transforms_1, transforms_2


CLASS_SUBSET_RATIO = 1 / 3
CLASS_SUPERSET_RATIO = 2.0


def _get_classes(
    class_relationship: str,
    num_target_classes: int,
    is_t2d: bool,
    rng: np.random.Generator,
    existing_target_classes: Optional[list[int]] = None,
    existing_training_classes: Optional[list[int]] = None,
) -> tuple[list[int], list[int]]:
    num_subset_classes = max(2, round(num_target_classes * CLASS_SUBSET_RATIO))
    num_superset_classes = round(num_target_classes * CLASS_SUPERSET_RATIO)

    if existing_target_classes is None:
        assert existing_training_classes is None
        n_classes_to_sample = max(num_superset_classes, 2 * num_target_classes)
        # Subsample foreground objects
        if is_t2d:
            class_candidates = t2d.OBJECTS
        else:
            class_candidates = np.arange(n_classes_to_sample)
        fg_indices = list(
            rng.choice(class_candidates, n_classes_to_sample, replace=False)
        )
        existing_target_classes = [
            int(fgi) for fgi in fg_indices[:num_target_classes]
        ]
    else:
        assert existing_training_classes is not None
        # We've already sampled training and target classes, now we have
        # to generate the class relationship classes.
        fg_indices = existing_target_classes + existing_training_classes

    if class_relationship == "subset":
        existing_training_classes = existing_target_classes[:num_subset_classes]
    elif class_relationship == "same":
        existing_training_classes = existing_target_classes
    elif class_relationship == "disjoint":
        existing_training_classes = fg_indices[
            num_target_classes : 2 * num_target_classes
        ]
    elif class_relationship == "superset":
        existing_training_classes = fg_indices[:num_superset_classes]
    else:
        raise ValueError(f"Invalid class relationship: {class_relationship}")

    existing_training_classes = [int(fgi) for fgi in existing_training_classes]
    existing_target_classes = [int(fgi) for fgi in existing_target_classes]
    return existing_training_classes, existing_target_classes


def _get_num_samples_config(
    config: TvODataConfig,
    num_training_samples: int,
) -> TvODataConfig:
    num_samples_config = deepcopy(config)
    num_samples_config.n_training_samples = num_training_samples
    return num_samples_config


def _create_dataset(
    data_type: str,
    config: t2d.Transforms2DConfig,
    transforms: list[str],
    classes: list[int],
    values: dict[str, Any],
    normalize: bool = True,
) -> MulticlassDataModule:
    if data_type == "t2d":
        # pass the arguments as kwargs so that arguments in values
        # can overwrite the other ones
        return t2d.create_t2d_dataset(
            **{
                "transforms": transforms,
                "config": config,
                "foregrounds": classes,
                "normalize": normalize,
                **values,
            }
        )
    elif data_type.startswith("cifar"):
        if "config" in values:
            # Use the overwritten config from the variation values
            config = values["config"]
        transforms_sampling_seed = config.transforms_sampling_seed
        assert transforms_sampling_seed is not None
        num_samples = config.n_training_samples
        if "foregrounds" in values:
            subsample_classes = values["foregrounds"]
        else:
            subsample_classes = classes
        return create_t2d_like_rw_dataset(
            data_type=data_type,
            transforms=transforms,
            sampling_seed=config.sampling_seed,
            transforms_sampling_seed=transforms_sampling_seed,
            classes=subsample_classes,
            subsample=num_samples,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unsupported data type '{data_type}'")
