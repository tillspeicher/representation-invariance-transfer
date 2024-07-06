from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from typing import Optional, Union

import lightning as L

from lib_vision.data.loading import DataLoaderConfig

from .datasets import CompositeTransform
from .transforms_2d import (
    BGSource,
    Composite2DDatasetConfig,
    FGSource,
    ImageBackgroundsConfig,
    ImageForegroundsConfig,
    Transforms2DData,
    Transforms2DDataConfig,
)


@dataclass
class Transforms2DConfig:
    """Configuration for the Transforms-2D dataset creation utility.

    Args:
        sampling_seed: Seed to used for sampling the dataset, i.e.
            for sampling transformations and backgrounds.
            This seeds stands in contrast to the config seed, which is
            used to determine the structure of the dataset, i.e. which
            objects and transformations to use.
        img_size: Size of the images in the dataset.
        n_training_samples: Number of training samples to use.
        n_val_samples: Number of validation samples to use.
        n_test_samples: Number of test samples to use.
        batch_size: Batch size to use for the dataset.
        transforms_sampling_seed: Seed to independently control the
            sampling of transformations. This is useful to create two
            Transform-2D datasets that only differ in the transformations
            they use. None by default uses the `sampling_seed` for
            transformation sampling as well.
    """

    sampling_seed: int
    img_size: int
    n_training_samples: int
    n_val_samples: int
    n_test_samples: int
    batch_size: int
    _: KW_ONLY
    transforms_sampling_seed: Optional[int] = None


def create_t2d_dataset(
    config: Transforms2DConfig,
    transforms: CompositeTransform,
    foregrounds: Union[FGSource, list[int]],
    backgrounds: Union[BGSource, None] = None,
    normalize: bool = True,
    class_from_foreground: bool = True,
    fg_bg_correlation: float = 0.0,
    fg_availability: float = 1.0,
    use_single_object: bool = True,
) -> Transforms2DData:
    """Utility method to create a Transforms-2D dataset.

    Args:
        config: Configuration for the dataset.
        transforms: The list of transformations to apply to the images.
            In the simplest case this is a list of transformation names,
            e.g. ["translate", "invert"].
        foregrounds: The list of foreground objects indices to use.
        backgrounds: Background to use. If None, use the default set of
            image backgrounds.
        normalize: Whether to normalize the data. This should be true
            if the images are to be consumed by a model, and false
            if they are to be visualized.
        class_from_foreground: Whether to use the foreground class as
            the class of the image. Uses the background class otherwise.
        fg_bg_correlation: Correlation between foreground and background classes.
        fg_availability: The probability of pasting a foreground object
            onto the background.
        use_single_object: Most classes in the base dataset have multiple
            image prototypes available. If `use_single_object` is true we
            only use one of them to ensure that all differences within
            the classes are due to the applied transformations. Otherwise,
            we load all prototypes per class.
    """
    if isinstance(foregrounds, list):
        foregrounds_config = ImageForegroundsConfig(
            fg_class_indices=foregrounds,
            transforms=transforms,
            img_size=config.img_size,
            use_single_object=use_single_object,
        )
    elif isinstance(foregrounds, L.LightningDataModule):
        foregrounds_config = foregrounds
    else:
        foregrounds_config = deepcopy(foregrounds)
        foregrounds_config.img_size = config.img_size
        foregrounds_config.transforms = transforms
    if backgrounds is None:
        backgrounds_config = ImageBackgroundsConfig(
            img_size=config.img_size,
        )
    elif isinstance(backgrounds, L.LightningDataModule):
        backgrounds_config = backgrounds
    else:
        backgrounds_config = deepcopy(backgrounds)
        backgrounds_config.img_size = config.img_size
    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
    )
    composition_config = Composite2DDatasetConfig(
        img_size=config.img_size,
        sampling_seed=config.sampling_seed,
        normalize=normalize,
        class_from_foreground=class_from_foreground,
        fg_bg_correlation=fg_bg_correlation,
        fg_availability=fg_availability,
    )
    dataset_config = Transforms2DDataConfig(
        foregrounds=foregrounds_config,
        backgrounds=backgrounds_config,
        composition=composition_config,
        n_training_samples=config.n_training_samples,
        n_val_samples=config.n_val_samples,
        n_test_samples=config.n_test_samples,
        sampling_seed=config.sampling_seed,
        transforms_sampling_seed=config.transforms_sampling_seed,
        loader=loader_config,
    )
    data = Transforms2DData(dataset_config)
    return data
