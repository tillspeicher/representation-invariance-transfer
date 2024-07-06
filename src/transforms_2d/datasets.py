import functools
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, TypeVar, Union, cast

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pandas.core.common import contextlib
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms as torch_transforms

from lib_vision.data.wrappers.data_sample import DataSample
from lib_vision.data.wrappers.module import get_pseudorandom_idx

from .image_utils import calc_top_left_coordinates, crop_image_to_square, resize
from .transforms import ImageWrapper, get_transform, scale_object


HF_BASE_DATASET_ID = "tillspeicher/transforms_2d_base"

OBJECTS = list(range(61))
DEFAULT_OBJ_SCALE = 0.7


TransformOptions = Union[list[str], list[torch.nn.Module]]
CompositeTransform = Union[TransformOptions, dict[int, TransformOptions]]


@dataclass
class ForegroundsConfig:
    img_size: int
    transforms: CompositeTransform


@dataclass
class ImageForegroundsConfig(ForegroundsConfig):
    fg_class_indices: list[int]
    use_single_object: bool = True


@dataclass
class RandomForegroundsConfig(ForegroundsConfig):
    n_classes: int
    seed: int


FGConfig = Union[ImageForegroundsConfig, RandomForegroundsConfig]

T = TypeVar("T")


@dataclass
class Transforms2DDataSample(DataSample[T], Generic[T]):
    """A potential second input, that uses the same foreground and
    background object, but differs in the transform applied to the
    foreground object."""

    twin_inupt: Optional[T] = None


@functools.cache
def load_foregrounds(
    img_size: int,
    use_single_object: bool,
) -> dict[str, list[ImageWrapper]]:
    """Loads foregrounds from the HF base dataset."""
    foreground_images = cast(
        HFDataset,
        load_dataset(
            HF_BASE_DATASET_ID,
            "foregrounds",
            # There's only one split, other splits are created
            # via transformations
            split="train",
        ),
    )
    labels = set(foreground_images["label"])
    fgs: dict[str, list[ImageWrapper]] = {label: [] for label in labels}

    for example in foreground_images:
        example = cast(dict, example)
        label = example["label"]
        if use_single_object and len(fgs[label]) > 0:
            # Only use one base image per class. Different images per class
            # are created by applying transformations to the base image.
            continue
        image = example["image"]
        fgs[label].append(
            scale_object(
                ImageWrapper(
                    img=resize(image, img_size),
                    bb_x=0,
                    bb_y=0,
                    bb_width=img_size,
                    bb_height=img_size,
                ),
                factor=DEFAULT_OBJ_SCALE,
            )
        )

    print(f"{len(fgs)} foregrounds loaded.")
    return fgs


def generate_random_foregrounds(
    config: RandomForegroundsConfig,
    img_size: int,
) -> dict[str, list[ImageWrapper]]:
    class_imgs: dict[str, list[ImageWrapper]] = {}
    rng = np.random.default_rng(config.seed)
    for i in range(config.n_classes):
        pixels = rng.random((img_size, img_size, 3), dtype=np.float32)
        img = Image.fromarray(pixels, "RGB")
        class_imgs[str(i)] = [
            scale_object(
                ImageWrapper(
                    img,
                    0,
                    0,
                    img_size,
                    img_size,
                ),
                factor=DEFAULT_OBJ_SCALE,
            )
        ]
    return class_imgs


class ForegroundsDataset(Dataset[DataSample[Image.Image]]):
    def __init__(
        self,
        config: FGConfig,
        n_samples: int,
        sampling_seed: int,
        transforms_sampling_seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples

        self.np_rng = np.random.default_rng(sampling_seed)
        self.class_salt = self.np_rng.integers(low=0, high=int(1e6))
        if transforms_sampling_seed is None:
            transforms_sampling_seed = sampling_seed
        # Create a generator for sampling the transformations in the same way
        # no matter what other operations are using torch.random in the
        # meantime.
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(transforms_sampling_seed)

        if isinstance(config, RandomForegroundsConfig):
            self.n_classes = config.n_classes
            self.fgs = generate_random_foregrounds(
                config,
                img_size=config.img_size,
            )
            self._filter_foregrounds()
            fg_class_indices = list(range(self.n_classes))
        else:
            fg_class_indices = config.fg_class_indices
            self.n_classes = len(fg_class_indices)
            self.fgs = load_foregrounds(
                config.img_size,
                use_single_object=config.use_single_object,
            )
            if len(fg_class_indices) > len(self.fgs):
                raise ValueError(
                    f"Number of classes ({self.n_classes}) greater than "
                    f"the number of foreground categories ({len(self.fgs)})"
                )
            self._filter_foregrounds(fg_class_indices)

        self._init_transforms(
            config=config,
            fg_class_indices=fg_class_indices,
        )

    def _init_transforms(
        self,
        config: FGConfig,
        fg_class_indices: list[int],
    ) -> None:
        self.n_transform_candidates = None
        if isinstance(config.transforms, dict):
            self.fg_transforms = {
                i: ForegroundsDataset._compose_transforms(
                    config.transforms[class_idx],
                )
                for i, class_idx in enumerate(fg_class_indices)
            }
        else:
            self.fg_transforms = ForegroundsDataset._compose_transforms(
                config.transforms,
            )

    @staticmethod
    def _compose_transforms(
        config_transforms: TransformOptions,
    ) -> torch_transforms.Compose:
        fg_transforms = [
            get_transform(transform)
            if isinstance(transform, str)
            else transform
            for transform in config_transforms
        ]
        return torch_transforms.Compose(fg_transforms)

    def _filter_foregrounds(
        self,
        fg_class_indices: Optional[list[int]] = None,
    ) -> None:
        fg_key_list = list(self.fgs.keys())
        if fg_class_indices is not None:
            filtered_fg_classes = [fg_key_list[idx] for idx in fg_class_indices]
            filtered_fgs = {
                fg_class: self.fgs[fg_class] for fg_class in filtered_fg_classes
            }
        else:
            filtered_fgs = self.fgs
        self.fgs = filtered_fgs
        self.fg_classes: list[str] = list(filtered_fgs.keys())
        self.fg_imgs: list[list[ImageWrapper]] = list(filtered_fgs.values())

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self,
        idx: int,
        is_class_idx: bool = False,
    ) -> Transforms2DDataSample[Image.Image]:
        if is_class_idx:
            class_idx = idx
        else:
            class_idx = get_pseudorandom_idx(
                idx, self.n_classes, self.class_salt
            )
        class_fg_imgs = self.fg_imgs[class_idx]
        if len(self.fg_imgs) > 0:
            object_idx = get_pseudorandom_idx(
                idx, len(class_fg_imgs), salt=self.class_salt + class_idx
            )
        else:
            object_idx = 0
        fg_img = class_fg_imgs[object_idx]

        # Apply transformations
        if isinstance(self.fg_transforms, dict):
            instance_transforms = self.fg_transforms[class_idx]
        else:
            instance_transforms = self.fg_transforms
        with use_local_rng(self.torch_rng):
            # Use the random state from this dataset, instead of
            # Pytorch's global random state, to enable consistent
            # samples
            transform_res = instance_transforms(fg_img)

        assert self.n_transform_candidates is None
        fg_aug = transform_res.img

        return Transforms2DDataSample(
            input=fg_aug,
            target=torch.tensor(class_idx, requires_grad=False),
        )


@dataclass
class ImageBackgroundsConfig:
    img_size: int


@dataclass
class RandomBackgroundsConfig:
    seed: int
    img_size: int


@dataclass
class UniformBackgroundsConfig:
    color: list[float]
    img_size: int


BGConfig = Union[
    ImageBackgroundsConfig,
    RandomBackgroundsConfig,
    UniformBackgroundsConfig,
]


@functools.cache
def load_backgrounds(
    img_size: int,
) -> list[ImageWrapper]:
    """Loads background images.

    Args:
        img_size: size of the images.

    Returns:
        bgs: a list of the form [bg0, bg1, ...]
    """
    background_images = cast(
        HFDataset,
        load_dataset(
            HF_BASE_DATASET_ID,
            "backgrounds",
            # There's only one split
            split="train",
        ),
    )
    bgs: list[ImageWrapper] = []

    for example in background_images:
        example = cast(dict, example)
        image = example["image"]
        bg_img = resize(
            crop_image_to_square(image),
            img_size,
        )
        bgs.append(
            ImageWrapper(
                img=bg_img,
                bb_x=0,
                bb_y=0,
                bb_width=img_size,
                bb_height=img_size,
            )
        )
    print(f"{len(bgs)} backgrounds loaded.")
    return bgs


def generate_uniform_background(
    config: UniformBackgroundsConfig,
) -> list[ImageWrapper]:
    if config.color != [0, 0, 0]:
        raise NotImplementedError("Only black backgrounds implemented yet.")
    pixels = np.zeros((config.img_size, config.img_size, 3), dtype=np.int8)
    img = Image.fromarray(pixels, "RGB")
    return [ImageWrapper(img, 0, 0, config.img_size, config.img_size)]


class BackgroundsDataset(Dataset[DataSample[Image.Image]]):
    def __init__(
        self,
        config: BGConfig,
        n_samples: int,
        sampling_seed: int,
    ) -> None:
        self.n_samples = n_samples

        self.rng = np.random.default_rng(sampling_seed)
        self.bgs_salt = self.rng.integers(low=0, high=10 ^ 6)
        self.random_bg_rng = None
        self.img_size = config.img_size

        if isinstance(config, RandomBackgroundsConfig):
            self.random_bg_rng = np.random.default_rng(config.seed)
        elif isinstance(config, ImageBackgroundsConfig):
            self.bgs = load_backgrounds(config.img_size)
        elif isinstance(config, UniformBackgroundsConfig):
            self.bgs = generate_uniform_background(config)
        else:
            raise ValueError("Invalid background config provided")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> DataSample:
        if self.random_bg_rng is None:
            bg_idx = get_pseudorandom_idx(idx, len(self.bgs), self.bgs_salt)
            bg_img = self.bgs[bg_idx].img.copy()
        else:
            bg_idx = 0
            pixels = self.random_bg_rng.random(
                (self.img_size, self.img_size, 3),
                dtype=np.float32,
            )
            bg_img = Image.fromarray(pixels, "RGB")

        return DataSample(
            input=bg_img,
            target=torch.tensor(bg_idx, requires_grad=False),
        )


@dataclass
class Composite2DDatasetConfig:
    img_size: int
    sampling_seed: int
    normalize: bool = True
    class_from_foreground: bool = True
    # The probability with which to equalize foreground and background classes
    fg_bg_correlation: float = 0.0
    fg_availability: float = 1.0


class Composite2DDataset(Dataset[Transforms2DDataSample[torch.Tensor]]):
    def __init__(
        self,
        config: Composite2DDatasetConfig,
        foregrounds: Dataset,
        backgrounds: Dataset,
    ) -> None:
        assert config.fg_availability == 1.0 or not config.class_from_foreground
        self.config = config
        self.foregrounds = foregrounds
        self.backgrounds = backgrounds

        self.n_samples = min(len(foregrounds), len(backgrounds))
        # Used for sampling correlated foreground objects
        self.rng = np.random.default_rng(self.config.sampling_seed + 583)

        self._init_transforms()

    def _init_transforms(self) -> None:
        img_transforms: list[Callable] = [
            torch_transforms.ToTensor(),
        ]
        if self.config.normalize:
            img_transforms.append(
                torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        self.img_transforms = torch_transforms.Compose(img_transforms)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Transforms2DDataSample[torch.Tensor]:
        bg_instance = self.backgrounds[idx]
        if isinstance(bg_instance, DataSample):
            bg_img, bg_class = bg_instance.input, bg_instance.target
        else:
            bg_img = bg_instance[0]
            bg_class = bg_instance[1]
        bg_img = bg_img.copy()

        if self.config.fg_availability < 1.0:
            fg_availability_prop = self.rng.random()
            show_fg = fg_availability_prop < self.config.fg_availability
        else:
            show_fg = True

        if show_fg:
            if self.config.fg_bg_correlation > 0:
                fg_bg_sync_prop = self.rng.random()
                sync_fg_bg = fg_bg_sync_prop < self.config.fg_bg_correlation
                if sync_fg_bg:
                    # Sample a foreground object with the same class as the
                    # background object
                    if bg_class is None:
                        raise ValueError()
                    if isinstance(bg_class, torch.Tensor):
                        fg_idx, is_class_idx = int(bg_class.item()), True
                    else:
                        fg_idx, is_class_idx = int(bg_class), True
                else:
                    fg_idx, is_class_idx = idx, False
            else:
                fg_idx, is_class_idx = idx, False

            img, fg_class = _generate_fg_bg_img(
                self.foregrounds,
                self.config.img_size,
                fg_idx=fg_idx,
                is_class_idx=is_class_idx,
                bg_img=bg_img,
            )
            class_idx = (
                fg_class if self.config.class_from_foreground else bg_class
            )
        else:
            img = bg_img
            class_idx = bg_class

        x = self.img_transforms(img).float()
        target = (
            class_idx
            if isinstance(class_idx, torch.Tensor)
            else torch.tensor(class_idx, requires_grad=False)
        )
        return Transforms2DDataSample(
            input=x,
            target=target,
        )


def _generate_fg_bg_img(
    foregrounds: Dataset,
    img_size: int,
    fg_idx: int,
    is_class_idx: bool,
    bg_img: Image.Image,
) -> tuple[Image.Image, Optional[torch.Tensor]]:
    fg_instance = foregrounds[fg_idx, is_class_idx]

    fg_class: Optional[torch.Tensor]
    if isinstance(fg_instance, Transforms2DDataSample):
        fg_img = fg_instance.input
        fg_class = fg_instance.target
    else:
        # It's a tuple
        fg_img = fg_instance[0]
        fg_class = fg_instance[1]

    fg_obj = resize(fg_img, img_size)
    start_x, start_y = calc_top_left_coordinates(
        fg_obj,
        img_size,
        0.5,
        0.5,
    )
    bg_img.paste(fg_obj, box=(start_x, start_y), mask=fg_obj)
    return bg_img, fg_class


@contextlib.contextmanager
def use_local_rng(rng: torch.Generator) -> Iterator[None]:
    global_random_state = torch.random.get_rng_state()
    torch.random.set_rng_state(rng.get_state())
    yield
    rng.set_state(torch.random.get_rng_state())
    torch.random.set_rng_state(global_random_state)
