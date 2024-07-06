from dataclasses import dataclass
from typing import Callable, Union, cast, overload

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional


@dataclass
class ImageWrapper:
    """Wrapper for images that tracks bounding box information
    for the actual object inside the image"""

    img: Image.Image
    bb_x: int
    bb_y: int
    bb_width: int
    bb_height: int


@overload
def scale_object(x: ImageWrapper, factor: float) -> ImageWrapper:
    ...


@overload
def scale_object(x: Image.Image, factor: float) -> Image.Image:
    ...


def scale_object(x, factor: float):
    if isinstance(x, ImageWrapper):
        img = x.img
    else:
        img = x
    w, h = img.size
    sw, sh = int(round(w * factor)), int(round(h * factor))
    scaled_img = img.resize((sw, sh)).convert("RGBA")
    start_x = (w - sw) // 2
    start_y = (h - sh) // 2
    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg.paste(scaled_img, box=(start_x, start_y))
    if isinstance(x, ImageWrapper):
        return ImageWrapper(
            img=bg,
            bb_x=x.bb_x + start_x,
            bb_y=x.bb_y + start_y,
            bb_width=int(round(x.bb_width * factor)),
            bb_height=int(round(x.bb_height * factor)),
        )
    else:
        return bg


def translate_object(
    x: ImageWrapper, target_x: int, target_y: int
) -> ImageWrapper:
    img = x.img
    start_x = target_x - x.bb_x
    start_y = target_y - x.bb_y

    bg = Image.new("RGBA", (img.width, img.height), (0, 0, 0, 0))
    bg.paste(img, box=(start_x, start_y))
    if isinstance(x, ImageWrapper):
        return ImageWrapper(
            img=bg,
            bb_x=target_x,
            bb_y=target_y,
            bb_width=x.bb_width,
            bb_height=x.bb_height,
        )


class RandomScale(torch.nn.Module):
    def __init__(self, min_size: float) -> None:
        super().__init__()
        self.min_size = min_size

    @overload
    def forward(self, x: ImageWrapper) -> ImageWrapper:
        ...

    @overload
    def forward(self, x: Image.Image) -> Image.Image:
        ...

    def forward(self, x: Union[Image.Image, ImageWrapper]):
        factor = (
            float(torch.rand(1).item()) * (1 - self.min_size) + self.min_size
        )
        return scale_object(x, factor)


class RandomTranslate(torch.nn.Module):
    def __init__(
        self,
        no_bb_translate_range: float = 0.3,
    ) -> None:
        super().__init__()
        self.non_bb_translate_range = no_bb_translate_range

    @overload
    def forward(self, x: ImageWrapper) -> ImageWrapper:
        ...

    @overload
    def forward(self, x: Image.Image) -> Image.Image:
        ...

    def forward(self, x):
        if isinstance(x, ImageWrapper):
            img = x.img
            wrapped_x = x
            translate_range_x = img.width - x.bb_width
            translate_range_y = img.height - x.bb_height
        else:
            img = x
            translate_range_x = img.width * self.non_bb_translate_range
            translate_range_y = img.height * self.non_bb_translate_range
            wrapped_x = ImageWrapper(
                img=img,
                bb_x=int(-translate_range_x / 2),
                bb_y=int(-translate_range_y / 2),
                bb_width=x.width,
                bb_height=x.height,
            )
        pos_x = int(torch.rand(1).item() * translate_range_x)
        pos_y = int(torch.rand(1).item() * translate_range_y)

        translated_x = translate_object(wrapped_x, pos_x, pos_y)
        if isinstance(x, ImageWrapper):
            return translated_x
        else:
            return translated_x.img


class RandomNoise(torch.nn.Module):
    """Randomly add Gaussian Noise to the image"""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x: Image.Image) -> Image.Image:
        if not _should_apply_transform(self.p):
            return x
        return self._add_noise(x)

    def _add_noise(self, x: Image.Image) -> Image.Image:
        w, h = x.size
        noise = torch.clip(torch.randn(3, h, w) * self.std + self.mean, 0, 1)
        x_tensor = functional.to_tensor(x)
        x_noisy = x_tensor + noise
        return functional.to_pil_image(x_noisy)


# Based on: https://github.com/bethgelab/imagecorruptions/blob/master/imagecorruptions/corruptions.py # noqa: W505
class RandomPixelate(torch.nn.Module):
    """Randomly pixelate the image"""

    def __init__(self, shrink_to: float = 0.5, p: float = 0.5) -> None:
        super().__init__()
        self.shrink_to = shrink_to
        self.p = p

    def forward(self, x: Image.Image) -> Image.Image:
        if not _should_apply_transform(self.p):
            return x
        return self._pixelate(x)

    def _pixelate(self, x: Image.Image) -> Image.Image:
        w, h = x.size
        x = x.resize(
            (int(w * self.shrink_to), int(h * self.shrink_to)), Image.BOX
        )
        x = x.resize((w, h), Image.NEAREST)
        return x


class RandomErase(torch.nn.Module):
    """Randomly erase a square from the image"""

    def __init__(self, square_size: int, p: float = 0.5) -> None:
        super().__init__()
        self.square_size = square_size
        self.p = p

    def forward(self, x: Image.Image) -> Image.Image:
        if not _should_apply_transform(self.p):
            return x
        return self._erase(x)

    def _erase(self, x: Image.Image) -> Image.Image:
        w, h = x.size
        assert w > self.square_size and h > self.square_size
        start_x = int(torch.rand(1).item() * (w - self.square_size))
        start_y = int(torch.rand(1).item() * (h - self.square_size))
        removed_area = Image.new(
            "1", (self.square_size, self.square_size), (0,)
        )
        mask = Image.new("1", (w, h), (1,))
        mask.paste(removed_area, box=(start_x, start_y))
        bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bg.paste(x, box=(0, 0), mask=mask)
        return bg


def _should_apply_transform(p: float) -> bool:
    return torch.rand(1).item() < p


def _with_transparency(
    transform: Callable[[Image.Image], Image.Image],
) -> Callable[[Image.Image], Image.Image]:
    def _transform(x: Image.Image) -> Image.Image:
        if x.mode == "RGB":
            return transform(x)

        transformed_x = transform(x.convert("RGB"))
        size_x, size_y = x.size
        bg = Image.new("RGBA", (size_x, size_y), (0, 0, 0, 0))
        bg.paste(transformed_x, box=(0, 0), mask=x)
        return bg

    return _transform


def with_image_wrapper(
    wrapped_transform: Callable[[Image.Image], Image.Image],
) -> Callable[[ImageWrapper], ImageWrapper]:
    def _transform(x: ImageWrapper) -> ImageWrapper:
        return ImageWrapper(
            img=wrapped_transform(x.img),
            bb_x=x.bb_x,
            bb_y=x.bb_y,
            bb_width=x.bb_width,
            bb_height=x.bb_height,
        )

    return _transform


TransformFunc = Callable[[Image.Image], Image.Image]

_SHEAR_RANGE = 50
_GEOMETRIC_TRANSFORM_FUNCS: dict[str, TransformFunc] = {
    "scale": RandomScale(0.4),
    "translate": RandomTranslate(),  # 0.5
    "v_flip": transforms.RandomVerticalFlip(),
    "h_flip": transforms.RandomHorizontalFlip(),
    "rotate": transforms.RandomRotation(degrees=(0, 360)),
    "shear": transforms.RandomAffine(
        degrees=0,
        shear=(-_SHEAR_RANGE, _SHEAR_RANGE, -_SHEAR_RANGE, _SHEAR_RANGE),
    ),
}
_PHOTOMETRIC_TRANSFORM_FUNCS: dict[str, TransformFunc] = {
    "hue": _with_transparency(
        transforms.ColorJitter(hue=0.5),
    ),
    "brightness": _with_transparency(
        transforms.ColorJitter(brightness=1.0),
    ),
    "grayscale": _with_transparency(
        transforms.RandomGrayscale(p=0.5),
    ),
    "posterize": _with_transparency(
        transforms.RandomPosterize(bits=1),
    ),
    "invert": _with_transparency(
        transforms.RandomInvert(),
    ),
    "sharpen": transforms.RandomAdjustSharpness(sharpness_factor=7),
}
_CORRUPTION_TRANSFORM_FUNCS = {
    "blur": transforms.GaussianBlur(7, sigma=(0.1, 1.5)),
    "noise": _with_transparency(RandomNoise()),
    "pixelate": RandomPixelate(shrink_to=0.3),
    "elastic": transforms.RandomApply(
        torch.nn.ModuleList([transforms.ElasticTransform(alpha=150.0)]),
        p=0.5,
    ),
    "erasing": RandomErase(square_size=14),
    "contrast": _with_transparency(transforms.ColorJitter(contrast=1.0)),
}
_TRANSFORM_FUNCS = (
    _GEOMETRIC_TRANSFORM_FUNCS
    | _PHOTOMETRIC_TRANSFORM_FUNCS
    | _CORRUPTION_TRANSFORM_FUNCS
)


def none_transform(img: Image.Image) -> Image.Image:
    return img


GEOMETRIC_TRANSFORMS = list(_GEOMETRIC_TRANSFORM_FUNCS.keys())
PHOTOMETRIC_TRANSFORMS = list(_PHOTOMETRIC_TRANSFORM_FUNCS.keys())
CORRUPTION_TRANSFORMS = list(_CORRUPTION_TRANSFORM_FUNCS.keys())
TRANSFORMS = list(_TRANSFORM_FUNCS.keys())


def get_transform(
    transform_name: str,
) -> Callable[[ImageWrapper], ImageWrapper]:
    transform = get_unwrapped_transform(transform_name)
    if isinstance(transform, RandomScale) or isinstance(
        transform, RandomTranslate
    ):
        return transform
    else:
        return with_image_wrapper(transform)


def get_unwrapped_transform(
    transform_name: str,
) -> Callable[[Image.Image], Image.Image]:
    if transform_name == "none":
        return none_transform
    else:
        return _TRANSFORM_FUNCS[transform_name]
