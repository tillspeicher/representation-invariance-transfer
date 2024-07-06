import io
from pathlib import Path

from PIL import Image


def resize(
    obj: Image.Image,
    img_size: int,
) -> Image.Image:
    """Resizes images to `img_size`*`img_size` size."""
    resized_obj = obj.copy()
    return resized_obj.resize(
        (
            img_size,
            img_size,
        ),
        Image.BILINEAR,
    )


def crop_image_to_square(img: Image.Image) -> Image.Image:
    """Crops image to the largest square that fits inside img.

    Crops from the top left corner.

    Args:
        img: image of type PIL image, e.g. PIL.JpegImagePlugin.JpegImageFile.

    Returns:
        Square image of same type as input image.
    """
    side_length = min(img.height, img.width)
    return img.crop((0, 0, side_length, side_length))


def calc_top_left_coordinates(
    fg: Image.Image,
    img_size: int,
    x_coord: float,
    y_coord: float,
) -> tuple[int, int]:
    """Returns coordinates of top left corner of object.

    Input coordinates are coordinates of centre of object scaled in the range
    [0, 1].

    Args:
        fg: PIL image. Foreground image.
        bg: PIL image. Background image.
        x_coord: central x-coordinate of foreground object scaled between
            0 and 1.
        0 = leftmost coordinate of image, 1 = rightmost coordinate of image.
        y_coord: central y-coordinate of foreground object scaled between
            0 and 1.
        0 = topmost coordinate of image, 1 = bottommost coordinate of image.
    """
    x_coord = int(x_coord * img_size)
    y_coord = int(y_coord * img_size)
    # x_coord, y_coord should be at the centre of the object.
    x_coord_start = int(x_coord - fg.width * 0.5)
    y_coord_start = int(y_coord - fg.height * 0.5)

    return x_coord_start, y_coord_start
