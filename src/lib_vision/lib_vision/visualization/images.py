import math
from typing import Iterable, Literal, Optional, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from lib_vision.data.loading import get_data_type_loader
from lib_vision.data.wrappers.data_sample import DataSample


DataType = Literal["train", "val", "test"]


def show_dataset_samples(
    data: L.LightningDataModule,
    n_samples: int,
    data_type: DataType,
    figsize=(20, 10),
    show_titles: bool = False,
    show_figure: bool = True,
) -> Figure:
    vis_loader = get_data_type_loader(data, data_type)
    fig = show_images(
        vis_loader,
        n_images=n_samples,
        figsize=figsize,
        show_titles=show_titles,
        show_figure=show_figure,
    )
    return fig


def show_images(
    images: Union[
        DataLoader[DataSample],
        list[torch.Tensor],
        list[np.ndarray],
    ],
    n_images: int = 8,
    titles: Optional[Iterable[str]] = None,
    cmaps: Optional[list] = None,
    n_cols: int = 8,
    figsize: tuple[int, int] = (20, 10),
    title_fontsize: int = 30,
    show_titles: bool = True,
    show_figure: bool = True,
) -> Figure:
    """
    Shows a grid of images with optional labels.
    Roughly inspired by:
    https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

    Parameters:
    ----------
    images: pytorch DataLoader, list of pytorch Tensors or numpy ndarrays
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    """

    if isinstance(images, DataLoader):
        dataiter = iter(images)
        batch = next(dataiter)
        image_list = batch.input
        labels = [str(yi.item()) for yi in batch.target]
        # TODO: make sure the batch has enough data available
        image_list = image_list[:n_images].detach().cpu().numpy()
        image_list = np.transpose(image_list, (0, 2, 3, 1))
        if titles is None:
            # titles = [label.item() for label in labels[:n_images]]
            titles = labels
    else:
        raise NotImplementedError

    n_cols = min(n_images, n_cols)
    n_rows = math.ceil(n_images / n_cols)

    # Create a grid of subplots.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    list_axes: list[Axes]
    if isinstance(axes, np.ndarray):
        list_axes = [ax for ax in axes.flat]
    elif isinstance(axes, Axes):
        list_axes = [axes]
    else:
        raise ValueError(f"Unknown axes type {type(axes)}")

    for i, (img, title, ax) in enumerate(zip(image_list, titles, list_axes)):
        cmap = (
            cmaps[i]
            if cmaps is not None
            else (None if _img_is_color(img) else "gray")
        )
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if show_titles:
            ax.set_title(title, fontsize=title_fontsize)
        ax.axis("off")

    for i in range(n_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    if show_figure:
        _ = plt.show()
    return fig


def _img_is_color(img: np.ndarray) -> bool:
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True
    return False
