from pathlib import Path

from IPython.display import Markdown as md
from IPython.display import display

from lib_vision.visualization.images import show_dataset_samples
from transforms_2d import TRANSFORMS, Transforms2DConfig, create_t2d_dataset


def plot_transforms(
    config: Transforms2DConfig,
    plot_items: list[tuple[str, list[int], int]],
    plots_dir: Path,
    figsize: tuple[int, int],
    use_single_object: bool = True,
):
    for transform, objects, seed in plot_items:
        config.sampling_seed = seed
        dataset = create_t2d_dataset(
            transforms=[transform],
            foregrounds=objects,
            config=config,
            normalize=False,
            use_single_object=use_single_object,
        )
        display(md(f"## {transform} examples"))
        fig = show_dataset_samples(
            dataset,
            n_samples=config.n_test_samples,
            data_type="test",
            figsize=figsize,
            show_titles=False,
        )
        if plots_dir is not None:
            fig.savefig(plots_dir / f"{transform}.png")
