from pathlib import Path
from typing import Collection, Literal, Optional, Union, cast

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from IPython.display import Markdown as md
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from lib_project.experiment import ExperimentResult, load_results
from lib_project.notebook import publish_notebook

from .data import CLASS_RELATIONSHIPS, SAMPLE_COUNTS_FT, SAMPLE_COUNTS_PT
from .experiment import EXP_NAME, ExperimentConfig
from .experiment import ExperimentResult as TvOExperimentResult


TRANSFER_TYPE_DESC = [
    ("single obj", "single_object"),
    ("multi obj", "multi_object"),
]


TvOResult = ExperimentResult[ExperimentConfig, TvOExperimentResult]


def load(
    config_name: str,
    seed_ids: list[int],
) -> list[TvOResult]:
    results = load_results(
        EXP_NAME,
        config_name,
        seed_ids,
        ExperimentConfig,
        TvOExperimentResult,
    )
    return _convert_ft_samples_to_pt_format(results)


def publish(
    notebook: str,
) -> None:
    # Use a random postfix to make it harder to guess the file name
    if notebook == "t2d_eval":
        output_path = f"experiments/{EXP_NAME}/t2d_eval_4274fd.html"
    elif notebook == "cifar_eval":
        output_path = f"experiments/{EXP_NAME}/cifar_eval_f4352s.html"
    else:
        raise ValueError(f"Unknown notebook: {notebook}")
    notebook_path = f"./experiments/{EXP_NAME}/notebooks/{notebook}.ipynb"
    publish_notebook(
        notebook_path,
        output_path,
    )


def _convert_ft_samples_to_pt_format(
    results: list[TvOResult],
) -> list[TvOResult]:
    """The 'finetuning_samples' category varies parameters during
    finetuning, whereas the other categories vary them during pretraining.
    This function converts the results from the 'finetuning_samples'
    to be compatible with the other categories.
    """
    for res in results:
        if res.config.variation_category not in {"finetuning_samples", "none"}:
            continue
        if res.config.data.data_type == "t2d":
            dataset_replacement = "single_object"
        else:
            dataset_replacement = "default"
        for training_type in [
            "same_transforms",
            "disjoint_transforms",
        ]:
            training_perf = getattr(
                res.value, training_type
            ).training_performance
            transfer_perf = getattr(
                res.value, training_type
            ).transfer_performance

            # Swap the "model" and "dataset" levels
            dataset_idx_values = transfer_perf.index.get_level_values("dataset")
            split_idx_values = transfer_perf.index.get_level_values("split")

            replacement_dataset_values = [dataset_replacement] * len(
                dataset_idx_values
            )

            new_index = pd.MultiIndex.from_arrays(
                [
                    dataset_idx_values,
                    replacement_dataset_values,
                    split_idx_values,
                ],
                names=["model", "dataset", "split"],
            )
            transfer_perf.index = new_index

            training_perf = getattr(
                res.value, training_type
            ).training_performance
            training_models = training_perf.index.get_level_values(
                "model"
            ).unique()
            training_datasets = training_perf.index.get_level_values(
                "dataset"
            ).unique()
            assert len(training_models) == 1
            assert len(training_datasets) == 1
            training_accuracy = training_perf.loc[("default", "default")]
            num_variations = len(dataset_idx_values.unique())
            index_values = list(
                zip(
                    dataset_idx_values.unique(),
                    ["default"] * num_variations,
                )
            )
            new_training_perf = pd.concat(
                [training_accuracy] * num_variations,
                keys=index_values,
                names=["model", "dataset"],
            )
            getattr(
                res.value, training_type
            ).training_performance = new_training_perf

    return results


FILE_DATASET_TYPES = {
    "t2d": "t2d",
    "cifar10": "ci10",
    "cifar100": "ci100",
}
FILE_VARIATION_CATEGORIES = {
    "class_rel": "class-rel",
    "arch": "arch",
    "pt_samples": "pt-samples",
    "ft_samples": "ft-samples",
    # Models
    "resnet-18": "resnet-18",
    "resnet-50": "resnet-50",
    "densenet-121": "densenet-121",
    "vgg-11": "vgg-11",
    "vit": "vit",
}
X_LABELS = {
    "class_rel": "Class Rel.",
    "arch": "Architecture",
    "pt_samples": "PT Samples",
    "ft_samples": "FT Samples",
    # Models
    "resnet-18": "resnet-18",
    "resnet-50": "resnet-50",
    "densenet-121": "densenet-121",
    "vgg-11": "vgg-11",
    "vit": "vit",
}


def show_results(
    dataset_type: str,
    variation_categories: list[str],
    ft_type: str,
    seeds: list[int],
    # multi-, single-object or default (in the case of RW-datasets like CIFAR)
    object_cardinality: str | list[str],
    variation_descriptions: dict[str, str] | None = None,
    show_legend_for: Collection[str] = [],
    show_diff_legend: bool = False,
    show_hyperparameters: bool = False,
) -> dict[str, plt.Figure]:
    figures = {}
    results = {}

    for var_cat in variation_categories:
        dataset_prefix = FILE_DATASET_TYPES[dataset_type]
        var_cat_postfix = FILE_VARIATION_CATEGORIES[var_cat]
        var_cat_results = load(
            (
                f"{dataset_prefix}_m-{var_cat_postfix}_ft-{ft_type}"
                if show_hyperparameters
                else f"{dataset_prefix}_v-{var_cat_postfix}_ft-{ft_type}"
            ),
            seeds,
        )

        display(md(f"### Transforms vs {X_LABELS[var_cat]}"))
        if variation_descriptions is not None:
            display(md(variation_descriptions[var_cat]))

        var_cat_fig, var_cat_legend = plot_performance_curves(
            var_cat_results,
            dataset_type=dataset_type,
            variation_category=var_cat,
            x_label=X_LABELS[var_cat],
            object_cardinality=object_cardinality,
            show_legend=var_cat in show_legend_for,
        )
        plt.show(var_cat_fig)
        figures[var_cat] = var_cat_fig
        results[var_cat] = var_cat_results

    display(md("### Performance differences"))
    difference_fig, difference_legend = plot_performance_differences(
        results,
        dataset_type,
        object_cardinality,
        show_legend=show_diff_legend,
    )
    plt.show(difference_fig)
    figures["diff"] = difference_fig

    return figures


PERF_COL = "Accuracy"
PERF_TYPE_COL = "Task"
TRANSFORM_REL_COL = "Transform"
VAR_VALUE_COL = "var_value"


def plot_performance_curves(
    results: list[TvOResult],
    dataset_type: str,
    variation_category: str,
    object_cardinality: str,
    x_label: str,
    show_legend: bool = False,
) -> tuple[plt.Figure, Union[plt.Figure, None]]:
    combined_results = _combine_results(
        variation_category, dataset_type, results, object_cardinality
    )

    combined_results["perf_type"] = combined_results.apply(
        lambda row: row[TRANSFORM_REL_COL] + ", " + row[PERF_TYPE_COL],
        axis=1,
    )
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#85c0ea",
        "#ffb370",
    ]

    fig, axes = plt.subplots(1, figsize=(4, 3), squeeze=False)
    perf_plot = axes[0][0]
    sns.lineplot(
        data=combined_results,
        x=VAR_VALUE_COL,
        y=PERF_COL,
        style=PERF_TYPE_COL,
        hue="perf_type",
        legend=show_legend,
        ax=perf_plot,
        markers=["o", "s"],
        palette=color_palette,
    )
    if show_legend:
        # Customize the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = [label for label in set(labels) if "," in label]
        custom_handles = [
            Line2D(
                [0],
                [0],
                color=handles[labels.index(label)].get_color(),
                marker="o" if "transfer" in label else "s",
                linestyle="--" if "training" in label else "-",
            )
            for label in unique_labels
        ]
        plt.legend(custom_handles, unique_labels, loc="best")
    perf_plot.set_xlabel(x_label)
    perf_plot.set_ylabel("Accuracy")

    fig.tight_layout()
    if not show_legend:
        figlegend = plt.figure(figsize=(3, 2))
    else:
        figlegend = None
    return fig, figlegend


def plot_performance_differences(
    results: dict[str, list[TvOResult]],
    dataset_type: str,
    object_cardinality: str,
    add_y_top: float = 0.05,
    show_legend: bool = False,
) -> tuple[plt.Figure, Union[plt.Figure, None]]:
    category_results = {
        category: _combine_results(
            category, dataset_type, category_results, object_cardinality
        )
        for category, category_results in results.items()
    }

    fig, axes = plt.subplots(1, figsize=(4, 3), squeeze=False)
    plot = axes[0][0]
    lines = []
    colors = sns.color_palette("tab10", n_colors=2)
    for i, (cat, cat_res) in enumerate(category_results.items()):
        if cat == "ft_samples":
            continue
        transform_type_means = {
            transform_type: (
                cat_res.loc[
                    (cat_res[TRANSFORM_REL_COL] == transform_type)
                    & (cat_res[PERF_TYPE_COL] == "transfer")
                ]
                .groupby("var_value")[PERF_COL]
                .mean()
            )
            for transform_type in ["same", "disjoint"]
        }

        base_idx = 4 * i
        for j, transform_type in enumerate(["disjoint", "same"]):
            min_value = transform_type_means[transform_type].min()
            max_value = transform_type_means[transform_type].max()
            mean_value = transform_type_means[transform_type].mean()
            line = _plot_bar(
                plot,
                base_idx + 2 * j,
                min_value,
                max_value,
                color=colors[0] if transform_type == "same" else colors[1],
                mean=mean_value,
                label=f"within {transform_type}",
            )
            if i == 0:
                lines.append(line)

        same_mean = transform_type_means["same"].mean()
        disjoint_mean = transform_type_means["disjoint"].mean()
        line = _plot_bar(
            plot,
            base_idx + 1,
            min(same_mean, disjoint_mean),
            max(same_mean, disjoint_mean),
            color="black",
            label="same vs. disjoint",
        )
        if i == 0:
            lines.append(line)

    plot.xaxis.set_major_locator(MultipleLocator(4, offset=1))
    plot.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: (
                X_LABELS[list(category_results.keys())[int(x // 4)]]
                if (0 <= x < 4 * len(category_results))
                else ""
            )
        )
    )
    plot.set_ylabel("Accuracy")
    if show_legend:
        plot.legend(
            handles=lines,
            title="Differences",
            loc="upper right",
            prop={"size": 8},
        )
    y_min, y_max = plot.get_ylim()
    plot.set_ylim(y_min, y_max + add_y_top)
    fig.tight_layout()
    return fig, None


def _plot_bar(
    plot: plt.Axes,
    x_pos: int,
    min_value: float,
    max_value: float,
    color: str,
    label: str,
    mean: float | None = None,
) -> plt.Line2D:
    line = plot.vlines(
        x_pos,
        min_value,
        max_value,
        color=color,
        linewidth=2,
        label=label,
    )
    overshoot = 0.6 if mean is None else 0.1
    plot.hlines(
        min_value,
        x_pos - overshoot,
        x_pos + 0.1,
        color=color,
        linewidth=0.5,
    )
    plot.hlines(
        max_value,
        x_pos - 0.1,
        x_pos + overshoot,
        color=color,
        linewidth=0.5,
    )
    # Add a mean marker
    if mean is not None:
        plot.plot(
            x_pos,
            mean,
            marker="o",
            color=color,
        )
    diff = max_value - min_value
    plot.text(
        x_pos,
        max_value + 0.01,
        f"{diff:.2f}",
        ha="center",
    )
    return line


def _combine_results(
    variation_category: str,
    dataset_type: str,
    results: list[TvOResult],
    object_cardinality: str,
) -> pd.DataFrame:
    all_results: list[pd.DataFrame] = []
    transfer_results = [
        _to_summary(
            variation_category,
            dataset_type,
            res.value.same_transforms.transfer_performance,
            res.value.disjoint_transforms.transfer_performance,
            "accuracy",
            object_cardinality,
        )
        for res in results
    ]
    _add_results(transfer_results, all_results, "transfer")
    training_results = [
        _to_summary(
            variation_category,
            dataset_type,
            res.value.same_transforms.training_performance,
            res.value.disjoint_transforms.training_performance,
            "accuracy",
        )
        for res in results
    ]
    _add_results(training_results, all_results, "training")
    combined_results = pd.concat(all_results)
    return combined_results


def _add_results(
    results: list[pd.DataFrame],
    all_results: list[pd.DataFrame],
    perf_type: str,
) -> None:
    dtypes = {
        VAR_VALUE_COL: "str",
        PERF_COL: "float",
        TRANSFORM_REL_COL: "str",
        PERF_TYPE_COL: "str",
    }
    for res in results:
        n_res = len(res)
        all_results.append(
            pd.DataFrame(
                {
                    VAR_VALUE_COL: res.index,
                    PERF_COL: res["same_transforms"],
                    TRANSFORM_REL_COL: ["same"] * n_res,
                    PERF_TYPE_COL: [perf_type] * n_res,
                }
            ).astype(dtypes)
        )
        all_results.append(
            pd.DataFrame(
                {
                    VAR_VALUE_COL: res.index,
                    PERF_COL: res["disjoint_transforms"],
                    TRANSFORM_REL_COL: ["disjoint"] * n_res,
                    PERF_TYPE_COL: [perf_type] * n_res,
                }
            ).astype(dtypes)
        )


ARCHICTURE_REMAPPING = {
    "resnet-18": "ResNet18",
    "resnet-50": "ResNet50",
    "densenet-121": "DN121",
    "vgg-11": "VGG11",
    "vit": "ViT",
}


def _to_summary(
    variation_category: str,
    dataset_type: str,
    same_transforms_res: pd.DataFrame,
    disjoint_transforms_res: pd.DataFrame,
    target_col: str,
    # multi-, single-object or default (in the case of RW-datasets like CIFAR)
    object_cardinality: Optional[str] = None,
) -> pd.DataFrame:
    if variation_category == "arch":
        same_transforms_res = same_transforms_res.rename(
            index=ARCHICTURE_REMAPPING
        )
        disjoint_transforms_res = disjoint_transforms_res.rename(
            index=ARCHICTURE_REMAPPING
        )
    if object_cardinality is None:
        res_df = pd.DataFrame(
            {
                "same_transforms": same_transforms_res[target_col],
                "disjoint_transforms": disjoint_transforms_res[target_col],
            }
        )
    else:
        res_df = pd.DataFrame(
            {
                "same_transforms": same_transforms_res.loc[
                    (slice(None), object_cardinality), target_col
                ],
                "disjoint_transforms": disjoint_transforms_res.loc[
                    (slice(None), object_cardinality), target_col
                ],
            }
        )
    target_order = _get_target_order(variation_category, dataset_type)
    reindexed_df = res_df.loc[
        res_df.index.get_level_values("split") == "test"
    ].droplevel(["dataset", "split"], axis=0)
    if target_order is not None:
        reindexed_df = reindexed_df.reindex(
            index=target_order,
            level="model",
        )
    return reindexed_df


def _get_target_order(
    variation_category: str, dataset_type: str
) -> list[str] | None:
    if variation_category == "class_rel":
        target_order = CLASS_RELATIONSHIPS
    elif variation_category == "arch":
        target_order = list(ARCHICTURE_REMAPPING.values())
    elif variation_category in ["pt_samples", "ft_samples"]:
        if dataset_type.startswith("t2d"):
            dataset_type = "t2d"
        if variation_category == "pt_samples":
            dataset_type_sample_counts = SAMPLE_COUNTS_PT[dataset_type]
        else:
            dataset_type_sample_counts = SAMPLE_COUNTS_FT[dataset_type]
        target_order = [
            str(n_samples) for n_samples in dataset_type_sample_counts
        ]
    else:
        target_order = None
    return target_order


def plot_training_curves(
    config_name: str,
    seed_id: int,
    transform_relationship: str,
    task_name: str,
    variation_value: str,
    dataset: str,
    base_path: Path = Path(f"../artifacts/{EXP_NAME}"),
) -> None:
    logs_path = (
        base_path
        / config_name
        / f"sid_{seed_id}"
        / transform_relationship
        / task_name
        / variation_value
        / dataset
        / "csv_logs/version_0"
        / "metrics.csv"
    )
    logs = pd.read_csv(logs_path)
    display(logs)
