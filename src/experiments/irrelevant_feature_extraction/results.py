from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

from lib_project.analysis.aggregate import summarize_df
from lib_project.experiment import ExperimentResult, load_results
from lib_project.notebook import publish_notebook

from .data import (
    AVAILABILITY_VALUES,
    MIXED_CIFAR_AVAILABILITY_KEY,
    MIXED_CIFAR_COR_KEY,
    MIXED_OBJECTS_KEY,
    OBJECTS_ONLY_KEY,
)
from .experiment import (
    CIFAR_ONLY_KEY,
    EXP_NAME,
    MIXED_CIFAR_KEY,
    ExperimentConfig,
)
from .experiment import ExperimentResult as IFEExperimentResult


IFEResult = ExperimentResult[ExperimentConfig, IFEExperimentResult]


def load(
    config_name: str,
    seed_ids: list[int],
) -> list[IFEResult]:
    return load_results(
        EXP_NAME,
        config_name,
        seed_ids,
        ExperimentConfig,
        IFEExperimentResult,
    )


def publish() -> None:
    # Use a random postfix to make it harder to guess the file name
    output_path = f"experiments/{EXP_NAME}/results_52384fj3.html"
    publish_notebook(
        f"./experiments/{EXP_NAME}/eval.ipynb",
        output_path,
    )


def show_results(
    results: list[IFEResult],
) -> None:
    print("In distribution/training performance:")
    training_res = summarize_df(
        [res.value.training_performance for res in results]
    )
    display(training_res)

    print("Transfer performance:")
    transfer_res = summarize_df(
        [res.value.core_ft_performance for res in results]
    )
    display(transfer_res)


def combine_seed_results(
    results: list[IFEResult],
) -> pd.DataFrame:
    combined_performance = []
    for i, result in enumerate(results):
        transfer_perf = result.value.core_ft_performance
        cols = transfer_perf.columns
        new_cols = pd.MultiIndex.from_product([[i], cols])
        recol_perf = transfer_perf.copy()
        recol_perf.columns = new_cols
        combined_performance.append(recol_perf)
    return pd.concat(combined_performance, axis=1)


DATASET_NAMES = {
    CIFAR_ONLY_KEY: "C & C",
    MIXED_CIFAR_KEY: "C + O & C",
    MIXED_OBJECTS_KEY: "C + O & O",
    OBJECTS_ONLY_KEY: "O & O",
}


def print_comp_table(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    column_names: Sequence[str] = list(DATASET_NAMES.keys()),
) -> None:
    for model_key, model_name in DATASET_NAMES.items():
        model_means = means.loc[model_key]
        model_stds = stds.loc[model_key]

        values = []
        for dataset_key in column_names:
            if (
                model_means.index.nlevels > 1
                and "split" in model_means.index.names
            ):
                loc_key = (dataset_key, "test")
            else:
                loc_key = dataset_key
            dataset_mean = model_means.loc[loc_key]
            dataset_std = model_stds.loc[loc_key]

            values.append(f"${dataset_mean:.2f}_{{\\pm {dataset_std:.2f}}}$")
        print(f"& {model_name} & {' & '.join(values)} \\\\")


def print_invariances(
    results: list[IFEResult],
) -> None:
    combined_invariances = _combine_invariance_results(results)
    grouped_invariances = combined_invariances.astype(float).T.groupby(
        level=["Model"]
    )
    mean_invariances = grouped_invariances.mean().T
    std_invariances = grouped_invariances.std().T
    print("Mean invariances:")
    display(mean_invariances)
    print("Standard deviation of invariances:")
    display(std_invariances)

    print_comp_table(
        mean_invariances,
        std_invariances,
        ["cifar", "objects"],
    )

    combined_performances = combine_seed_results(results)
    performances = (
        combined_performances.loc[
            combined_performances.index.get_level_values("split") == "test"
        ]
        .droplevel("split")
        .mean(axis=1)
        .astype(float)
    )
    perf_idx = performances.index
    cifar_corr = mean_invariances["cifar"].corr(
        performances.loc[
            perf_idx.get_level_values("dataset") == "cifar_mixed"
        ].droplevel(["dataset"])
    )
    objects_corr = mean_invariances["objects"].corr(
        performances.loc[
            perf_idx.get_level_values("dataset") == "objects_mixed"
        ].droplevel(["dataset"])
    )
    print("Correlation between invariances and performance:")
    print("CIFAR correlation:", cifar_corr)
    print("Objects correlation:", objects_corr)


def _combine_invariance_results(
    results: list[IFEResult],
) -> pd.DataFrame:
    combined_invariances = []
    for i, result in enumerate(results):
        invariance = result.value.invariances["l2"]
        cols = invariance.columns
        new_cols = pd.MultiIndex.from_product(
            [[i], cols], names=["seed", "Model"]
        )
        recol_invariance = invariance.copy()
        recol_invariance.columns = new_cols
        combined_invariances.append(recol_invariance)
    return pd.concat(combined_invariances, axis=1)


DATASET_KEY = "Target task"
MODEL_KEY = "Model"
PERF_KEY = "Accuracy"


def plot_transfer_barplots(
    results: list[dict[str, ExperimentResult]],
    transform: str,
) -> plt.Figure:
    serialized_res = pd.DataFrame(columns=[DATASET_KEY, MODEL_KEY, PERF_KEY])
    for transfer_res in results:
        for target_name, target_res in transfer_res[
            transform
        ].transfer_performance.items():
            for model_name, model_res in target_res.items():
                serialized_res = pd.concat(
                    [
                        serialized_res,
                        pd.DataFrame(
                            {
                                DATASET_KEY: [
                                    TRANSFER_SUBSTITUTIONS[str(target_name)]
                                ],
                                MODEL_KEY: [
                                    TRANSFER_SUBSTITUTIONS[str(model_name)]
                                ],
                                PERF_KEY: [model_res],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    perf_fig = sns.catplot(
        data=serialized_res,
        x=DATASET_KEY,
        y=PERF_KEY,
        hue=MODEL_KEY,
        kind="bar",
        legend_out=False,
        height=3,
        aspect=4 / 3,
    )
    perf_fig.tight_layout()
    return perf_fig


def plot_transfer_heatmap(
    result: ExperimentResult,
    with_title: bool = False,
    full_results: bool = False,
) -> plt.Figure:
    transfer_res = result.transfer_performance
    if not full_results:
        filtered_rows = list(TRANSFER_SUBSTITUTIONS.keys())
        filtered_cols = list(TRANSFER_SUBSTITUTIONS.keys())
        transfer_res = transfer_res.loc[filtered_rows, filtered_cols]
        ood_fig, ood_axes = plt.subplots(1, figsize=(4, 3), squeeze=False)
    else:
        ood_fig, ood_axes = plt.subplots(1, figsize=(5, 7), squeeze=False)
    ood_plot = ood_axes[0][0]
    if with_title:
        ood_plot.set_title("Transfer Accuracy")
    if full_results:
        tick_labels = result.transfer_performance.columns, transfer_res.index
    else:
        tick_labels = _get_transfer_axis_labels(transfer_res)
    sns.heatmap(
        transfer_res,
        annot=True,
        ax=ood_plot,
        xticklabels=tick_labels[0],
        yticklabels=tick_labels[1],
        cmap="magma",
    )
    ood_plot.set_xlabel("Target Dataset")
    ood_plot.set_ylabel("Pre-training Dataset")
    ood_fig.tight_layout()
    return ood_fig


def _get_transfer_axis_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    col_substitutions = [
        TRANSFER_SUBSTITUTIONS[data_name] for data_name in df.columns
    ]
    row_substitutions = [
        TRANSFER_SUBSTITUTIONS[model_name] for model_name in df.index
    ]
    return col_substitutions, row_substitutions


TRANSFER_SUBSTITUTIONS = {
    CIFAR_ONLY_KEY: "X = C\nY = C",
    f"{MIXED_CIFAR_COR_KEY}0": "X = C + O\nY = C",
    MIXED_OBJECTS_KEY: "X = C + O\nY = O",
    OBJECTS_ONLY_KEY: "X = O\nY = O",
}


@dataclass
class CorrelationPerformancePlots:
    cifar_target_perf: plt.Figure
    objects_target_perf: plt.Figure
    cifar_rep_dists: plt.Figure
    objects_rep_dists: plt.Figure


CORRELATION_VALUES = [
    0,
    0.2,
    0.4,
    0.6,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]
CORRELATION_ROWS = [
    f"{MIXED_CIFAR_COR_KEY}{cor_value}" for cor_value in CORRELATION_VALUES
]
CIFAR_PERF_COL = f"{MIXED_CIFAR_COR_KEY}0"
OBJECTS_PERF_COL = MIXED_OBJECTS_KEY
CORRELATION_COL = "CIFAR-Object Label Correlation $\\alpha$"


def plot_correlation_performance(
    results: list[ExperimentResult],
) -> CorrelationPerformancePlots:
    full_results = [res.transfer_performance for res in results]
    transfer_per_results = [
        res.transfer_performance.loc[CORRELATION_ROWS] for res in results
    ]
    cifar_target_fig = _plot_label_type_performance(
        transfer_per_results,
        full_results,
        CORRELATION_COL,
        CORRELATION_VALUES,
        CIFAR_PERF_COL,
        "CIFAR Accuracy",
    )
    objects_target_fig = _plot_label_type_performance(
        transfer_per_results,
        full_results,
        CORRELATION_COL,
        CORRELATION_VALUES,
        OBJECTS_PERF_COL,
        "Objects Accuracy",
    )

    return CorrelationPerformancePlots(
        cifar_target_perf=cifar_target_fig,
        objects_target_perf=objects_target_fig,
        cifar_rep_dists=None,
        objects_rep_dists=None,
    )


AVAILABILITY_ROWS = (
    [f"{CIFAR_ONLY_KEY}"]
    + [
        f"{MIXED_CIFAR_AVAILABILITY_KEY}{av_value}"
        for av_value in AVAILABILITY_VALUES
    ]
    + [f"{CIFAR_PERF_COL}"]
)
FULL_AVAILABILITY_VALUES = [0.0] + AVAILABILITY_VALUES + [1.0]
AVAILABILITY_COL = "Object Availability $\\beta$"


def plot_availability_performance(
    results: list[ExperimentResult],
) -> tuple[plt.Figure, plt.Figure]:
    full_results = [res.transfer_performance for res in results]
    transfer_per_results = [
        res.transfer_performance.loc[AVAILABILITY_ROWS] for res in results
    ]
    cifar_target_fig = _plot_label_type_performance(
        transfer_per_results,
        full_results,
        AVAILABILITY_COL,
        FULL_AVAILABILITY_VALUES,
        CIFAR_PERF_COL,
        "CIFAR Accuracy",
        with_legend=True,
    )
    objects_target_fig = _plot_label_type_performance(
        transfer_per_results,
        full_results,
        AVAILABILITY_COL,
        FULL_AVAILABILITY_VALUES,
        OBJECTS_PERF_COL,
        "Objects Accuracy",
    )

    return cifar_target_fig, objects_target_fig


def _plot_label_type_performance(
    results: list[pd.DataFrame],
    full_results: list[pd.DataFrame],
    target_col: str,
    target_values: list[float],
    perf_col: str,
    y_label: str,
    with_legend: bool = False,
) -> plt.Figure:
    correlation_cifar_perf = pd.concat(
        [
            # result[[perf_col]].assign(**{CORRELATION_COL: CORRELATION_VALUES})
            result[[perf_col]].assign(**{target_col: target_values})
            for result in results
        ],
        ignore_index=True,
    )

    perf_fig, perf_axes = plt.subplots(1, figsize=(3, 2.5), squeeze=False)
    perf_plot = perf_axes[0][0]
    sns.lineplot(
        data=correlation_cifar_perf,
        x=target_col,
        y=perf_col,
        label="mixed CIFAR labels",
        # label="X = C + O, Y = C",
        ax=perf_plot,
    )
    const_results = {
        "CIFAR-only": _get_const_res(
            # "X = C, Y = C": _get_const_res(
            full_results,
            target_col,
            target_values,
            f"m_{CIFAR_ONLY_KEY}",
            perf_col,
        ),
        "mixed object labels": _get_const_res(
            # "X = C + O, Y = O": _get_const_res(
            full_results,
            target_col,
            target_values,
            f"m_{MIXED_OBJECTS_KEY}",
            perf_col,
        ),
        "objecs-only": _get_const_res(
            # "X = O, Y = O": _get_const_res(
            full_results,
            target_col,
            target_values,
            f"m_{OBJECTS_ONLY_KEY}",
            perf_col,
        ),
    }
    for model_name, model_res in const_results.items():
        sns.lineplot(
            data=model_res,
            x=target_col,
            y=perf_col,
            label=model_name,
            ax=perf_plot,
            linestyle="dashed",
        )
    # perf_plot.set_xlabel("CIFAR-Object correlation in Training Data")
    perf_plot.set_ylabel(y_label)
    if not with_legend:
        perf_plot.legend().remove()

    perf_fig.tight_layout()
    return perf_fig


def _get_const_res(
    results: list[pd.DataFrame],
    target_col: str,
    target_values: list[float],
    model_name: str,
    col_name: str,
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {col_name: [res.loc[model_name, col_name]] * len(target_values)}
            ).assign(**{target_col: target_values})
            for res in results
        ],
        ignore_index=True,
    )
