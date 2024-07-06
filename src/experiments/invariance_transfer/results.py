from typing import Literal, cast

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown as md
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lib_project.analysis.aggregate import aggregate_mean_std_dev
from lib_project.experiment import ExperimentResult, load_results
from lib_project.notebook import publish_notebook

from .experiment import EXP_NAME, ExperimentConfig
from .experiment import ExperimentResult as ITExperimentResult


ITResult = ExperimentResult[ExperimentConfig, ITExperimentResult]


def load(
    config_name: str,
    seed_ids: list[int],
) -> list[ITResult]:
    return load_results(
        EXP_NAME,
        config_name,
        seed_ids,
        ExperimentConfig,
        ITExperimentResult,
    )


def publish(
    notebook: str,
) -> None:
    # Use a random postfix to make it harder to guess the file name
    if notebook == "eval":
        output_path = f"experiments/{EXP_NAME}/results_594bmdsh.html"
        notebook_path = f"./experiments/{EXP_NAME}/eval.ipynb"
    elif notebook == "clustering":
        output_path = f"experiments/{EXP_NAME}/clustering_3jfw3u5j.html"
        notebook_path = f"./experiments/{EXP_NAME}/transforms_clustering.ipynb"
    else:
        raise ValueError(f"Unknown notebook: {notebook}")
    publish_notebook(
        notebook_path,
        output_path,
    )


def show_results(
    results: list[ITResult],
    metrics: list[str],
    show_training: bool = False,
) -> None:
    if show_training:
        print("Training performance:")
        training_res, _ = aggregate_mean_std_dev(
            [res.value.training_performance for res in results]
        )
        model_names = training_res.index.get_level_values("model")
        training_res.index = model_names
        display(training_res.T)
    print("Distance metrics:")
    for metric in metrics:
        metric_invariances = get_distances(results, metric)
        display(md(f"### Metric: {metric}"))
        for dataset, dataset_distances in metric_invariances.items():
            display(md(f"#### Dataset: {dataset}"))
            fig = plot_distances(
                dataset_distances,
                y_label="Training Transformations",
                reverse_colors=metric == "l2",
            )
            pio.show(fig)


def get_distances(
    results: list[ITResult],
    metric: str,
) -> dict[str, pd.DataFrame]:
    dataset_invariances: dict[str, list[pd.DataFrame]] = {}
    for result in results:
        for dataset, invariances in result.value.invariances.items():
            metric_invariances = invariances[metric]
            if dataset not in dataset_invariances:
                dataset_invariances[dataset] = []
            dataset_invariances[dataset].append(metric_invariances)
    return {
        dataset: cast(
            pd.DataFrame,
            aggregate_mean_std_dev(dataset_distances)[0].apply(pd.to_numeric),
        )
        for dataset, dataset_distances in dataset_invariances.items()
    }


def filter_category(
    distances: pd.DataFrame,
    category: str | None,
) -> pd.DataFrame:
    """Isolate either the objects or random-only parts of the results"""
    if category == "objects":

        def filter(name: str) -> bool:
            return not name.endswith("_rand") and name != "untrained"

    elif category == "random":

        def filter(name: str) -> bool:
            return name.endswith("_rand") and name != "untrained"

    elif category is None:

        def filter(name: str) -> bool:
            return name != "untrained"

    else:
        raise ValueError(f"Unknown category: {category}")
    columns = [c for c in distances.columns if filter(c)]
    rows = [r for r in distances.index if filter(r)]
    return distances.loc[rows + ["untrained"], columns]


DIST_PLOT_SCALE = 30


def plot_distances(
    distances: pd.DataFrame,
    y_label: str = "Models",
    reindex: bool = True,
    reverse_colors: bool = False,
) -> go.Figure:
    if reindex:
        model_candidates = list(distances.columns) + [
            "none",
            # "untrained",
        ]
        index_values = []
        for model in model_candidates:
            if model in distances.index:
                index_values.append(model)
            if f"{model}_rand" in distances.index:
                index_values.append(f"{model}_rand")
        index = pd.Index(index_values)
        distances = distances.reindex(index=index)
    else:
        index = distances.index
    y_values = (
        [" - ".join(keys) for keys in index] if index.nlevels > 1 else index
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=distances,
            texttemplate="%{z:.2f}",
            x=distances.columns,
            y=y_values,
            colorscale="magma",
            colorbar=dict(title=""),
            hoverongaps=False,
            reversescale=reverse_colors,
        )
    )
    fig.update_layout(
        width=DIST_PLOT_SCALE * (6 + len(distances.columns)),
        height=DIST_PLOT_SCALE * (6 + len(distances.index)),
        xaxis_title="Target Transformations",
        yaxis_title=y_label,
        # font=dict(size=12),
        # margin=dict(t=50),
    )
    fig.update_traces(
        showscale=True,
        hovertemplate=(
            "Training Transformation: %{y}<br>"
            "Target Transformation: %{x}<br>"
            "Invariance: %{z:.5f}<extra></extra>"
        ),
    )
    # pio.show(fig)
    return fig


def normalize_by_transformation(distances: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the representation distances for each model
    to each transformation by the mean distance of all models across that
    transformation.

    We do this to see how much more or less invariant the model is to this
    transformation compared to others.
    """
    # mean distance of all models across each transformation
    mean_col_dist = distances.mean(axis=0)
    return distances.divide(mean_col_dist, axis=1)


def normalize_by_rank(
    distances: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Normalize the representation distances for each model
    to each transformation by computing the rank of each model's distance
    across the traget transformation.

    We do this to see the how many-th most invariant the model is to this
    transformation compared to others.
    """
    return {
        dataset: dataset_distances.rank(axis=0, ascending=True).astype(int)
        for dataset, dataset_distances in distances.items()
    }


def transformation_difficulties(
    distances: pd.DataFrame,
    print_values: bool = False,
) -> pd.DataFrame:
    """
    Compute the mean distance across all models for each transformation.
    Sort the distances in ascending order.
    """
    transformation_means = cast(pd.Series, distances.mean(axis=0)).to_frame()
    transformation_means.columns = ["mean"]
    transformation_means.sort_values(by="mean", axis=0, inplace=True)
    if print_values:
        for transformation, value in transformation_means.iterrows():
            print(f"\"{transformation}\": {value['mean']},")
    return transformation_means.T


def get_full_distances(
    results: list[ITResult],
    metric: str,
) -> dict[str, pd.DataFrame]:
    collected_invariances: dict[str, list[pd.DataFrame]] = {}
    for result in results:
        for dataset, invariances in result.value.invariances.items():
            metric_invariances = invariances[metric]
            if dataset not in collected_invariances:
                collected_invariances[dataset] = []
            collected_invariances[dataset].append(metric_invariances)

    combined_invariances = {}
    for dataset, dataset_results in collected_invariances.items():
        dataset_invariances = []
        for i, result in enumerate(dataset_results):
            dataset_cols = pd.MultiIndex.from_product(
                [[i], result.columns],
                names=["seed_id", "transform"],
            )
            recol_result = result.copy()
            recol_result.columns = dataset_cols
            dataset_invariances.append(recol_result)
        combined_invariances[dataset] = pd.concat(dataset_invariances, axis=1)
    return combined_invariances


def target_vs_non_target_comparison(
    distances: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    For each transformation, show the invariance of the two models trained
    to be invariant to it (on objects and random data) and the average
    invariance of the models not trained to be invariant to it as reference.
    """
    rows = [
        # "untrained",
        "img_target",
        "img_other",
        "rand_target",
        "rand_other",
        "img_base",  # None models without transformation
        "rand_base",  # None models without transformation
    ]
    aggregated_distances = {}
    for dataset, dataset_distances in distances.items():
        seed_ids = dataset_distances.columns.get_level_values(
            "seed_id"
        ).unique()
        for seed_id in seed_ids:
            seed_distances = dataset_distances.loc[slice(None), seed_id]
            transforms = seed_distances.columns
            models_object = set(transforms)
            models_rand = set(f"{t}_rand" for t in transforms)
            dataset_aggregated_distances = {
                transform: [
                    # seed_distances.loc["untrained", transform],
                    seed_distances.loc[transform, transform],
                    seed_distances.loc[
                        list(models_object - {transform}), transform
                    ].mean(),
                    seed_distances.loc[f"{transform}_rand", transform],
                    seed_distances.loc[
                        list(models_rand - {f"{transform}_rand"}), transform
                    ].mean(),
                    seed_distances.loc["none", transform],
                    seed_distances.loc["none_rand", transform],
                ]
                for transform in transforms
            }
            if dataset not in aggregated_distances:
                aggregated_distances[dataset] = []
            columns_idx = pd.MultiIndex.from_product(
                [[seed_id], transforms],
                names=["seed_id", "transform"],
            )
            dataset_seed_comparison = pd.DataFrame(
                dataset_aggregated_distances,
                index=rows,
                dtype=float,
            )
            dataset_seed_comparison.columns = columns_idx
            aggregated_distances[dataset].append(dataset_seed_comparison)
    return {
        dataset: pd.concat(dataset_distances, axis=1)
        for dataset, dataset_distances in aggregated_distances.items()
    }


DATASET_ORDER = [
    "op-img",
    "op-rand",
    "os-img",
    "os-rand",
    "rp-img",
    "rp-rand",
    "rs-img",
    "rs-rand",
]


def compute_target_vs_non_target_means(
    comparisons: dict[str, pd.DataFrame],
    reduction_func: str,
) -> pd.DataFrame:
    assert reduction_func in ["mean", "std"]
    dataset_means = {
        dataset: (
            dataset_comparison.mean(axis=1)
            if reduction_func == "mean"
            else dataset_comparison.std(axis=1)
        )
        for dataset, dataset_comparison in comparisons.items()
    }
    return pd.DataFrame(dataset_means)


MODEL_TYPE_REMAPPING = {
    "img_target": "Same",
    "img_other": "Other",
    "img_base": "None",
    "rand_target": "Same",
    "rand_other": "Other",
    "rand_base": "None",
    "untrained": "Untrained",
}
IMG_MODELS = ["img_target", "img_other", "img_base"]
RAND_MODELS = ["rand_target", "rand_other", "rand_base"]
COL_ORDER = {
    "img": [
        "op-img",
        "os-img",
        "rs-rand",
        "cifar10",
        "cifar100",
    ],
    "rand": [
        "rp-rand",
        "rs-rand",
        "os-img",
        "cifar10",
        "cifar100",
    ],
}


def print_comp_table(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    mean_ranks: pd.DataFrame,
) -> None:
    mean_mins = {
        "img": means.loc[IMG_MODELS].min(axis=0),
        "rand": means.loc[RAND_MODELS].min(axis=0),
    }
    for model_name, name_remapping in MODEL_TYPE_REMAPPING.items():
        if model_name == "untrained":
            continue
        model_means = means.loc[model_name]
        model_stds = stds.loc[model_name]
        values = []
        if model_name in IMG_MODELS:
            category_mins = mean_mins["img"]
        else:
            category_mins = mean_mins["rand"]

        if model_name.startswith("img"):
            datasets = COL_ORDER["img"]
        elif model_name.startswith("rand"):
            datasets = COL_ORDER["rand"]
        else:
            raise ValueError()
        for dataset in datasets:
            mean = model_means[dataset]
            std = model_stds[dataset]

            val = f"${mean:.2f}_{{\\pm {std:.2f}}}$"
            if mean == category_mins[dataset]:
                val = "$\\mathbf{" + val[1:-1] + "}$"
            values.append(val)
        print(f"& {name_remapping} & {' & '.join(values)} \\\\")


def plot_comparison(comparisons: pd.DataFrame) -> go.Figure:
    """
    Expects the output of `target_vs_non_target_comparison`.
    Plots it in a ranked way.
    """
    ranked_comparison = comparisons.sort_values(by="untrained", axis=1)
    fig = go.Figure()
    for model, model_invariance in ranked_comparison.iterrows():
        fig.add_trace(
            go.Bar(
                x=ranked_comparison.columns,
                y=model_invariance,
                name=model,
                texttemplate="%{z:.2f}",
            )
        )
    fig.update_layout(
        xaxis_title="Target Transformations",
        yaxis_title="Invariance",
    )
    pio.show(fig)
    return fig


def plot_paper_comparisons(
    invariance_means: pd.DataFrame,
    invariance_stds: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    for model_name, name_remapping in MODEL_TYPE_REMAPPING.items():
        model_means = invariance_means.loc[model_name]
        model_stds = invariance_stds.loc[model_name]
        fig.add_trace(
            go.Scatter(
                x=model_means.index,
                y=model_means,
                name=name_remapping,
                error_y=dict(
                    type="data",
                    array=model_stds,
                    visible=True,
                ),
                mode="lines+markers",
                texttemplate="%{y:.2f}",
            )
        )
    fig.update_layout(
        xaxis_title="Target Transformations",
        yaxis_title="l2-Distance Ratio",
    )
    return fig


def plot_paper_distances(
    results: list[ITResult],
    category: str,
) -> go.Figure:
    if category == "img":
        dataset = "op-img"
    elif category == "rand":
        dataset = "rp-rand"
    else:
        raise ValueError()
    dataset_distances = get_distances(results, "l2")[dataset]
    if category == "img":
        model_names = [
            m for m in dataset_distances.index if not m.endswith("_rand")
        ]
        model_distances = dataset_distances.loc[model_names]
    elif category == "rand":
        rand_model_names = [
            m for m in dataset_distances.index if m.endswith("_rand")
        ]
        model_distances = dataset_distances.loc[rand_model_names]
        model_name_remapping = {
            model_name: model_name[:-5] for model_name in rand_model_names
        }
        model_distances.rename(index=model_name_remapping, inplace=True)
    else:
        raise ValueError()
    fig = plot_distances(
        model_distances,
        y_label="Training Transformations",
        reverse_colors=True,
        # reindex=False,
    )
    fig.update_layout(
        width=1100,
        height=1000,
        font={
            "size": 20,
        },
        # xaxis_title="Models 1",
        # yaxis_title="Models 2",
    )
    pio.show(fig)
    return fig


def get_training_time(results: list[ITResult]) -> pd.DataFrame:
    training_time = {
        "model": [],
        "time": [],
    }
    for result in results:
        training_time["model"].append(result.model_name)
        training_time["time"].append(result.training_time)
    return pd.DataFrame(training_time).set_index("model")


def compare_rankings(
    comparisons: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Expects the output of `target_vs_non_target_comparison`.
    Computes the correlations between different model invariances.
    """
    correlations = {
        "pearson": pd.DataFrame(),
        "spearman": pd.DataFrame(),
        "kendall": pd.DataFrame(),
    }

    combined_comparisons = comparisons
    target_order = list(comparisons.columns) + ["untrained"]

    for row_1, row_distances_1 in combined_comparisons.iterrows():
        for row_2, row_distances_2 in combined_comparisons.iterrows():
            for correlation_type, correlation_values in correlations.items():
                correlation_values.loc[row_1, row_2] = row_distances_1.corr(
                    row_distances_2, method=correlation_type
                )
    correlations = {
        metric: metric_correlations.reindex(
            index=target_order,
            columns=target_order,
        )
        for metric, metric_correlations in correlations.items()
    }
    return correlations


def plot_correlations(
    correlations: dict[str, pd.DataFrame],
) -> dict[str, go.Figure]:
    """
    Expects the output of `compare_rankings`.
    Plots a heatmap of the correlations.
    """
    figures = {}
    for correlation_type, correlation_values in correlations.items():
        fig = plot_distances(correlation_values, reindex=False)
        fig.update_layout(
            # width=600,
            # height=600,
            xaxis_title="Models 1",
            yaxis_title="Models 2",
        )
        fig.update_traces(
            showscale=True,
            # hovertemplate=(
            #     "Training Transformation: %{y}<br>"
            #     "Target Transformation: %{x}<br>"
            #     "Invariance: %{z:.5f}<extra></extra>"
            # )
        )
        display(md(f"#### {correlation_type} correlation:"))
        pio.show(fig)
        figures[correlation_type] = fig
    return figures


ClusterAxis = Literal["training_transforms", "target_transforms"]


def visualize_invariance_2d(
    distances: pd.DataFrame,
    axis: ClusterAxis,
    clusters: pd.DataFrame,
) -> go.Figure:
    """
    Visualize the invariance/distance vectors as a 2D projection.
    """
    axis_invariances = _get_axis_invariance(distances, axis)
    pca = PCA(n_components=2)
    low_dim_embedding = pca.fit_transform(axis_invariances)
    fig = go.Figure(
        data=go.Scatter(
            x=low_dim_embedding[:, 0],
            y=low_dim_embedding[:, 1],
            mode="markers+text",
            text=axis_invariances.index,
            textposition="top center",
            marker=dict(
                size=10,
                color=clusters["cluster"],
                colorscale="Viridis",
            ),
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        #     font=dict(size=12),
        #     margin=dict(t=50),
    )
    pio.show(fig)
    print("Explained variance (components):", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())
    return fig


def cluster_transformations(
    distances: pd.DataFrame,
    axis: ClusterAxis,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """
    Cluster the transformations based on their invariance to each other.
    """
    axis_invariances = _get_axis_invariance(distances, axis)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto",
    ).fit(axis_invariances)
    return pd.DataFrame(
        kmeans.labels_,
        index=axis_invariances.index,
        columns=["cluster"],
    )


def _get_axis_invariance(
    distances: pd.DataFrame,
    axis: ClusterAxis,
) -> pd.DataFrame:
    """
    Compute the invariance vectors for each transformation.
    """
    if axis == "training_transforms":
        return distances
    elif axis == "target_transforms":
        return distances.T
    else:
        raise ValueError(f"Unknown axis: {axis}")
