from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from IPython.display import display

from lib_project.experiment import ExperimentResult, load_results
from lib_project.notebook import publish_notebook

from .experiment import EXP_NAME, ExperimentConfig
from .experiment import ExperimentResult as TMExperimentResult


TMResult = ExperimentResult[ExperimentConfig, TMExperimentResult]


def load(
    config_name: str,
    seed_ids: list[int],
) -> list[TMResult]:
    return load_results(
        EXP_NAME,
        config_name,
        seed_ids,
        ExperimentConfig,
        TMExperimentResult,
    )


def publish(
    notebook: str,
) -> None:
    # Use a random postfix to make it harder to guess the file name
    if notebook == "eval":
        output_path = f"experiments/{EXP_NAME}/eval_593dus.html"
    else:
        raise ValueError(f"Unknown notebook: {notebook}")
    notebook_path = f"./experiments/{EXP_NAME}/{notebook}.ipynb"
    publish_notebook(
        notebook_path,
        output_path,
    )


SOURCE_KEY = "# Training\nTransforms"
TRANSFER_KEY = "# Transfer\nTransforms"
PERF_MEAN_KEY = "perf"
PERF_STD_DEV_KEY = "std_dev"

TRANSFORM_COUNTS = list(range(1, 9))


def show_line_plots(
    results: list[ExperimentResult],
) -> plt.Figure:
    transfer_results = []
    for result in results:
        transfer_res = result.value.transfer_performance
        transfer_idx = transfer_res.index
        transfer_res = transfer_res.loc[
            transfer_idx.get_level_values("split") == "test"
        ].droplevel("split", axis=0)
        transfer_results.append(transfer_res)
    transfer_results = pd.concat(transfer_results, axis=0)

    performance_results = {}
    for model_key, model_res in transfer_results.iterrows():
        model_key = cast(tuple[str, str], model_key)
        num_source_transforms = int(model_key[0][2:])
        num_transfer_transforms = int(model_key[1][2:])
        performance_results.setdefault(SOURCE_KEY, []).append(
            num_source_transforms
        )
        performance_results.setdefault(TRANSFER_KEY, []).append(
            num_transfer_transforms
        )
        performance_results.setdefault(PERF_MEAN_KEY, []).append(
            model_res["accuracy"]
        )
    performance_df = pd.DataFrame(performance_results)

    perf_fig, perf_axes = plt.subplots(1, figsize=(4.5, 3.5), squeeze=False)
    perf_plot = perf_axes[0][0]
    sns.lineplot(
        data=performance_df,
        x=TRANSFER_KEY,
        y=PERF_MEAN_KEY,
        hue=SOURCE_KEY,
        # style=SOURCE_KEY,
        errorbar="ci",
        ax=perf_plot,
        palette="tab10",
    )
    perf_plot.set_xlabel("# Transfer Transforms")
    perf_plot.set_ylabel("Transfer accuracy")
    perf_plot.set_xticks(TRANSFORM_COUNTS, TRANSFORM_COUNTS)

    perf_fig.tight_layout()
    plt.show()
    return perf_fig
