from typing import Literal, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from plotly import graph_objects as go


AllLevels = Literal["all"]


# @overload
# def aggregate_mean_std_dev(
#     values: list[pd.DataFrame],
#     levels_to_preserve: list[str] | AllLevels = []
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     ...


# @overload
# def aggregate_mean_std_dev(
#     values: list[pd.Series],
#     levels_to_preserve: list[str] | AllLevels = [],
# ) -> tuple[pd.Series, pd.Series]:
#     ...


def aggregate_mean_std_dev(
    values: Sequence[pd.DataFrame | pd.Series],
    levels_to_preserve: Sequence[str] | AllLevels = [],
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """Compute the mean and standard deviation of a list of dataframes."""
    index_names = values[0].index.names
    if levels_to_preserve == "all":
        levels_to_preserve = index_names
    levels_to_preserve = cast(list[str], levels_to_preserve)

    # Pre-average the values to compute the std-dev only across std_dev_levels
    pre_averaged_values = [
        cast(pd.DataFrame, val.groupby(level=levels_to_preserve).mean())
        for val in values
    ]

    combined = pd.concat(
        pre_averaged_values,
        axis=0,
        keys=range(len(values)),
        names=["seed_id"] + levels_to_preserve,
    )
    groups = combined.groupby(level=levels_to_preserve)
    mean = groups.mean()
    std_dev = groups.std()
    return cast(tuple[pd.DataFrame, pd.DataFrame], (mean, std_dev))


HEX_COLOR_SEQUENCE = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#e377c2",  # raspberry yogurt pink
    "#8c564b",  # chestnut brown
    "#bcbd22",  # curry yellow-green
    "#7f7f7f",  # middle gray
    "#17becf",  # blue-teal
]
COLOR_SEQUENCE = [
    (int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16))
    for col in HEX_COLOR_SEQUENCE
]


def add_mean_std_dev_trace(
    fig: go.Figure,
    trace_idx: int,
    mean_values: np.ndarray | pd.Series | pd.DataFrame,
    std_dev_values: np.ndarray | pd.Series | pd.DataFrame | None,
    name: str,
    x_values: np.ndarray | pd.Series | pd.DataFrame | list | None = None,
    dashed: bool = False,
) -> go.Figure:
    if x_values is None:
        assert isinstance(mean_values, pd.Series) or isinstance(
            mean_values, pd.DataFrame
        )
        x_vals = mean_values.index
    else:
        x_vals = x_values

    color = get_color(trace_idx)
    fillcolor = get_color(trace_idx, transparent=True)

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=mean_values,
            name=name,
            line=dict(
                color=color,
                width=3,
                dash="dash" if dashed else "solid",
            ),
            showlegend=name != "",
            legendgroup=trace_idx,
        )
    )
    if std_dev_values is None:
        return fig

    upper_bound = mean_values + std_dev_values
    lower_bound = mean_values - std_dev_values
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=upper_bound,
            mode="lines",
            fillcolor=fillcolor,
            fill="tonexty",
            line=dict(width=0),
            showlegend=False,
            legendgroup=trace_idx,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=lower_bound,
            mode="lines",
            fillcolor=fillcolor,
            fill="tonexty",
            line=dict(width=0),
            showlegend=False,
            legendgroup=trace_idx,
        )
    )
    return fig


def plot_mean_std_dev(
    result_means: dict[str, pd.DataFrame],
    result_std_devs: dict[str, pd.DataFrame] | None,
    figure_offset: tuple[go.Figure, int] | None = None,
) -> go.Figure:
    fig, idx_offset = (
        (go.Figure(), 0) if figure_offset is None else figure_offset
    )
    for i, (res_name, res_mean) in enumerate(result_means.items()):
        if result_std_devs is None:
            res_std = None
        else:
            res_std = result_std_devs[res_name]
        add_mean_std_dev_trace(
            fig,
            i + idx_offset,
            res_mean.to_numpy(),
            res_std,
            name=res_name,
            x_values=res_mean.index.to_numpy(),
        )
    return fig


def get_color(
    index: int,
    transparent: bool = False,
) -> str:
    color_vals = COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]
    if transparent:
        return f"rgba({color_vals[0]}, {color_vals[1]}, {color_vals[2]}, 0.2)"
    else:
        return f"rgb({color_vals[0]}, {color_vals[1]}, {color_vals[2]})"
