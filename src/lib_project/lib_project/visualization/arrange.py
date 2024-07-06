import math
from collections import defaultdict
from typing import Any, cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def arrange_figures_in_grid(
    figures: dict[str, go.Figure],
    n_cols: int,
    size: tuple[int, int] | None = None,
) -> go.Figure:
    n_rows = math.ceil(len(figures) / n_cols)
    combined_fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(figures.keys()),
    )

    # Store trace names and their corresponding colors to check
    # for inconsistencies
    trace_color_map = defaultdict(set)
    for i, fig in enumerate(figures.values()):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        for trace in fig.data:
            trace = cast(Any, trace)
            combined_fig.add_trace(
                trace,
                row=row,
                col=col,
            )
            if trace.name is not None:
                trace_color_map[trace.name].add(trace.line.color)

        for shape in fig.layout.shapes:
            combined_fig.add_shape(shape, row=row, col=col)
        for annotation in fig.layout.annotations:
            combined_fig.add_annotation(annotation, row=row, col=col)

        # Set axis labels based on the original figure
        combined_fig.update_xaxes(
            title_text=fig.layout.xaxis.title.text, row=row, col=col
        )
        combined_fig.update_yaxes(
            title_text=fig.layout.yaxis.title.text, row=row, col=col
        )

    # Check for traces with the same name but different colors
    for trace_name, colors in trace_color_map.items():
        if len(colors) > 1:
            raise ValueError(
                f"Trace '{trace_name}' has inconsistent colors across subplots."
            )
    # Show only one legend item per trace
    seen = set()
    combined_fig.for_each_trace(
        lambda trace: trace.update(
            showlegend=(
                trace.name is not None
                and trace.name not in seen
                and (seen.add(trace.name) or True)
            )
        )
    )

    if size is None:
        width = next(iter(figures.values())).layout.width * n_cols
        height = next(iter(figures.values())).layout.height * n_rows
    else:
        width, height = size
    combined_fig.update_layout(
        width=width,
        height=height,
        legend=dict(orientation="h", y=-0.05),  # Place legend at the bottom
        legend_traceorder="normal",
    )
    return combined_fig
