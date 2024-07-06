from dataclasses import dataclass
from typing import Optional

from plotly import graph_objects as go


@dataclass
class PaperStyle:
    font_family: str = "Open Sans"
    font_size: int = 32
    legend_font_family: str = "Open Sans"
    legend_font_size: int = 28
    plot_bgcolor = "rgba(0,0,0,0)"
    gridwidth: int = 1
    gridcolor: str = "rgba(0,0,0,0.1)"
    legend_bgcolor: str = "rgba(1.0,1.0,1.0,0.8)"


def with_paper_style(
    fig: go.Figure,
    config: PaperStyle = PaperStyle(),
    # Top left corner of the plot
    legend_pos: Optional[tuple[float, float]] = (0.98, 0.98),
    legend_xanchor: str = "right",
    legend_yanchor: str = "top",
    legend_orientation: str = "v",
    scale: float = 1.0,
) -> go.Figure:
    if legend_pos is None:
        show_legend = False
        legend = None
    else:
        show_legend = True
        legend = dict(
            x=legend_pos[0],
            y=legend_pos[1],
            xanchor=legend_xanchor,
            yanchor=legend_yanchor,
            traceorder="normal",
            font=dict(
                family=config.legend_font_family,
                size=config.legend_font_size * scale,
                # color="black"
            ),
            bordercolor=config.gridcolor,
            borderwidth=config.gridwidth,
            bgcolor=config.legend_bgcolor,
            orientation=legend_orientation,
        )

    axis_config = dict(
        showgrid=True,
        gridwidth=config.gridwidth,
        gridcolor=config.gridcolor,
        ticks="outside",
        tickwidth=config.gridwidth,
        tickcolor=config.gridcolor,
        zeroline=False,
        showline=False,
    )
    fig.update_layout(
        font_family=config.font_family,
        font_size=config.font_size * scale,
        plot_bgcolor=config.plot_bgcolor,
        title_text="",
        legend_title="",
        showlegend=show_legend,
        legend=legend,
        xaxis=axis_config,
        yaxis=axis_config,
    )
    return fig


# Old Matplotlib style
# import matplotlib.pyplot as plt


# def set_paper_style(scale: float = 1.0):
#     # Use before creating the figure
#     plt.rcParams["font.size"] = int(scale * 13)
#     plt.rcParams["pdf.fonttype"] = 42
#     plt.rcParams["ps.fonttype"] = 42
#     plt.rcParams["axes.labelsize"] = int(scale * 14)
#     plt.rcParams["axes.labelweight"] = "bold"
#     plt.rcParams["axes.titlesize"] = int(scale * 8)
#     plt.rcParams["axes.linewidth"] = int(scale * 2)
#     plt.rcParams["xtick.labelsize"] = int(scale * 12)
#     plt.rcParams["ytick.labelsize"] = int(scale * 12)
#     plt.rcParams["legend.fontsize"] = int(scale * 13)
#     plt.rcParams["figure.titlesize"] = int(scale * 14)
#     plt.rcParams["lines.linewidth"] = int(scale * 2.5)
