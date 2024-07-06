import os
import sys
from pathlib import Path

import plotly
import plotly.io as pio


# from .visualize.paper_style import set_paper_style


def setup_notebook(
    path_to_src: str,
) -> None:
    """Common setup utils for Jupyter notebooks like changing the working
    directory and setting environment variables."""
    os.chdir(path_to_src)

    os.environ["ARTIFACTS_DIR"] = "../artifacts"
    os.environ["DATA_DIR"] = "../data"

    # pio.renderers.default = "plotly_mimetype+notebook_connected+pdf"
    pio.renderers.default = "plotly_mimetype+notebook_connected"
    plotly.offline.init_notebook_mode(connected=True)
    # NOTE: Switching to Plotly, so this is not needed anymore for now
    # set_paper_style()
