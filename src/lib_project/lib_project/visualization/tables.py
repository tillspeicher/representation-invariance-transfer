import pandas as pd
from IPython.display import Markdown as md
from IPython.display import display


class TableFormatter:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy().astype(float)
        self.styler = self.df.style

    def display_with_heatmap(
        self,
        invert: bool = False,
    ) -> "TableFormatter":
        cmap = "Blues_r" if invert else "Blues"
        self.styler = self.styler.background_gradient(
            cmap=cmap,
        )
        return self

    def highlight_extreme(
        self,
        max: bool = True,
    ) -> "TableFormatter":
        params = {"color": "lightgreen", "axis": 0}
        if max:
            self.styler = self.styler.highlight_max(**params)
        else:
            self.styler = self.styler.highlight_min(**params)
        return self

    def show(self) -> None:
        display(self.styler)
