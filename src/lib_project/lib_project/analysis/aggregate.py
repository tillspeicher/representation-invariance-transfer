import pandas as pd

from lib_dl_base.results.aggregate import (
    COLOR_SEQUENCE,
    add_mean_std_dev_trace,
    aggregate_mean_std_dev,
)


def summarize_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    combined_dfs = pd.concat(dfs)
    n_idx_cols = len(combined_dfs.index.names)
    group_by_levels = tuple(range(n_idx_cols))
    return pd.concat(dfs).groupby(level=group_by_levels).mean()


def summarize_df_dict(
    dfs: list[dict[str, pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    assert len(dfs) > 0
    res_1 = dfs[0]
    combined_dfs: dict[str, list[pd.DataFrame]] = {}
    for res in dfs:
        assert isinstance(res, dict)
        assert res.keys() == res_1.keys()
        for key, df in res.items():
            combined_dfs.setdefault(key, []).append(df)
    return {key: summarize_df(dfs) for key, dfs in combined_dfs.items()}
