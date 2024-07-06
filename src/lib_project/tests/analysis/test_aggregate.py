import pandas as pd

from lib_project.analysis.aggregate import summarize_df


def test_aggregate_no_idx():
    df_1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, dtype="float")
    df_2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}, dtype="float")
    agg = summarize_df([df_1, df_2])
    expected = pd.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]}, dtype="float")
    pd.testing.assert_frame_equal(agg, expected)


def test_aggregate_idx():
    idx = pd.Index([1, 2, 3])
    df_1 = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=idx, dtype="float"
    )
    df_2 = pd.DataFrame(
        {"a": [7, 8, 9], "b": [10, 11, 12]}, index=idx, dtype="float"
    )
    agg = summarize_df([df_1, df_2])
    expected = pd.DataFrame(
        {"a": [4, 5, 6], "b": [7, 8, 9]}, index=idx, dtype="float"
    )
    pd.testing.assert_frame_equal(agg, expected)


def test_aggregate_multi_index():
    idx = pd.MultiIndex.from_tuples(
        [(1, 2), (2, 3), (3, 4)], names=["i1", "i2"]
    )
    df_1 = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=idx, dtype="float"
    )
    df_2 = pd.DataFrame(
        {"a": [7, 8, 9], "b": [10, 11, 12]}, index=idx, dtype="float"
    )
    agg = summarize_df([df_1, df_2])
    expected = pd.DataFrame(
        {"a": [4, 5, 6], "b": [7, 8, 9]}, index=idx, dtype="float"
    )
    pd.testing.assert_frame_equal(agg, expected)
