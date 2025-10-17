import numpy as np
import pandas as pd
import pytest

from rhythmic_segments.helpers import split_into_blocks
from rhythmic_segments.meta import (
    aggregate_meta,
    coerce_meta_frame,
    agg_copy,
    agg_first,
    agg_index,
    agg_join,
    agg_last,
    agg_list,
)


def test_coerce_meta_frame_defaults_and_expected_rows():
    empty_df = coerce_meta_frame(None)
    assert empty_df.empty

    sized_df = coerce_meta_frame(None, expected_rows=2)
    assert len(sized_df) == 2

    data_df = coerce_meta_frame({"a": [1, 2]}, expected_rows=2)
    assert list(data_df["a"]) == [1, 2]

    with pytest.raises(ValueError):
        coerce_meta_frame({"a": [1]}, expected_rows=2)


def test_split_into_blocks_with_boundaries():
    boundaries = [False, False, True, False]
    meta = pd.DataFrame({"label": ["a", "b", "nan", "c"]})
    blocks = split_into_blocks(meta, boundaries=boundaries)

    assert len(blocks) == 2
    assert blocks[0]["label"].tolist() == ["a", "b"]
    assert blocks[1]["label"].tolist() == ["c"]


def test_split_into_blocks_with_boundaries_numeric():
    data = [1.0, 2.0, np.nan, 3.0]
    boundaries = np.isnan(np.asarray(data, dtype=float))
    blocks = split_into_blocks(data, boundaries=boundaries)
    assert len(blocks) == 2
    np.testing.assert_array_equal(blocks[0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(blocks[1], np.array([3.0]))


def test_aggregate_meta_basic_sliding():
    meta_blocks = [pd.DataFrame({"value": [1, 2, 3]})]
    value_blocks = [np.array([0.1, 0.2, 0.3])]

    result = aggregate_meta(
        meta_blocks,
        value_blocks,
        window_len=2,
        meta_agg=lambda df: {"sum": df["value"].sum()},
        expected_records=2,
    )

    assert list(result["sum"]) == [3, 5]


def test_aggregate_meta_validates_lengths():
    meta_blocks = [pd.DataFrame({"value": [1, 2]})]
    value_blocks = [np.array([0.1])]

    with pytest.raises(ValueError):
        aggregate_meta(
            meta_blocks,
            value_blocks,
            window_len=1,
            meta_agg=lambda df: {"value": df.iloc[0]["value"]},
            expected_records=1,
        )


def test_agg_copy():
    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    result = agg_copy(df, columns=["label"])
    assert result == {"label_1": "a", "label_2": "b"}

def test_agg_index():
    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    assert agg_index(df, 0, columns=["label"]) == {"label": "a"}
    assert agg_index(df, -1, columns=["section"]) == {"section": "y"}

def test_agg_first_last():
    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    assert agg_first(df, columns=["label"]) == {"label": "a"}
    assert agg_last(df, columns=["section"]) == {"section": "y"}

def test_agg_join():
    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    result = agg_join(df, separator="|")
    assert result == {"label": "a|b", "section": "x|y"}


def test_agg_list():
    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    result = agg_list(df, columns=["label"])
    assert result == {"label": ["a", "b"]}


def test_get_aggregator():
    from rhythmic_segments.meta import get_aggregator

    df = pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]})
    agg = get_aggregator("first", columns=["label"])
    assert agg(df) == {"label": "a"}

    join_agg = get_aggregator("join", columns=["label"], separator="-")
    assert join_agg(df) == {"label": "a-b"}

    copy_agg = get_aggregator("copy", columns=["label"])
    assert copy_agg(df) == {"label_1": "a", "label_2": "b"}

    list_agg = get_aggregator("list", columns=["label"])
    assert list_agg(df) == {"label": ["a", "b"]}

    with pytest.raises(ValueError):
        get_aggregator("unknown")
