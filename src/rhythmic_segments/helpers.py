"""Interval utilities shared across rhythmic segments components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def split_into_blocks(
    data: Union[Sequence[float], np.ndarray, pd.DataFrame],
    boundaries: Iterable[bool],
    *,
    drop_empty: bool = True,
    copy: bool = True,
) -> List[Union[np.ndarray, pd.DataFrame]]:
    """Split *data* into contiguous blocks using ``boundaries``.

    Parameters
    ----------
    data : Sequence, np.ndarray, or pandas.DataFrame
        Interval or metadata values to split.
    boundaries : Iterable[bool]
        Boolean mask marking split points.
    drop_empty : bool, optional
        Drop zero-length blocks when ``True`` (default).
    copy : bool, optional
        Return copies of the underlying data when ``True`` (default).
    """

    boundary_mask = np.asarray(list(boundaries), dtype=bool)

    data_df: Optional[pd.DataFrame]
    arr: Optional[np.ndarray]

    if isinstance(data, pd.DataFrame):
        data_df = data
        arr = None
        length = len(data_df)
    else:
        data_df = None
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape(0)
        length = arr.shape[0]

    if boundary_mask.shape != (length,):
        raise ValueError("boundaries mask must match the length of the input data")

    def make_block(start: int, end: int) -> Union[np.ndarray, pd.DataFrame]:
        if data_df is not None:
            block_df = data_df.iloc[start:end]
            block_df = block_df.reset_index(drop=True)
            return block_df.copy() if copy else block_df
        assert arr is not None
        block_arr = arr[start:end]
        return block_arr.copy() if copy else block_arr

    blocks: List[Union[np.ndarray, pd.DataFrame]] = []
    start = 0
    for idx, is_boundary in enumerate(boundary_mask):
        if is_boundary:
            if idx > start or not drop_empty:
                blocks.append(make_block(start, idx))
            start = idx + 1

    if length > start or not drop_empty:
        blocks.append(make_block(start, length))

    return blocks
