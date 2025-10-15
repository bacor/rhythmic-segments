"""Utility functions for working with rhythmic segments.

This module provides helpers for constructing and manipulating n-gram segments
that represent consecutive rhythmic events.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any, Callable, Iterable, List, Optional, Union, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from .utils import is_nan


def split_blocks(
    intervals: Iterable[float],
    *,
    separator: Any = np.nan,
    drop_empty: bool = True,
    copy: bool = True,
) -> List[np.ndarray]:
    """Split *intervals* into contiguous blocks using *separator* as delimiter.

    >>> import numpy as np
    >>> split_blocks([1, 2, np.nan, 3])
    [array([1., 2.]), array([3.])]
    >>> split_blocks([1, 2, 3], separator=None)
    [array([1., 2., 3.])]
    """

    arr = np.asarray(intervals, dtype=float)
    if arr.ndim != 1:
        raise ValueError("intervals must be one-dimensional")

    boundary_mask = (
        np.array([is_nan(value) for value in arr], dtype=bool)
        if is_nan(separator)
        else arr == separator
    )

    blocks: List[np.ndarray] = []
    start = 0
    for idx in np.flatnonzero(boundary_mask):
        section = arr[start:idx]
        if section.size > 0 or not drop_empty:
            blocks.append(section.copy() if copy else section)
        start = idx + 1

    tail = arr[start:]
    if tail.size > 0 or not drop_empty:
        blocks.append(tail.copy() if copy else tail)
    return blocks


def _split_meta_blocks(meta: pd.DataFrame, intervals: np.ndarray) -> List[pd.DataFrame]:
    """Split metadata to mirror interval blocks.

    >>> intervals = np.array([0.5, 1.0, np.nan, 0.75])
    >>> meta = pd.DataFrame({'label': ['a', 'b', 'nan', 'c']})
    >>> [block['label'].tolist() for block in _split_meta_blocks(meta, intervals)]
    [['a', 'b'], ['c']]
    """

    blocks: List[pd.DataFrame] = []
    start = 0
    for idx, value in enumerate(intervals):
        if np.isnan(value):
            block = meta.iloc[start:idx]
            if len(block) > 0:
                blocks.append(block.reset_index(drop=True))
            start = idx + 1
    block = meta.iloc[start:]
    if len(block) > 0:
        blocks.append(block.reset_index(drop=True))
    return blocks


def _coerce_meta_frame(
    meta: Any,
    expected_rows: int,
    missing_rows_message: str,
) -> pd.DataFrame:
    """Return *meta* as a DataFrame with ``expected_rows`` rows.

    Returns an empty DataFrame with the requested number of rows
    when meta is None.

    Parameters
    ----------
    meta : Any
        Input metadata convertible to :class:`pandas.DataFrame`.
    expected_rows : int
        Required number of rows once converted.
    missing_rows_message : str
        Error message raised when the row count does not match.
    """

    if meta is None:
        return pd.DataFrame(index=pd.RangeIndex(expected_rows))

    if isinstance(meta, pd.DataFrame):
        meta_df = meta
    else:
        try:
            meta_df = pd.DataFrame(meta)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("meta must be convertible to a pandas DataFrame") from exc
    meta_df = meta_df.reset_index(drop=True)
    if len(meta_df) != expected_rows:
        raise ValueError(missing_rows_message)
    return meta_df


def _aggregate_meta(
    meta_blocks: Iterable[Optional[pd.DataFrame]],
    blocks: Iterable[np.ndarray],
    length: int,
    meta_aggregator: Callable[[pd.DataFrame], Mapping[str, Any]],
    expected_segments: int,
) -> pd.DataFrame:
    """Aggregate per-interval metadata into per-segment records."""

    aggregated_meta: List[Mapping[str, Any]] = []
    for block_meta, block in zip(meta_blocks, blocks):
        if block_meta is None:
            continue
        if len(block_meta) != len(block):
            raise ValueError("meta rows must match intervals within each block")
        for start in range(len(block) - length + 1):
            window = block_meta.iloc[start : start + length]
            aggregated = meta_aggregator(window)
            if isinstance(aggregated, pd.Series):
                aggregated = aggregated.to_dict()
            elif not isinstance(aggregated, Mapping):
                raise TypeError(
                    "meta_aggregator must return a mapping or pandas Series"
                )
            aggregated_meta.append(aggregated)

    if len(aggregated_meta) != expected_segments:
        raise ValueError("Aggregated metadata must match number of segments")
    return pd.DataFrame(aggregated_meta)


def extract_segments(
    intervals: Iterable[float],
    length: int,
    *,
    warn_on_short: bool = True,
    copy: bool = True,
    allow_zero: bool = False,
    drop_zeros: bool = False,
) -> np.ndarray:
    """Return a vectorized sliding-window matrix of interval segments.

    Parameters
    ----------
    intervals : Iterable[float]
        Contiguous numeric intervals. Inputs containing ``np.nan`` must be
        pre-split via :func:`split_blocks`.
    length : int
        Window size of each produced segment.
    warn_on_short : bool, optional
        Emit a :class:`UserWarning` when the data is shorter than ``length`` and
        no segments can be formed.
    copy : bool, optional
        Return a copy of the data (default) instead of a view.
    allow_zero, drop_zeros : bool, optional
        Control whether zero-valued intervals are permitted or removed.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_segments, length)`` containing the extracted
        segments.

    Examples
    --------
    >>> import numpy as np
    >>> extract_segments(np.arange(1, 6, dtype=float), 3)
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> extract_segments([1, 0, 2], 2, allow_zero=True)
    array([[1., 0.],
           [0., 2.]])
    """

    if length < 1:
        raise ValueError("length must be a positive integer")

    arr = np.asarray(intervals, dtype=float)
    if arr.ndim != 1:
        raise ValueError("intervals must be one-dimensional")

    if drop_zeros:
        arr = arr[arr != 0]
    elif not allow_zero and np.any(arr == 0):
        raise ValueError("intervals contain zeros; enable allow_zero or drop_zeros")

    if np.any(np.isnan(arr)):
        raise ValueError(
            "Intervals contain NaN values; preprocess with split_blocks()."
        )

    if arr.size < length:
        if warn_on_short and arr.size > 0:
            warnings.warn(
                "Encountered data shorter than the requested segment length; skipping it.",
                UserWarning,
            )
        return np.empty((0, length), dtype=float)

    windows = sliding_window_view(arr, length)
    return windows.copy() if copy else windows


def normalize_segments(
    segments: Iterable[Iterable[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize each segment to sum to one and return scaling factors.

    >>> normalize_segments([[1, 1], [2, 1]])
    (array([[0.5       , 0.5       ],
           [0.66666667, 0.33333333]]), array([2., 3.]))
    """

    segments_arr = np.asarray(segments, dtype=float)
    if segments_arr.ndim != 2:
        raise ValueError("segments must be a 2D iterable of numeric values")
    if segments_arr.shape[0] == 0:
        return segments_arr.copy(), np.asarray([], dtype=float)
    duration = segments_arr.sum(axis=1)
    normalized = np.divide(
        segments_arr,
        duration[:, np.newaxis],
        out=np.zeros_like(segments_arr),
        where=duration[:, np.newaxis] != 0,
    )
    return normalized, duration


@dataclass(frozen=True)
class RhythmicSegments:
    """Immutable container for rhythmic segment matrices.

    >>> rs = RhythmicSegments.from_intervals([0.5, 1.0, 0.75, 1.25], length=2)
    >>> rs.segments.shape
    (3, 2)
    >>> rs.durations
    array([1.5 , 1.75, 2.  ], dtype=float32)
    """

    segments: np.ndarray
    patterns: np.ndarray
    durations: np.ndarray
    length: int
    meta: pd.DataFrame

    def __repr__(self) -> str:
        count = self.count
        summary = f"RhythmicSegments(segment_length={self.length}, n_segments={count}"

        if not self.meta.empty:
            meta_cols = ", ".join(str(col) for col in self.meta.columns)
            summary += f", meta_columns=[{meta_cols}]"
        else:
            summary += ", n_meta_cols=0"

        max_preview = min(count, 3)
        if max_preview:
            preview_rows = ", ".join(
                np.array2string(
                    self.segments[i], precision=3, separator=", ", max_line_width=75
                )
                for i in range(max_preview)
            )
            if count > max_preview:
                preview_rows += ", ..."
            summary += f", segments=[{preview_rows}]"

        summary += ")"
        return summary

    @staticmethod
    def from_segments(
        segments: Iterable[Iterable[float]],
        *,
        length: Optional[int] = None,
        meta: Optional[Any] = None,
        dtype=np.dtype("float32"),
    ) -> "RhythmicSegments":
        """Create an instance from a precomputed segment matrix.

        Parameters
        ----------
        segments : Iterable[Iterable[float]]
            Matrix of segment data.
        length : Optional[int]
            Expected segment length. Required when ``segments`` is empty and must
            be at least ``2``.
        meta : Optional[Any]
            Per-segment metadata; anything convertible to :class:`pandas.DataFrame`
            with one row per segment.
        dtype : data-type, optional
            Target dtype for the internal arrays. Defaults to ``np.float32``.

        Examples
        --------
        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]], meta={'label': ['a', 'b']})
        >>> rs.segments
        array([[1., 2.],
               [3., 4.]], dtype=float32)
        >>> list(rs.meta['label'])
        ['a', 'b']
        """

        segments = np.asarray(segments, dtype=dtype)
        if segments.ndim != 2:
            raise ValueError("segments must be a 2D iterable of numeric values")

        if segments.shape[0] == 0:
            if length is None:
                raise ValueError("length must be provided when segments are empty")
            segments = segments.reshape(0, length)
        inferred_length = segments.shape[1]
        if length is None:
            length = inferred_length
        elif length != inferred_length:
            raise ValueError("Provided length does not match segment width")
        if length < 2:
            raise ValueError("segment length must be at least 2")

        patterns, durations = normalize_segments(segments)

        meta_df = _coerce_meta_frame(
            meta,
            expected_rows=len(segments),
            missing_rows_message="meta must have the same number of rows as segments",
        )

        return RhythmicSegments(
            np.ascontiguousarray(segments, dtype=dtype),
            np.ascontiguousarray(patterns, dtype=dtype),
            np.ascontiguousarray(durations, dtype=dtype),
            length,  # type: ignore
            meta_df,
        )

    @staticmethod
    def from_intervals(
        intervals: Iterable[Any],
        length: int,
        *,
        split_at_nan: bool = True,
        warn_on_short: bool = True,
        copy: bool = True,
        drop_zeros: bool = False,
        allow_zero: bool = False,
        dtype: np.dtype = np.dtype("float32"),
        meta: Optional[Any] = None,
        meta_aggregator: Optional[Callable[[pd.DataFrame], Mapping[str, Any]]] = None,
    ) -> "RhythmicSegments":
        """Create an instance from sequential interval data.

        Parameters
        ----------
        intervals : Iterable[Any]
            Contiguous numeric intervals to window. Inputs containing ``np.nan``
            delimiters can be handled by enabling ``split_at_nan``.
        length : int
            Segment length. Must be at least ``2``.
        split_at_nan : bool, optional
            If ``True`` (default) split the interval stream on ``np.nan``
            boundaries before extraction.
        warn_on_short, copy, drop_zeros, allow_zero : bool
            Forwarded to :func:`extract_segments` for each contiguous block.
        dtype : numpy.dtype, optional
            Target dtype for the internal arrays passed to :meth:`from_segments`.
        meta : Optional[Any]
            Optional metadata with one row per input interval. Anything that can
            be converted to :class:`pandas.DataFrame` is accepted. Rows
            corresponding to ``np.nan`` boundaries are dropped automatically
            when ``split_at_nan`` is ``True``.
        meta_aggregator : Optional[Callable[[pandas.DataFrame], Mapping[str, Any]]]
            Aggregation function that converts per-interval metadata into a
            single record for each produced segment. Required when ``meta`` is
            supplied.

        Examples
        --------

        >>> rs = RhythmicSegments.from_intervals([0.5, 1.0, 0.75, 1.25], length=2)
        >>> rs.segments
        array([[0.5 , 1.  ],
               [1.  , 0.75],
               [0.75, 1.25]], dtype=float32)
        >>> rs.patterns
        array([[0.33333334, 0.6666667 ],
               [0.5714286 , 0.42857143],
               [0.375     , 0.625     ]], dtype=float32)
        >>> rs.durations
        array([1.5 , 1.75, 2.  ], dtype=float32)

        By default, np.nan values are treated as boundaries between blocks of intervals.
        Segments are not allowed to cross such boundaries, as in the following example.
        This behaviour can be disabled using `split_at_nan=False`.

        >>> intervals = [1, 2, 3, np.nan, 4, 5, np.nan, 6, 7, 8]
        >>> rs = RhythmicSegments.from_intervals(intervals, length=2)
        >>> rs.segments
        array([[1., 2.],
           [2., 3.],
           [4., 5.],
           [6., 7.],
           [7., 8.]], dtype=float32)

        You can also pass metadata. It has to have the same shape as the intervals: rows corresponding
        to NaN intervals will be dropped, essentially. An aggregator function specifies how meta rows
        for all intervals in a segment are combined into the metadata for that segment. Here is an
        example where the labels of intervals in a segment are joined by dashes to form a segment label.

        >>> intervals = [0.5, 1.0, np.nan, 0.75, 1.0]
        >>> meta = {'label': ['a', 'b', 'nan', 'c', 'd']}
        >>> agg = lambda df: {'labels': '-'.join(df['label'])}
        >>> rs = RhythmicSegments.from_intervals(intervals, length=2, meta=meta, meta_aggregator=agg)
        >>> rs.segments
        array([[0.5 , 1.  ],
           [0.75, 1.  ]], dtype=float32)
        >>> list(rs.meta['labels'])
        ['a-b', 'c-d']

        If the number of intervals is smaller than the segment length, a warning is thrown,
        this can be turned off using the warn_on_short flag:

        >>> rs = RhythmicSegments.from_intervals([1, 2], length=3, warn_on_short=False)

        """

        arr = np.asarray(list(intervals), dtype=float)

        blocks: List[np.ndarray]
        if split_at_nan and np.any(np.isnan(arr)):
            blocks = split_blocks(arr, drop_empty=True, copy=False)
        else:
            if np.isnan(arr).any():
                raise ValueError(
                    "Intervals contain NaN values; enable split_at_nan or preprocess via split_blocks()."
                )
            blocks = [arr]

        matrices: List[np.ndarray] = []
        for block in blocks:
            block_segments = extract_segments(
                block,
                length,
                warn_on_short=warn_on_short,
                copy=copy,
                drop_zeros=drop_zeros,
                allow_zero=allow_zero,
            )
            count = block_segments.shape[0]
            if count:
                matrices.append(block_segments)

        if matrices:
            segments_matrix = np.concatenate(matrices, axis=0)
        else:
            segments_matrix = np.empty((0, length), dtype=float)

        if meta is None:
            meta_df = None
        else:
            meta_df = _coerce_meta_frame(
                meta,
                expected_rows=len(arr),
                missing_rows_message="meta must have the same number of rows as intervals",
            )
            meta_blocks = (
                _split_meta_blocks(meta_df, arr) if split_at_nan else [meta_df]
            )
            if meta_aggregator is None:
                raise ValueError(
                    "meta_aggregator must be provided when meta is supplied"
                )
            meta_df = _aggregate_meta(
                meta_blocks,
                blocks,
                length,
                meta_aggregator,
                segments_matrix.shape[0],
            )

        return RhythmicSegments.from_segments(
            segments_matrix, length=length, meta=meta_df, dtype=dtype
        )

    @staticmethod
    def from_events(
        events: Iterable[Any],
        length: int,
        *,
        drop_na: bool = False,
        split_at_nan: bool = True,
        allow_zero_intervals: bool = False,
        warn_on_short: bool = True,
        copy: bool = True,
        dtype: np.dtype = np.dtype("float32"),
        meta: Optional[Any] = None,
        meta_aggregator: Optional[Callable[[pd.DataFrame], Mapping[str, Any]]] = None,
    ) -> "RhythmicSegments":
        """Create an instance from timestamped event data.

        Parameters
        ----------
        events : Iterable[Any]
            Monotonic (or at least ordered) series of onset timestamps. Must be
            convertible to ``float``.
        length : int
            Segment length passed to :meth:`from_intervals`. Must be at least ``2``.
        drop_na : bool, optional
            Remove ``NaN`` timestamps before differencing. When ``False``
            (default), the resulting interval stream will contain ``NaN``
            markers wherever the original event data did, which in turn act as
            block boundaries for :meth:`from_intervals`.
        split_at_nan : bool, optional
            Forwarded to :meth:`from_intervals`; controls whether extracted
            segments can span across ``NaN`` interval boundaries.
        allow_zero_intervals : bool, optional
            Allow identical consecutive events (zero-length intervals). When
            ``False`` (default) such ties raise a :class:`ValueError`.
        warn_on_short, copy : bool, optional
            Forwarded to :meth:`from_intervals`.
        dtype : numpy.dtype, optional
            Target dtype for the internal arrays passed to :meth:`from_segments`.
        meta : Optional[Any]
            Optional metadata aligned with the derived intervals, i.e. it must
            contain exactly ``len(events) - 1`` rows after any ``NaN`` removal.
        meta_aggregator : Optional[Callable[[pandas.DataFrame], Mapping[str, Any]]]
            Forwarded to :meth:`from_intervals`.

        Examples
        --------
        >>> events = [0.0, 0.5, 1.0, np.nan, 1.5, 2.0, 2.5]
        >>> rs = RhythmicSegments.from_events(events, length=2)
        >>> rs.segments
        array([[0.5, 0.5],
           [0.5, 0.5]], dtype=float32)

        Segments never span the ``np.nan`` boundary. To discard the boundary
        entirely, enable ``drop_na=True``:

        >>> RhythmicSegments.from_events(events, length=2, drop_na=True).segments
        array([[0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5]], dtype=float32)

        Passing ``split_at_nan=False`` while retaining the ``NaN`` intervals
        raises an error because :meth:`from_intervals` forbids segments crossing
        the boundary:

        >>> RhythmicSegments.from_events(events, length=2, split_at_nan=False)
        Traceback (most recent call last):
        ...
        ValueError: Intervals contain NaN values; enable split_at_nan or preprocess via split_blocks().
        """

        events_arr = np.asarray(list(events), dtype=float)
        if events_arr.ndim != 1:
            raise ValueError("events must be one-dimensional")

        if drop_na:
            events_arr = events_arr[~np.isnan(events_arr)]

        if events_arr.size > 1:
            # Note that this results in two np.na values for every np.na in the input.
            # However, from_intervals handles that fine, so that's no problem.
            intervals = np.diff(events_arr)
        else:
            intervals = np.empty(0, dtype=float)

        finite_intervals = intervals[np.isfinite(intervals)]
        if np.any(finite_intervals < 0):
            raise ValueError("events must be in non-decreasing order")
        if not allow_zero_intervals and np.any(finite_intervals == 0):
            raise ValueError(
                "events contain zero-length intervals; enable allow_zero_intervals=True to permit ties"
            )

        interval_meta: Optional[pd.DataFrame]
        if meta is None:
            interval_meta = None
        else:
            expected_rows = intervals.size
            interval_meta = _coerce_meta_frame(
                meta,
                expected_rows=expected_rows,
                missing_rows_message=(
                    "meta must have the same number of rows as derived intervals "
                    "(len(events) - 1)"
                ),
            )

        return RhythmicSegments.from_intervals(
            intervals,
            length=length,
            split_at_nan=split_at_nan,
            warn_on_short=warn_on_short,
            copy=copy,
            allow_zero=allow_zero_intervals,
            dtype=dtype,
            meta=interval_meta,
            meta_aggregator=meta_aggregator,
        )

    @staticmethod
    def concat(
        *segments: "RhythmicSegments",
        source_col: Optional[str] = None,
    ) -> "RhythmicSegments":
        """Concatenate multiple :class:`RhythmicSegments` objects.

        Metadata columns are merged using :func:`pandas.concat`; missing values
        are filled with ``NaN`` as usual.

        Parameters
        ----------
        segments : RhythmicSegments
            Objects to concatenate.
        source_col : Optional[str]
            Name of a metadata column storing the positional index of the source
            object. ``None`` disables the column.

        Examples
        --------
        >>> rs1 = RhythmicSegments.from_segments([[1, 2]], meta=dict(label=['a']))
        >>> rs2 = RhythmicSegments.from_segments([[3, 4]], meta=dict(label=['b']))
        >>> merged = RhythmicSegments.concat(rs1, rs2, source_col='source')
        >>> merged.segments
        array([[1., 2.],
               [3., 4.]], dtype=float32)
        >>> list(merged.meta['source'])
        [0, 1]
        """

        if not segments:
            raise ValueError("At least one RhythmicSegments object is required")
        if len(segments) == 1:
            return segments[0]

        first = segments[0]
        length = first.length
        dtype = first.segments.dtype

        seg_arrays = []
        pat_arrays = []
        dur_arrays = []
        meta_frames = []

        for seg in segments:
            if seg.length != length:
                raise ValueError(
                    "All rhythmic segments must have the same segment length (number of columns)"
                )
            seg_arrays.append(np.ascontiguousarray(seg.segments, dtype=dtype))
            pat_arrays.append(np.ascontiguousarray(seg.patterns, dtype=dtype))
            dur_arrays.append(np.ascontiguousarray(seg.durations, dtype=dtype))
            meta_frames.append(seg.meta)

        combined_segments = np.concatenate(seg_arrays, axis=0)
        combined_patterns = np.concatenate(pat_arrays, axis=0)
        combined_durations = np.concatenate(dur_arrays, axis=0)

        combined_meta = pd.concat(
            meta_frames, ignore_index=True, sort=False
        ).reset_index(drop=True)
        if source_col is not None:
            indices = [np.repeat(i, len(seg.meta)) for i, seg in enumerate(segments)]
            combined_meta[source_col] = (
                np.concatenate(indices) if indices else np.array([], dtype=int)
            )

        return RhythmicSegments(
            combined_segments,
            combined_patterns,
            combined_durations,
            length,
            combined_meta,
        )

    @property
    def count(self) -> int:
        """Number of stored segments."""

        return int(self.segments.shape[0])

    def take(self, idx: Union[np.ndarray, List[int]]) -> "RhythmicSegments":
        """Return a new instance containing only the segments at *idx*.

        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]], meta=dict(id=[0, 1]))
        >>> rs.take([1]).segments
        array([[3., 4.]], dtype=float32)
        >>> list(rs.take([1]).meta['id'])
        [1]
        """
        idx_arr = np.asarray(idx)
        return replace(
            self,
            segments=self.segments[idx_arr],
            patterns=self.patterns[idx_arr],
            durations=self.durations[idx_arr],
            meta=self.meta.iloc[idx_arr].reset_index(drop=True),
        )

    def filter(self, mask: Union[np.ndarray, pd.Series]) -> "RhythmicSegments":
        """Return a new instance containing segments where *mask* is true.

        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]], meta=dict(id=[0, 1]))
        >>> rs.filter([True, False]).segments
        array([[1., 2.]], dtype=float32)
        """
        mask_arr = np.asarray(mask, dtype=bool)
        return self.take(np.nonzero(mask_arr)[0])

    def filter_by_duration(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_quantile: Optional[float] = None,
        max_quantile: Optional[float] = None,
    ) -> "RhythmicSegments":
        """Return a new instance filtered by duration thresholds.

        Parameters
        ----------
        min_value, max_value : Optional[float], optional
            Absolute duration bounds (inclusive). When supplied, these override
            the corresponding quantile parameters.
        min_quantile, max_quantile : Optional[float], optional
            Quantile-based bounds (inclusive) used when explicit ``min_value`` or
            ``max_value`` are not provided. Pass ``None`` to disable a bound.

        Examples
        --------
        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4], [5, 6]])
        >>> rs.durations
        array([ 3.,  7., 11.], dtype=float32)
        >>> short = rs.filter_by_duration(max_quantile=0.5)
        >>> short.durations
        array([3., 7.], dtype=float32)
        >>> rs.filter_by_duration(min_value=8.0).durations
        array([11.], dtype=float32)
        >>> rs.filter_by_duration(min_value=3.0, max_value=8.0).durations
        array([3., 7.], dtype=float32)
        >>> rs.filter_by_duration()
        Traceback (most recent call last):
        ...
        ValueError: At least one duration bound must be specified
        """

        if (
            min_value is None
            and max_value is None
            and min_quantile is None
            and max_quantile is None
        ):
            raise ValueError("At least one duration bound must be specified")

        if self.count == 0:
            return self

        durations = self.durations

        lower_bound: Optional[float]
        if min_value is not None:
            lower_bound = float(min_value)
        elif min_quantile is not None:
            if not 0.0 <= min_quantile <= 1.0:
                raise ValueError("min_quantile must be between 0 and 1")
            lower_bound = float(np.quantile(durations, min_quantile))
        else:
            lower_bound = None

        upper_bound: Optional[float]
        if max_value is not None:
            upper_bound = float(max_value)
        elif max_quantile is not None:
            if not 0.0 <= max_quantile <= 1.0:
                raise ValueError("max_quantile must be between 0 and 1")
            upper_bound = float(np.quantile(durations, max_quantile))
        else:
            upper_bound = None

        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("Lower duration bound exceeds upper bound")

        if lower_bound is not None and upper_bound is not None:
            return self.filter((durations >= lower_bound) & (durations <= upper_bound))
        if lower_bound is not None:
            return self.filter(durations >= lower_bound)
        return self.filter(durations <= upper_bound)

    def with_meta(self, **cols: Any) -> "RhythmicSegments":
        """Return a new instance with additional metadata columns.

        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]])
        >>> rs.with_meta(label=['a', 'b']).meta['label'].tolist()
        ['a', 'b']
        """
        new_meta = self.meta.assign(**cols)
        if len(new_meta) != self.count:
            raise ValueError(
                "Meta assignment must maintain the same number of rows as segments"
            )
        return replace(self, meta=new_meta.reset_index(drop=True))

    def query(self, expr: str, **query_kwargs: Any) -> "RhythmicSegments":
        """Return a new instance filtered by evaluating *expr* on the metadata.

        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]], meta={'id': [0, 1]})
        >>> rs.query('id == 1').segments
        array([[3., 4.]], dtype=float32)
        """

        mask = self.meta.query(expr, **query_kwargs).index.to_numpy()
        return self.take(mask)

    def shuffle(self, random_state: Optional[int] = None) -> "RhythmicSegments":
        """Return a new instance with rows shuffled uniformly at random.

        >>> rs = RhythmicSegments.from_segments([[1, 2], [3, 4]])
        >>> rs.shuffle(random_state=3).segments
        array([[3., 4.],
           [1., 2.]], dtype=float32)
        """

        rng = np.random.default_rng(random_state)
        idx = rng.permutation(self.count)
        return self.take(idx)
