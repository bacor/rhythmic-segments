"""Utility functions for working with rhythmic segments.

This module provides helpers for constructing and manipulating n-gram segments
that represent consecutive rhythmic events.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any, Iterable, List, Optional, Union, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from .helpers import split_into_blocks
from .meta import Aggregator, aggregate_meta, coerce_meta_frame, get_aggregator


def extract_segments(
    intervals: Iterable[float],
    length: int,
    *,
    warn_on_short: bool = True,
    copy: bool = True,
    check_zero_intervals: bool = True,
    check_nan_intervals: bool = True,
) -> np.ndarray:
    """Return a vectorized sliding-window matrix of interval segments.

    Parameters
    ----------
    intervals : Iterable[float]
        Contiguous numeric intervals. Inputs containing ``np.nan`` must be
        pre-split via :func:`rhythmic_segments.helpers.split_into_blocks`.
    length : int
        Window size of each produced segment.
    warn_on_short : bool, optional
        Emit a :class:`UserWarning` when the data is shorter than ``length`` and
        no segments can be formed.
    copy : bool, optional
        Return a copy of the data (default) instead of a view.
    check_zero_intervals, check_nan_intervals : bool, optional
        Enable validation that forbids zero or NaN intervals, respectively.

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
    >>> extract_segments([1, 0, 2], 2, check_zero_intervals=False)
    array([[1., 0.],
           [0., 2.]])
    """

    if length < 1:
        raise ValueError("length must be a positive integer")

    arr = np.asarray(intervals, dtype=float)
    if arr.ndim != 1:
        raise ValueError("intervals must be one-dimensional")

    if check_zero_intervals and np.any(arr == 0):
        raise ValueError(
            "intervals contain zeros; disable check_zero_intervals to allow them"
        )

    if check_nan_intervals and np.any(np.isnan(arr)):
        raise ValueError(
            "Intervals contain NaN values; disable check_nan_intervals or preprocess with split_into_blocks()."
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


def process_input_data(
    data: Any,
    *,
    column: Optional[str],
    meta: Optional[Any],
    meta_columns: Optional[Iterable[str]],
    meta_constants: Optional[Mapping[str, Any]],
    data_label: str,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """Return numeric data and processed metadata extracted from *data*.

    When *column* is provided and *meta* is supplied, the explicit metadata
    takes precedence over the inferred DataFrame columns. Metadata selection
    via *meta_columns* and constant assignments from *meta_constants* are
    applied before returning the DataFrame.
    """
    # No column specified; no need to separate input from metadata
    if column is None:
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                f"When passing a DataFrame to {data_label}, 'column' must be specified."
            )
        if isinstance(data, np.ndarray):
            vector = data.astype(float, copy=False)
        elif isinstance(data, pd.Series):
            vector = data.to_numpy(dtype=float, copy=False)
        else:
            vector = np.asarray(list(data), dtype=float)
        meta_source = meta

    # Separate metadata from input data
    else:
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            try:
                df = pd.DataFrame(data)
            except Exception as exc:  # pragma: no cover - defensive
                raise TypeError(
                    f"{data_label.capitalize()} must be convertible to a pandas DataFrame when 'column' is provided."
                ) from exc
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in provided {data_label}.")
        vector = df[column].to_numpy(dtype=float, copy=False)
        inferred_meta = df.drop(columns=[column]).reset_index(drop=True)
        meta_source = meta if meta is not None else inferred_meta

    expected_rows = len(vector)
    if meta_source is None and meta_columns is None and meta_constants is None:
        return vector, None

    meta_frame_source = (
        meta_source
        if meta_source is not None
        else pd.DataFrame(index=pd.RangeIndex(expected_rows))
    )
    meta_df = coerce_meta_frame(
        meta_frame_source,
        expected_rows=expected_rows,
        missing_rows_message=f"meta must have the same number of rows as {data_label}",
        columns=meta_columns,
        constants=meta_constants,
    )
    return vector, meta_df


_AGG_COPY: Aggregator = get_aggregator("copy")
_AGG_FIRST: Aggregator = get_aggregator("first")


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
        meta_columns: Optional[Iterable[str]] = None,
        meta_constants: Optional[Mapping[str, Any]] = None,
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
        meta_columns : Optional[Iterable[str]], optional
            Names of metadata columns to retain. When ``None`` all columns are
            kept.
        meta_constants : Optional[Mapping[str, Any]], optional
            Constant metadata columns to add.
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
        if length < 2:  # type: ignore
            raise ValueError("segment length must be at least 2")

        patterns, durations = normalize_segments(segments)

        meta_df = coerce_meta_frame(
            meta,
            expected_rows=len(segments),
            missing_rows_message="meta must have the same number of rows as segments",
            columns=meta_columns,
            constants=meta_constants,
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
        check_zero_intervals: bool = True,
        column: Optional[str] = None,
        meta: Optional[Any] = None,
        meta_columns: Optional[Iterable[str]] = None,
        meta_constants: Optional[Mapping[str, Any]] = None,
        meta_agg: Optional[Aggregator] = _AGG_COPY,
        **from_segments_kwargs: Any,
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
        warn_on_short, copy, check_zero_intervals : bool, optional
            Forwarded to :func:`extract_segments` (see that function for details).
        column : Optional[str], optional
            When provided, ``intervals`` must be DataFrame-like and the selected
            column supplies the numeric intervals. All remaining columns are
            treated as metadata.
        meta : Optional[Any]
            Optional metadata with one row per input interval. Anything that can
            be converted to :class:`pandas.DataFrame` is accepted. Ignored when
            ``column`` is provided. Rows corresponding to ``np.nan`` boundaries
            are dropped automatically when ``split_at_nan`` is ``True``.
        meta_columns : Optional[Iterable[str]], optional
            Names of metadata columns to retain. When ``None`` all columns are
            kept.
        meta_constants : Optional[Mapping[str, Any]], optional
            Constant metadata columns to add to each resulting segment.
        meta_agg : Aggregator
            Aggregation function that converts per-interval metadata into a
            single record for each produced segment. Defaults to
            :func:`get_aggregator("copy")`.
        **from_segments_kwargs : Any
            Additional keyword arguments forwarded to :meth:`from_segments`.

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
        >>> rs = RhythmicSegments.from_intervals(intervals, length=2, meta=meta, meta_agg=agg)
        >>> rs.segments
        array([[0.5 , 1.  ],
           [0.75, 1.  ]], dtype=float32)
        >>> list(rs.meta['labels'])
        ['a-b', 'c-d']

        If the number of intervals is smaller than the segment length, a warning is thrown,
        this can be turned off using the warn_on_short flag:

        >>> RhythmicSegments.from_intervals([1, 2], length=3)
        Traceback (most recent call last):
        ...
        ValueError: At least three intervals are required to form segments of length >= 2.

        """
        # Coerce input to numpy array and ensure sufficient length
        intervals_arr, interval_meta = process_input_data(
            intervals,
            column=column,
            meta=meta,
            meta_columns=meta_columns,
            meta_constants=None,
            data_label="intervals",
        )
        if meta_agg is None:
            meta_agg = _AGG_COPY
        if intervals_arr.size < 3:
            raise ValueError(
                "At least three intervals are required to form segments of length >= 2."
            )
        if intervals_arr.size < length:
            raise ValueError(
                f"Not enough intervals to form a segment of length {length}; requires at least {length} intervals."
            )

        # Split intervals into blocks
        has_nan = np.isnan(intervals_arr).any()
        if not split_at_nan and has_nan:
            raise ValueError(
                "Intervals contain NaN values; enable split_at_nan or preprocess via split_into_blocks()."
            )
        elif has_nan:
            boundaries = np.isnan(intervals_arr)
            blocks = split_into_blocks(
                intervals_arr,
                boundaries=boundaries,
                drop_empty=True,
                copy=False,
            )
        else:
            blocks = [intervals_arr]

        # Extract all segments from all blocks and combine them
        kws = dict(
            warn_on_short=warn_on_short,
            copy=copy,
            check_zero_intervals=check_zero_intervals,
            check_nan_intervals=False,
        )
        block_segments = [extract_segments(block, length, **kws) for block in blocks]  # type: ignore
        segments = np.concatenate(block_segments, axis=0)

        # Aggregate interval metadata to segment metadata
        segment_meta: Optional[pd.DataFrame] = None
        if interval_meta is not None:
            if split_at_nan and has_nan:
                boundaries = np.isnan(intervals_arr)
                meta_blocks = split_into_blocks(interval_meta, boundaries=boundaries)
            else:
                meta_blocks = [interval_meta]

            segment_meta = aggregate_meta(
                meta_blocks,  # type: ignore
                blocks,
                window_len=length,
                meta_agg=meta_agg,  # type: ignore
                expected_records=segments.shape[0],
            )

        if meta_constants:
            const_dict = dict(meta_constants)
            if segment_meta is None:
                segment_meta = pd.DataFrame(
                    {key: [value] * segments.shape[0] for key, value in const_dict.items()}
                )
            else:
                segment_meta = segment_meta.assign(**const_dict)

        return RhythmicSegments.from_segments(
            segments,
            length=length,
            meta=segment_meta,
            **from_segments_kwargs,
        )

    @staticmethod
    def from_events(
        events: Iterable[Any],
        length: int,
        *,
        drop_nan: bool = False,
        column: Optional[str] = None,
        meta: Optional[Any] = None,
        meta_columns: Optional[Iterable[str]] = None,
        meta_constants: Optional[Mapping[str, Any]] = None,
        interval_meta_agg: Optional[Aggregator] = _AGG_FIRST,
        segment_meta_agg: Optional[Aggregator] = _AGG_COPY,
        **from_intervals_kwargs: Any,
    ) -> "RhythmicSegments":
        """Create an instance from timestamped event data.

        Parameters
        ----------
        events : Iterable[Any]
            Monotonic (or at least ordered) series of onset timestamps. Must be
            convertible to ``float``.
        length : int
            Segment length passed to :meth:`from_intervals`. Must be at least ``2``.
        drop_nan : bool, optional
            Remove ``NaN`` timestamps before differencing. When ``False``
            (default), the resulting interval stream will contain ``NaN``
            markers wherever the original event data did, which in turn act as
            block boundaries for :meth:`from_intervals`.
        column : Optional[str], optional
            When provided, ``events`` must be DataFrame-like and the specified
            column supplies the timestamp values. All remaining columns are
            treated as metadata.
        meta : Optional[Any]
            Optional metadata aligned with the input events. Anything that can be
            converted to :class:`pandas.DataFrame` is accepted. Ignored when
            ``column`` is provided. When
            ``drop_nan=True`` the rows corresponding to dropped events are
            removed automatically.
        meta_columns : Optional[Iterable[str]], optional
            Names of metadata columns to retain. When ``None`` all columns are
            preserved.
        meta_constants : Optional[Mapping[str, Any]], optional
            Constant metadata columns to add to each resulting segment.
        interval_meta_agg : Aggregator
            Aggregation function that combines metadata for pairs of consecutive
            events into per-interval records. Defaults to
            :func:`get_aggregator("first")`.
        segment_meta_agg : Aggregator
            Forwarded to :meth:`from_intervals` to transform per-interval
            metadata into per-segment records. Defaults to
            :func:`get_aggregator("copy")`.
        **from_intervals_kwargs : Any
            Additional keyword arguments forwarded to :meth:`from_intervals`,
            such as ``split_at_nan`` or ``dtype``.

        Examples
        --------
        >>> events = [0.0, 0.5, 1.0, np.nan, 1.5, 2.0, 2.5]
        >>> rs = RhythmicSegments.from_events(events, length=2)
        >>> rs.segments
        array([[0.5, 0.5],
           [0.5, 0.5]], dtype=float32)

        Segments never span the ``np.nan`` boundary. To discard the boundary
        entirely, enable ``drop_nan=True``:

        >>> RhythmicSegments.from_events(events, length=2, drop_nan=True).segments
        array([[0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5]], dtype=float32)

        Note that passing ``split_at_nan=False`` while retaining the ``NaN`` intervals
        will raise an error because :meth:`from_intervals` forbids segments crossing
        the boundary:

        >>> RhythmicSegments.from_events(events, length=2, split_at_nan=False)
        Traceback (most recent call last):
        ...
        ValueError: Intervals contain NaN values; enable split_at_nan or preprocess via split_into_blocks().
        """

        from_intervals_kwargs = dict(from_intervals_kwargs)
        if "meta_agg" in from_intervals_kwargs:
            raise TypeError(
                "'meta_agg' is not an allowed keyword. From_events 'segment_meta_agg' for segment-level aggregation."
            )

        # Validate all input
        events_arr, event_meta = process_input_data(
            events,
            column=column,
            meta=meta,
            meta_columns=meta_columns,
            meta_constants=None,
            data_label="events",
        )
        if meta_constants is not None and "meta_constants" not in from_intervals_kwargs:
            from_intervals_kwargs["meta_constants"] = meta_constants
        if interval_meta_agg is None:
            interval_meta_agg = _AGG_FIRST
        if segment_meta_agg is None:
            segment_meta_agg = _AGG_COPY
        if events_arr.ndim != 1:
            raise ValueError("events must be one-dimensional")
        if events_arr.size < length + 1:
            raise ValueError(
                "Not enough events to form a segment of length "
                f"{length}; requires at least {length + 1} events."
            )

        # Optionally drop NaN entries from both events and metadata
        if drop_nan:
            keep_mask = ~np.isnan(events_arr)
            events_arr = events_arr[keep_mask]
            if event_meta is not None:
                event_meta = event_meta.loc[keep_mask].reset_index(drop=True)

        # Computer intervals (event differences) and check they're positive
        # Note that this results in two np.na values for every np.na in the input.
        # However, from_intervals handles that fine, so that's no problem.
        intervals = np.diff(events_arr)
        finite_intervals = intervals[np.isfinite(intervals)]
        if np.any(finite_intervals < 0):
            raise ValueError("events must be in non-decreasing order")

        # Aggregate event meta to interval meta
        interval_meta: Optional[pd.DataFrame] = None
        if event_meta is not None:
            interval_meta = aggregate_meta(
                meta_blocks=[event_meta],
                value_blocks=[np.arange(len(event_meta), dtype=float)],
                window_len=2,
                meta_agg=interval_meta_agg,  # type: ignore
                expected_records=intervals.size,
            )

        return RhythmicSegments.from_intervals(
            intervals,
            length=length,
            meta=interval_meta,
            meta_agg=segment_meta_agg,
            **from_intervals_kwargs,
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
