"""Metadata-handling utilities for rhythmic segments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Aggregator = Callable[[pd.DataFrame], Mapping[str, Any]]


def coerce_meta_frame(
    meta: Any,
    expected_rows: Optional[int] = None,
    missing_rows_message: str = "meta must match the expected row count",
    *,
    columns: Optional[Iterable[str]] = None,
    constants: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Return *meta* as a :class:`pandas.DataFrame`."""

    if meta is None:
        if expected_rows is None:
            return pd.DataFrame()
        return pd.DataFrame(index=pd.RangeIndex(expected_rows))

    if isinstance(meta, pd.DataFrame):
        meta_df = meta
    else:
        try:
            meta_df = pd.DataFrame(meta)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("meta must be convertible to a pandas DataFrame") from exc
    meta_df = meta_df.reset_index(drop=True)
    if expected_rows is not None and len(meta_df) != expected_rows:
        raise ValueError(missing_rows_message)
    if columns is not None:
        columns = list(columns)
        missing = [col for col in columns if col not in meta_df.columns]
        if missing:
            raise KeyError(f"Columns {missing!r} not found in metadata")
        meta_df = meta_df.loc[:, columns]
    if constants:
        meta_df = meta_df.assign(**dict(constants))
    return meta_df


def aggregate_meta(
    meta_blocks: Iterable[pd.DataFrame],
    value_blocks: Iterable[np.ndarray],
    window_len: int,
    meta_agg: Aggregator,
    expected_records: int,
) -> pd.DataFrame:
    """Aggregate metadata windows aligned with numeric value blocks."""

    aggregated_meta: List[Mapping[str, Any]] = []
    for meta_block, value_block in zip(meta_blocks, value_blocks):
        if len(meta_block) != len(value_block):
            raise ValueError("meta rows must match values within each block")
        if len(meta_block) < window_len:
            continue
        for start in range(len(meta_block) - window_len + 1):
            window = meta_block.iloc[start : start + window_len]
            aggregated = meta_agg(window)
            if isinstance(aggregated, pd.Series):
                aggregated = aggregated.to_dict()
            elif not isinstance(aggregated, Mapping):
                raise TypeError("meta_agg must return a mapping or pandas Series")
            aggregated_meta.append(aggregated)

    if len(aggregated_meta) != expected_records:
        raise ValueError("Aggregated metadata must match number of produced records")
    return pd.DataFrame(aggregated_meta)


def resolve_columns_and_names(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    resolved_columns = list(df.columns) if columns is None else list(columns)
    missing = [col for col in resolved_columns if col not in df]
    if missing:
        raise KeyError(f"Columns {missing!r} not found in metadata")
    if names is None:
        return resolved_columns, resolved_columns
    resolved_names = list(names)
    if len(resolved_names) != len(resolved_columns):
        raise ValueError("'names' must match the number of selected columns")
    return resolved_columns, resolved_names


def agg_copy(
    df: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return every column value keyed by its position.

    >>> import pandas as pd
    >>> agg_copy(pd.DataFrame({"label": ["a", "b"]}))
    {'label_1': 'a', 'label_2': 'b'}
    """

    columns_list, names_list = resolve_columns_and_names(df, columns, names)
    data: Dict[str, Any] = {}
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        for col, alias in zip(columns_list, names_list):
            data[f"{alias}_{idx}"] = row[col]
    return data


def agg_index(
    df: pd.DataFrame,
    index: int,
    *,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return metadata from the row at ``index``.

    >>> import pandas as pd
    >>> agg_index(pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]}), 1)
    {'label': 'b', 'section': 'y'}
    >>> agg_index(pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]}), 1, columns=["label"])
    {'label': 'b'}
    """

    if df.empty:
        raise ValueError("Cannot aggregate metadata from an empty DataFrame")
    columns_list, names_list = resolve_columns_and_names(df, columns, names)
    row = df.iloc[index]
    return {alias: row[col] for col, alias in zip(columns_list, names_list)}


def agg_first(
    df: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return metadata from the first interval.

    >>> import pandas as pd
    >>> agg_first(pd.DataFrame({"label": ["a", "b"]}))
    {'label': 'a'}
    """

    return agg_index(df, 0, columns=columns, names=names)


def agg_last(
    df: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return metadata from the last interval.

    >>> import pandas as pd
    >>> agg_last(pd.DataFrame({"label": ["a", "b"]}))
    {'label': 'b'}
    """

    return agg_index(df, -1, columns=columns, names=names)


def agg_join(
    df: pd.DataFrame,
    *,
    separator: str = "-",
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Join values from each column.

    >>> import pandas as pd
    >>> agg_join(pd.DataFrame({"label": ["a", "b"], "section": ["x", "y"]}), separator="|")
    {'label': 'a|b', 'section': 'x|y'}
    """

    columns_list, names_list = resolve_columns_and_names(df, columns, names)
    return {
        alias: separator.join(df[column].astype(str))
        for column, alias in zip(columns_list, names_list)
    }


def agg_list(
    df: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return metadata as lists per column.

    >>> import pandas as pd
    >>> agg_list(pd.DataFrame({"label": ["a", "b"]}))
    {'label': ['a', 'b']}
    """

    columns_list, names_list = resolve_columns_and_names(df, columns, names)
    return {alias: list(df[column]) for column, alias in zip(columns_list, names_list)}


_DEFAULT_AGGREGATORS: Dict[str, Callable[..., Aggregator]] = {
    "copy": lambda **kwargs: lambda df: agg_copy(df, **kwargs),
    "index": lambda **kwargs: lambda df: agg_index(df, **kwargs),
    "first": lambda **kwargs: lambda df: agg_first(df, **kwargs),
    "last": lambda **kwargs: lambda df: agg_last(df, **kwargs),
    "join": lambda **kwargs: lambda df: agg_join(df, **kwargs),
    "list": lambda **kwargs: lambda df: agg_list(df, **kwargs),
}


def get_aggregator(name: str, **kwargs: Any) -> Aggregator:
    """Return a named metadata aggregator.

    Supported names are ``"copy"``, ``"index"``, ``"first"``, ``"last"``,
    ``"join"``, and ``"list"``. Additional keyword arguments are forwarded to
    the underlying aggregator.

    >>> import pandas as pd
    >>> agg = get_aggregator("first", columns=["label"])
    >>> agg(pd.DataFrame({"label": ["a", "b"]}))
    {'label': 'a'}
    >>> join = get_aggregator("join", columns=["label"], separator="|")
    >>> join(pd.DataFrame({"label": ["a", "b"]}))
    {'label': 'a|b'}
    """

    key = name.lower()
    if key not in _DEFAULT_AGGREGATORS:
        available = ", ".join(sorted(_DEFAULT_AGGREGATORS))
        raise ValueError(f"Unknown aggregator '{name}'. Available: {available}")
    factory = _DEFAULT_AGGREGATORS[key]
    return factory(**kwargs)
