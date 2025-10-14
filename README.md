[![CI](https://github.com/bacor/rhythmic-segments/actions/workflows/ci.yaml/badge.svg)](https://github.com/bacor/rhythmic-segments/actions/workflows/ci.yaml)
[![docs](https://github.com/bacor/rhythmic-segments/actions/workflows/docs.yaml/badge.svg)](https://github.com/bacor/rhythmic-segments/actions/workflows/docs.yaml)


# Rhythmic Segments

This project provides some basic code to simplify **Rhythmic Segment Analysis** in Python. 
A rhythmic segment analysis (RSA) analyzes every fixed-length segment of a sequence of time intervals: the short groups you obtain by sliding a window across the data. Each segment has a *duration* and a *pattern*. The pattern captures the relative durations of a segment’s intervals, either as a normalized vector or as a ratio. For example, the segment (2, 4, 4) has the pattern (.2, .4, .4) or 1 : 2 : 2; both descriptions are interchangeable. Thinking of patterns as normalized vectors shows that all patterns of a given length live on a simplex: a line for length 2, a triangle for length 3, and so on. The goal is to study rhythmic material by analysing how its segments are distributed of that simplex.

Computing patterns is as simple as normalising the segment, and so you can absolutely do a rhythmic segment analysis without using this package. This package however provides some utilities that make things easier: the RhythmicSegments class allows you to conveniently store large numbers of segments and handle associated metadata.

[For more details, have a look at the docs.](https://bacor.github.io/rhythmic-segments)



## Installation

The package has been tested with Python 3.11 and 3.12.
You can install the package using pip:

```sh
pip install rhythmic-segments
```

## Getting started

```python
from rhythmic_segments import RhythmicSegments

intervals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
rs = RhythmicSegments.from_intervals(intervals, length=3)
rs.segments
# array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], ... ])
```

## License

The code is distributed under an MIT license.

## Contributing

Feel free to contribute via GitHub: https://github.com/bacor/rhythmic-segments

## Citation

A paper describing the idea in more details is currently in preparation. Until a formal reference is available, please cite the repository:

```bibtex
@misc{cornelissen_rhythmic_segments,
  author = {Bas Cornelissen},
  title = {rhythmic_segments},
  howpublished = {\url{https://github.com/bascornelissen/rhythmic-segments}},
  year = {2025},
  note = {Version 0.1.3}
}

