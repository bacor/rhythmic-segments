# Rhythmic Segments

Welcome to the documentation for the `rhythmic_segments` project that helps 
you doing a Rhythmic Segment Analysis in Python. These pages
collect some tutorials and the automatically generated API
reference.

```{admonition} Early development
:class: warning

This project is under active development and the public API may change without
notice. Expect rough edges while things stabilise—and please report anything
you find surprising.
```


## Installation
The package requires a Python version between 3.11 and 3.14.
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

{doc}`Read more about the basic usage <Basic Usage>`


## License

The code is distributed under an MIT license.

## Contributing

Feel free to contribute via GitHub: https://github.com/bacor/rhythmic-segments


## Outline


```{toctree}
:maxdepth: 2

Getting started <self>
Basic usage <Basic Usage>
API Reference <api>
```
