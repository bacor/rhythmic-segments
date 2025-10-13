from rhythmic_segments.utils import is_nan
import numpy as np

def test_is_nan_helper():
    assert is_nan(float("nan"))
    assert is_nan(np.nan)
    assert not is_nan("nan")

