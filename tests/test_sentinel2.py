from types import SimpleNamespace

import numpy as np

from plotscreen.sentinel2 import _offset, nbr, periods


def test_nbr_values():
    out = nbr(np.array([3000.0, 1000.0, 0.0]), np.array([1000.0, 3000.0, 0.0]))
    assert np.allclose(out[:2], [0.5, -0.5]) and np.isnan(out[2])


def test_offset_only_for_new_unshifted_scenes():
    old = SimpleNamespace(properties={"s2:processing_baseline": "02.14"})
    new = SimpleNamespace(properties={"s2:processing_baseline": "05.11"})
    fixed = SimpleNamespace(properties={"s2:processing_baseline": "05.11", "earthsearch:boa_offset_applied": True})
    odd = SimpleNamespace(properties={"s2:processing_baseline": "n/a"})
    assert [_offset(i) for i in (old, new, fixed, odd)] == [0.0, 1000.0, 0.0, 0.0]


def test_periods_use_the_same_months():
    before, after = periods(2020, 2025)
    assert before[4:] == after[4:].replace("2025", "2020") and before.startswith("2020-08-01")
