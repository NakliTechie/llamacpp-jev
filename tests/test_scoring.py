import math

import pytest

from llamajev.scoring import confidence, normalize


def test_normalize_is_softmax():
    probs = normalize([0.0, -1.0])
    assert probs[0] == pytest.approx(1 / (1 + math.exp(-1)))
    assert sum(probs) == pytest.approx(1)


def test_normalize_ignores_shift():
    assert normalize([-100.0, -101.0]) == pytest.approx(normalize([0.0, -1.0]))


def test_truncated_labels_get_zero():
    probs = normalize([0.0, -math.inf])
    assert probs == [1.0, 0.0]


def test_all_truncated_is_an_error():
    with pytest.raises(ValueError):
        normalize([-math.inf, -math.inf])


def test_confidence_bounds():
    assert confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert 0 < confidence([0.7, 0.2, 0.1]) < 1
