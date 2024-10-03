import numpy as np
from  src.app import normalize_data

def test_normalize_data():
    data = np.array([[10, 200], [5, 100], [8, 150]])
    result = normalize_data(data)
    expected = np.array([[1.0, 1.0], [0.0, 0.0], [0.6, 0.5]])
    assert np.allclose(result, expected, atol=0.01)
