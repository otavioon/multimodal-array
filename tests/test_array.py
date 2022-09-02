import pytest

import numpy as np
from multimodal.array import MultiModalArray, apply_multimodal_array_func

# Utilitary functions
def generate_array(shape):
    return np.arange(np.prod(shape)).reshape(shape)

def simple_transform(X: np.ndarray):
    return X+1

def simple_transform_2(X: np.ndarray):
    return np.expand_dims(np.ones(X.shape[0]), axis=1)

def test_len():
    x = generate_array((4, 4))
    array = MultiModalArray(x)
    assert len(array) == len(x), "Length of array mismatch"

def test_window_slice_get():
    windows = [
        (0, 2), (2, 4)
    ]

    x = generate_array((4, 4))
    array = MultiModalArray(x, windows=windows, names=["a", "b"])

    window = array.window_loc[0]
    assert np.array_equal(window.data, x[:, 0:2]), "Window 0 data mismatch"
    assert window.window_names[0] ==  "a", "Name for window 0 mismatch"
    assert window.num_windows ==  1, "Number of windows mismatch"
    assert window.window_slices[0][1].start ==  0 and window.window_slices[0][1].stop ==  2, "Window slice mismatch"

    window = array.window_loc[1]
    assert np.array_equal(window.data, x[:, 2:4]), "Window 1 data mismatch"
    assert window.window_names[0] ==  "b", "Name for window 1 mismatch"
    assert window.num_windows ==  1, "Number of windows mismatch"
    assert window.window_slices[0][1].start ==  0 and window.window_slices[0][1].stop ==  2, "Window slice mismatch"


def test_assignment():
    windows = [
        (0, 2), (2, 4)
    ]

    x, y = generate_array((4, 4)), generate_array((4, 4))
    array = MultiModalArray(x, windows=windows, names=["a", "b"])

    array.window_loc[0] = 1
    y[:, 0:2] = 1

    assert np.array_equal(array.data, y)


def test_indexing():
    windows = [
        (0, 2), (2, 4)
    ]

    x = generate_array((4, 4))
    array = MultiModalArray(x, windows=windows, names=["a", "b"])

    element = array[0]
    assert isinstance(element, MultiModalArray), "Should return a MultiModalArray"
    assert element.data.ndim == 1
    assert element.num_windows == 2

    assert np.isscalar(element[0]), "Should be an scalar"

    element = element.window_loc[1]
    assert element.data.ndim == 1
    assert element.num_windows == 1
    assert element.window_names[0] == "b"
    assert element.window_slices[0][1].start == 0 and element.window_slices[0][1].stop == 2

    assert np.array_equal(array[:].data, x)


def test_1d_array():
    windows = [
        (0, 2), (2, 4)
    ]

    x = generate_array((4,))
    array = MultiModalArray(x, windows=windows, names=["a", "b"])

    element = array.window_loc[1]
    assert element.data.ndim == 1
    assert element.num_windows == 1
    assert element.window_names[0] == "b"
    assert element.window_slices[0][1].start == 0 and element.window_slices[0][1].stop == 2
    assert np.array_equal(element.data, np.array([2, 3]))


def test_exception_3d_array():
    with pytest.raises(ValueError):
        windows = [
            (0, 2), (2, 4)
        ]
        x = generate_array((4,4,4))
        array = MultiModalArray(x, windows=windows, names=["a", "b"])


def test_apply_func():
    windows = [
        (0, 2), (2, 4)
    ]
    x = generate_array((4, 4))
    array = MultiModalArray(x, windows=windows)
    res = apply_multimodal_array_func(simple_transform, array)
    assert np.array_equal(x + 1, res.data), f"Apply func fails. Arrays are not the same"

    x = generate_array((4, 4))
    result_array = np.ones((4, 2))
    array = MultiModalArray(x, windows=windows)
    res = apply_multimodal_array_func(simple_transform_2, array)
    assert np.array_equal(result_array, res.data), f"Apply func fails. Arrays are not the same (2)"

if __name__ == '__main__':
    unittest.main()
