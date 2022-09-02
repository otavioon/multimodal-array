from typing import List, Tuple, Union, Iterable
import numpy as np

class MultiModalArray:
    class WindowSlice:
        def __init__(self, obj_ref: "MultiModalArray"):
            self.obj_ref = obj_ref

        def __getmask(self, index):
            if isinstance(index, str):
                index = self.obj_ref.window_names.index(index)
            if isinstance(index, int):
                mask = np.ones(self.obj_ref.data.shape, dtype=bool)
                mask[self.obj_ref.window_slices[index]] = False
                return mask
            else:
                raise TypeError("Index must be integer os str")

        def __getslice(self, index):
            if isinstance(index, str):
                index = self.obj_ref.window_names.index(index)
            if isinstance(index, int):
                return self.obj_ref.window_slices[index]
            else:
                raise TypeError("Index must be integer os str")

        def __getitem__(self, index):
            if isinstance(index, str):
                name = index
            elif isinstance(index, int):
                name = self.obj_ref.window_names[index]

            if self.obj_ref.data.ndim > 1:
                mask = self.__getmask(index)
                arr = np.ma.array(self.obj_ref.data, mask=mask, copy=False)
                return MultiModalArray(np.ma.compress_cols(arr), windows=None, names=[name])
            else:
                return MultiModalArray(self.obj_ref.data[self.__getslice(index)], windows=None, names=[name])

        def __setitem__(self, index, value):
            mask = self.__getmask(index)
            self.obj_ref.data[~mask] = value


    def __init__(self, array: np.ndarray, windows: List[Tuple[int, int]] = None, names: List[str] = None):
        if array.ndim > 2:
            raise ValueError("Arrays must be at most 2-D")

        self._array = array
        if windows is None:
            windows = [(0, array.shape[-1])]
        self._windows = [
            (..., slice(start, stop))
            for start, stop in windows
        ]
        self._names = names or [f"Unnamed {i}" for i in range(len(windows))]
        if len(self._names) != len(self._windows):
            raise ValueError("Names and windows shoud have the same length")

    @property
    def num_windows(self) -> int:
        return len(self._windows)

    @property
    def window_slices(self) -> List[slice]:
        return self._windows

    @property
    def window_names(self) -> List[str]:
        return self._names

    @property
    def data(self) -> np.ndarray:
        return self._array

    @property
    def window_loc(self):
        return MultiModalArray.WindowSlice(self)

    @staticmethod
    def __remove_invalid_windows(shape: tuple, windows: List[slice], names: List[str], axis: int = -1):
        new_windows, new_names = [], []
        for i in range(len(windows)):
            window, name = windows[i], names[i]
            if window[-1].start <= shape[axis] and window[-1].stop <= shape[axis]:
                new_windows.append((window[-1].start, window[-1].stop))
                new_names.append(name)

        if len(new_windows) == 0:
            return None, None
        return new_windows, new_names


    def __getitem__(self, key) -> np.ndarray:
        arr = self._array[key]
        if np.isscalar(arr):
            return arr
        windows, names = self.__remove_invalid_windows(arr.shape, self._windows, self._names)
        return MultiModalArray(arr, windows, names=names)


    def __setitem__(self, key, value):
        return self._array.__setitem__(key, value)

    def __pprint_window_slices(self) -> str:
        slice_strs = []
        for i in range(self.num_windows):
            s = f"{self._names[i]}: ({self.window_slices[i][1].start}, {self.window_slices[i][1].stop})"
            slice_strs.append(s)
        return ", ".join(slice_strs)


    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{str(self._array)}\nWindows: [{self.__pprint_window_slices()}]"


def apply_multimodal_array_func(func: callable, arr: MultiModalArray, collate_fn: callable = np.hstack):
    arrs = []
    windows = []
    names = []
    start = 0
    for i in range(arr.num_windows):
        res = func(arr.window_loc[i].data)
        arrs.append(res)
        names.append(arr.window_names[i])
        end = start+res.shape[-1]
        windows.append((start, end))
        start = end
    return MultiModalArray(collate_fn(arrs), windows=windows, names=names)
