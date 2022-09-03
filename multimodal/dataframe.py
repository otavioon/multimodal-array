from typing import List, Tuple, Union, Iterable
import numpy as np
import pandas as pd
from functools import partial
from pandas.api.types import is_scalar

pandas_column_stack = partial(pd.concat, axis=1)

def remove_invalid_windows(columns: List[str], windows: List[slice], names: List[str], axis: int = -1):
    new_windows, new_names = [], []
    for i in range(len(windows)):
        window, name = windows[i], names[i]
        if all(w in columns for w in window):
            new_windows.append(window)
            new_names.append(name)
    if len(new_windows) == 0:
        return None, None
    return new_windows, new_names

def convert_series_to_frame(df):
    if isinstance(df, pd.core.series.Series):
        return df.to_frame().T
    else:
        return df

class MultiModalDataframe:
    class ILoc:
        def __init__(self, obj_ref: "MultiModalDataframe"):
            self.obj_ref = obj_ref

        def __getitem__(self, index):
            df = convert_series_to_frame(self.obj_ref.data.iloc[index])
            if is_scalar(df):
                return df
            windows, names = remove_invalid_windows(list(df.columns), self.obj_ref.window_slices, self.obj_ref.window_names)
            return MultiModalDataframe(df, windows=windows, names=names)

    class Loc:
        def __init__(self, obj_ref: "MultiModalDataframe"):
            self.obj_ref = obj_ref

        def __getitem__(self, index):
            df = convert_series_to_frame(self.obj_ref.data.loc[index])
            if is_scalar(df):
                return df
            windows, names = remove_invalid_windows(list(df.columns), self.obj_ref.window_slices, self.obj_ref.window_names)
            return MultiModalDataframe(df, windows=windows, names=names)

    class WindowSlice:
        def __init__(self, obj_ref: "MultiModalDataframe"):
            self.obj_ref = obj_ref

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

            df = convert_series_to_frame(self.obj_ref.data[self.__getslice(index)])
            if is_scalar(df):
                return df

            return MultiModalDataframe(
                df
                windows=None,
                names=[name]
            )

        def __setitem__(self, index, value):
            self.obj_ref.data[self.__getslice(index)] = value

    def __init__(self, df: pd.DataFrame, windows: List[List[str]] = None, names: List[str] = None):
        self._df = df
        if windows is None:
            windows = [df.columns.to_list()]
        self._windows = windows
        self._names = names or [f"Unnamed {i}" for i in range(len(windows))]
        if len(self._names) != len(self._windows):
            raise ValueError("Names and windows shoud have the same length")

    @property
    def num_windows(self) -> int:
        return len(self._windows)

    @property
    def window_slices(self) -> List[List[str]]:
        return self._windows

    @property
    def window_names(self) -> List[str]:
        return self._names

    @property
    def iloc(self):
        return MultiModalDataframe.ILoc(self)

    @property
    def loc(self):
        return MultiModalDataframe.Loc(self)

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    @property
    def window_loc(self):
        return MultiModalDataframe.WindowSlice(self)

    def __getitem__(self, key) -> pd.DataFrame:
        df = convert_series_to_frame(self._df[key])
        if is_scalar(df):
            return df
        windows, names = remove_invalid_windows(list(df.columns), self._windows, self._names)
        return MultiModalDataframe(df, windows=windows, names=names)

    def __setitem__(self, key, value):
        return self._df.__setitem__(key, value)

    def __pprint_window_slices(self) -> str:
        slice_strs = []
        for i in range(self.num_windows):
            s = f"{self._names[i]}: ({self.window_slices[i]})"
            slice_strs.append(s)
        return ", ".join(slice_strs)

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{str(self._df)}\nWindows: [{self.__pprint_window_slices()}]"


def apply_multimodal_dataframe_func(func: callable, mdf: MultiModalDataframe, collate_fn: callable = pandas_column_stack):
    dfs = []
    windows = []
    names = []
    start = 0
    for i in range(mdf.num_windows):
        data = mdf.window_loc[i].data
        res = func(data)
        columns = [data.columns[i] for i in range(res.shape[-1])]
        res = pd.DataFrame(res, columns=columns)
        dfs.append(res)
        names.append(mdf.window_names[i])
        end = start+res.shape[-1]
        windows.append(columns)
        start = end
    return MultiModalDataframe(collate_fn(dfs), windows=windows, names=names)
