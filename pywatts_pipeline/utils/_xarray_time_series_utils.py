from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray as xr


def get_start(x: Union[xr.DataArray, Dict[str, xr.DataArray]]):
    dim = _get_time_indexes(x)[0]
    if isinstance(x, dict):
        min_date = None
        for val in x.values():
            if min_date is None:
                min_date = val[dim].values[0]
            elif min_date > val[dim].values[0]:
                min_date = val[dim].values[0]
        return min_date
    else:
        return x[dim].values[-1]
def get_last(x: Union[xr.DataArray, Dict[str, xr.DataArray]]):
    dim = _get_time_indexes(x)[0]
    if isinstance(x, dict):
        min_date = None
        for val in x.values():
            if min_date is None:
                min_date = val[dim].values[-1]
            elif min_date > val[dim].values[-1]:
                min_date = val[dim].values[-1]
        return min_date
    else:
        return x[dim].values[-1]

def _get_time_indexes(x: Union[xr.DataArray, Dict[str, xr.DataArray]], get_all=True) -> Union[List[str], str]:
    indexes = []
    if isinstance(x, xr.DataArray):
        for k, v in x.indexes.items():
            if isinstance(v, pd.DatetimeIndex):
                indexes.append(k)
        if get_all:
            return indexes
        return indexes[0]
    for k, v in list(x.values())[0].indexes.items():
        if isinstance(v, pd.DatetimeIndex):
            indexes.append(k)
    if get_all:
        return indexes
    return indexes[0]


def xarray_to_numpy(x: Dict[str, xr.DataArray]):
    if x is None:
        return None
    result = None
    for da in x.values():
        if result is not None:
            result = np.concatenate([result, da.values.reshape((len(da.values), -1))], axis=1)
        else:
            result = da.values.reshape((len(da.values), -1))
    return result


def numpy_to_xarray(x: np.ndarray, reference: xr.DataArray) -> xr.DataArray:
    time_index = reference.indexes[_get_time_indexes(reference)[0]]
    coords = {
        # first dimension is number of batches. We assume that this is the time.
        time_index.name: time_index.values,
        **{f"dim_{j}": list(range(size)) for j, size in enumerate(x.shape[1:])}
    }

    return xr.DataArray(x, coords=coords, dims=list(coords.keys()))
