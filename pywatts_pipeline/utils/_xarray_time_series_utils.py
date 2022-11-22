from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray as xr


def get_start(x: Union[xr.DataArray, Dict[str, xr.DataArray]]):
    """
    Returns the first date of the given time series
    Args:
        x: the given time series as Dict

    Returns: The date of the first entry in the time series.

    """
    dim = _get_time_indexes(x)[0]
    if isinstance(x, dict):
        min_date = None
        for val in x.values():
            if min_date is None:
                min_date = val[dim].values[0]
            elif min_date > val[dim].values[0]:
                min_date = val[dim].values[0]
        return min_date
    return x[dim].values[-1]


def get_last(x: Union[xr.DataArray, Dict[str, xr.DataArray]]):
    """
    Returns the last date of the given time series
    Args:
        x: the given time series as Dict

    Returns: The date of the last entry in the time series.

    """
    dim = _get_time_indexes(x)[0]
    if isinstance(x, dict):
        min_date = None
        for val in x.values():
            if min_date is None:
                min_date = val[dim].values[-1]
            elif min_date > val[dim].values[-1]:
                min_date = val[dim].values[-1]
        return min_date

    return x[dim].values[-1]

def _get_time_indexes(
    x: Union[xr.DataArray, Dict[str, xr.DataArray]], get_all=True, get_index=False
) -> Union[List[str], str]:
    """
    Returns a list of time indexes
    Args:
        x: The Dict of xarrays or the xarray from which the name time indexes should be returned
        get_all: If all or only one time index name should be returned

    Returns: A list of names or one name as a string
    """
    indexes = []
    if isinstance(x, xr.DataArray):
        for k, v in x.indexes.items():
            if isinstance(v, pd.DatetimeIndex):
                indexes.append(k)
        if get_all:
            return indexes
        if get_index:
            return v.indexes[indexes[0]]
        return indexes[0]
    if not x:
        return []
    for k, v in list(x.values())[0].indexes.items():
        if isinstance(v, pd.DatetimeIndex):
            indexes.append(k)
    if get_all:
        return indexes
    if get_index:
        return list(x.values())[0].indexes[indexes[0]]
    return indexes[0]

def xarray_to_numpy(x: Dict[str, xr.DataArray]):
    """
    Converts a dict of xarray into a numpy array
    Args:
        x: Dict of xarrays

    Returns: numpy array
    """
    if x is None:
        return None
    result = None
    for da in x.values():
        if result is not None:
            result = np.concatenate(
                [result, da.values.reshape((len(da.values), -1))], axis=1
            )
        else:
            result = da.values.reshape((len(da.values), -1))
    return result


def numpy_to_xarray(x: np.ndarray, reference: xr.DataArray) -> xr.DataArray:
    """
    Transforms a numpy array into an xr.Dataarray with the same coords ands indexes as the reference DataArray
    Args:
        x: numpy array that should be transformed into a xarray Dataarray
        reference: The reference xr.DataArray

    Returns: xr.DataArray

    """
    time_index = reference.indexes[_get_time_indexes(reference)[0]]
    coords = {
        # first dimension is number of batches. We assume that this is the time.
        time_index.name: time_index.values,
        **{f"dim_{j}": list(range(size)) for j, size in enumerate(x.shape[1:])},
    }

    return xr.DataArray(x, coords=coords, dims=list(coords.keys()))
