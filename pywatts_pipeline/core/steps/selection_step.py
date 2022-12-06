from typing import Union, List, Dict

import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes


class SelectionStep(BaseStep):
    """
    TODO Update docs
    This steps fetch the correct column if the previous step provides data with multiple columns as output
    """

    def __init__(self, input_steps, selection: Union[int, List[int], slice]):
        super().__init__(input_steps=input_steps)
        if isinstance(selection, int):
            self.selection_list = [selection]
        elif isinstance(selection, slice):
            stop = selection.stop
            # Swap start and stop if only stop is provided and if it is negative.
            if stop < 0 and selection.start is None:
                start = stop
                stop = 0
            else:
                start = selection.start if selection.start is not None else 0
            step = selection.step if selection.step is not None else 1
            assert start < stop, "Slicing for selecting the input of a step is only supported if the start is smaller" \
                                 " than the stop"
            self.selection_list = list(range(start, stop, step))
        elif isinstance(selection, list):
            assert np.all(list(map(lambda x: isinstance(x, int), selection))), "All elements for specifying" \
                                                                                    " inputs with a list should be ints"
            self.selection_list = selection
        else:
            raise Exception("Parameters for selecting the indexes is neither a int, list of ints or a slice")
    def get_result(
            self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))
    ):
        """
        Returns the specified result of the previous step.
        """
        result = list(self.input_steps.values())[0].get_result(
            start, True, minimum_data=minimum_data
        )
        return self._select(result, self.selection_list)

    @staticmethod
    def _select(dict_arr, indexes):
        assert len(dict_arr) == 1, "Slicing is only possible if the previous step only provides one output"
        arr = list(dict_arr.values())[0]
        r = [arr.shift({index: -i for index in _get_time_indexes(dict_arr)}) for i in indexes]
        return xr.DataArray(np.stack(r, axis=-1), dims=(*arr.dims, "horizon"), coords=arr.coords).dropna(
            _get_time_indexes(dict_arr)[0])

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns all information for restoring the resultStep.
        """
        json_dict = super().get_json(fm)
        json_dict["selection_list"] = self.selection_list
        return json_dict

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load a stored ResultStep.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        step = cls(inputs, [])
        step = stored_step["selection"]
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step
