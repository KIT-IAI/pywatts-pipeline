import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
import copy

import pandas as pd
import xarray as xr

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.util.run_setting import RunSetting
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes, get_last
from pywatts_pipeline.core.summary.summary_object import (
    SummaryObjectList,
    SummaryCategory,
)

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """
    The base class of all steps.
    :param input_steps: The input steps
    :type input_steps: Optional[Dict[str, BaseStep]]
    :param targets: The target steps
    :type targets: Optional[Dict[str, BaseStep]]
    :param condition: A function which evaluates to False or True for detecting if the module should be executed.
    :type condition: Callable
    :param computation_mode: The computation mode for this module
    :type computation_mode: ComputationMode
    """

    def __init__(
        self,
        input_steps: Optional[Dict[str, "BaseStep"]] = None,
        targets: Optional[Dict[str, "BaseStep"]] = None,
        condition=None,
        computation_mode=ComputationMode.Default,
        name="BaseStep",
    ):
        self.default_run_setting = RunSetting(computation_mode=computation_mode)
        self.current_run_setting = self.default_run_setting.clone()
        self.input_steps: Dict[str, "BaseStep"] = ({} if input_steps is None else input_steps)
        self.targets: Dict[str, "BaseStep"] = {} if targets is None else targets
        self.name = name
        self.id = -1
        self.last = True
        self.buffer: Dict[str, xr.DataArray] = {}

        self.condition = condition
        self.transform_time = SummaryObjectList(
            self.name + " Transform Time", category=SummaryCategory.TransformTime
        )
        self.training_time = SummaryObjectList(
            self.name + " Training Time", category=SummaryCategory.FitTime
        )
        self.finished = False

    @abstractmethod
    def get_result(
        self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))
    ):
        """
        This method is responsible for providing the result of this step.
        Therefore,
        this method triggers the get_input and get_target data methods.
        Additionally, it triggers the computations and checks if all data are processed.

        :param start: The start date of the requested results of the step
        :type start: pd.Timedstamp
        :param end: The end date of the requested results of the step (exclusive)
        :type end: Optional[pd.Timestamp]
        :param return_all: Flag that indicates if all results in the buffer should be returned.
        :type return_all: bool
        :return: The resulting data or None if no data are calculated
        """
        # Check if step should be executed.

    def _pack_data(self, start, return_all=False, minimum_data=(0, pd.Timedelta(0))):
        # Provide requested data
        if len(self.buffer) == 0:
            return None
        time_index = _get_time_indexes(self.buffer, get_all=False)

        if start:
            index = list(self.buffer.values())[0].indexes[time_index]
            if len(index) > 1:
                freq = index[1] - index[0]
            else:
                freq = 0
            start = start - pd.Timedelta(minimum_data[0] * freq) - minimum_data[1]
            # If end is not set, all values should be considered. Thus we add a small timedelta to the last index entry.
            # After sel copy is not needed, since it returns a new array.
            if return_all:
                return {
                    key: b.sel(**{time_index: b.indexes[_get_time_indexes(b)[0]][(b.indexes[_get_time_indexes(b)[0]] >= start)]})
                    for key, b in self.buffer.items()
                }
            return list(self.buffer.values())[0].sel(
                    **{time_index: index[(index >= start)]}
                )
        self.finished = True
        if return_all:
            return copy.deepcopy(self.buffer)
        return list(self.buffer.values())[0].copy()

    def update_buffer(self, x: xr.DataArray, index):
        if len(x) == 0:
            pass
        elif index not in self.buffer:
            self.buffer[index] = x
        else:
            dim = _get_time_indexes(self.buffer[index], get_all=False)
            last = get_last(self.buffer[index])
            if index != dim:
                self.buffer[index] = xr.concat(
                    [self.buffer[index], x[x[dim] > last].dropna(dim)], dim=dim
                )
            else:
                self.buffer[index] = xr.concat(
                    [self.buffer[index], x[x[dim] > last]], dim=dim
                )


    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns a dictionary containing all information needed for restoring the step.

        :param fm: The filemanager which can be used by the step for storing the state of the step.
        :type fm: FileManager
        :return: A dictionary containing all information needed for restoring the step.
        :rtype: Dict
        """
        return {
            "target_ids": {step.id: key for key, step in self.targets.items()},
            "input_ids": {step.id: key for key, step in self.input_steps.items()},
            "id": self.id,
            "module": self.__module__,
            "class": self.__class__.__name__,
            "name": self.name,
            "last": self.last,
            "default_run_setting": self.default_run_setting.save(),
        }

    @classmethod
    @abstractmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Restores the step.

        :param stored_step: Information about the stored step
        :param inputs: The input steps of the step which should be restored
        :param targets: The target steps of the step which should be restored
        :param module: The module which is contained by this step
        :param file_manager: The filemanager of the step
        :return: The restored step.
        """

    def _get_inputs(self, input_steps, start, minimum_data=(0, pd.Timedelta(0))):
        return {}

    def _should_stop(self, start, minimum_data) -> bool:
        # Fetch input and target data
        input_result = self._get_inputs(
            self.input_steps, start, minimum_data=minimum_data
        )
        target_result = self._get_inputs(self.targets, start, minimum_data=minimum_data)

        # Check if either the condition is True or some of the previous steps stopped (return_value is None)
        return (
            (
                self.condition is not None
                and not self.condition(input_result, target_result)
            )
            or self._input_stopped(input_result)
            or (self.current_run_setting.computation_mode in [ComputationMode.Default, ComputationMode.FitTransform,
                                                              ComputationMode.Refit] and self._input_stopped(target_result))
        )

    @staticmethod
    def _input_stopped(input_data):
        return (
            input_data is not None
            and len(input_data) > 0
            and any(map(lambda x: x is None, input_data.values()))
        )

    def reset(self, keep_buffer=False):
        """
        Resets all information of the step concerning a specific run.
        :param keep_buffer: Flag indicating if the buffer should be resetted too.
        """
        if not keep_buffer:
            self.buffer = {}
        self.finished = False
        self._last_computed_entry = None
        self.current_run_setting = self.default_run_setting.clone()

    def set_run_setting(self, run_setting: RunSetting):
        """
        Sets the computation mode of the step for the current run. Note that after reset the all mode is restored.
        Moreover, setting the computation_mode is only possible if the computation_mode is not set explicitly while
        adding the corresponding module to the pipeline.

        :param computation_mode: The computation mode which should be set.
        :type computation_mode: ComputationMode
        """
        self.current_run_setting = self.default_run_setting.update(run_setting)
