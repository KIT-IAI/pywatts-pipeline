import xarray as xr
from typing import Dict

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.filemanager import FileManager

from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes, get_last, get_start


class StartStep(BaseStep):
    """
    Start Step of the pipeline.
    """

    def __init__(self, index: str):
        super().__init__()
        self.name = index
        self.index = index

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        A classmethod which reloads a previously stored step.

        :param stored_step:
        :param inputs:
        :param targets:
        :param module:
        :return:
        """
        step = cls(index=stored_step["index"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def further_elements(self, counter):
        """
        Checks if there exist at least one data for the time after counter.

        :param counter: The timestamp for which it should be tested if there exist further data after it.
        :type counter: pd.Timestamp
        :return: True if there exist further data
        :rtype: bool
        """
        indexes = _get_time_indexes(self.buffer)
        if len(indexes) == 0 or not all(
                [counter < b.indexes[_get_time_indexes(self.buffer)[0]][-1] for b in self.buffer.values()]):
            return False
        else:
            return True

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns all information that are needed for restoring the start step
        """
        json = super().get_json(fm)
        json["index"] = self.index
        return json

    def update_buffer(self, x:xr.DataArray):
        if self.index not in self.buffer:
            self.buffer[self.index] = x
        else:
            dim = _get_time_indexes(self.buffer[self.index], get_all=False)
            last = get_last(self.buffer[self.index])
            self.buffer[self.index] = xr.concat([self.buffer[self.index], x[x[dim] > last]], dim=dim)
        self.finished = True