from typing import Dict

import pandas as pd

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.filemanager import FileManager


class StartStep(BaseStep):
    """
    Start Step of the pipeline.
    """

    def __init__(self, index: str):
        super().__init__()
        self.name = index
        self.index = index

    def get_result(
        self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))
    ):
        return self._pack_data(start, minimum_data=minimum_data, return_all=return_all)

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
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns all information that are needed for restoring the start step
        """
        json = super().get_json(fm)
        json["index"] = self.index
        return json
