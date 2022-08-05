from typing import Dict

import pandas as pd

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.filemanager import FileManager


class DummyStep(BaseStep):
    """
    Start Step of the pipeline.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.last = False

    def get_result(self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))):
        # TODO return as dict?
        return self.value

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
        step = cls(value=stored_step["value"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns all information that are needed for restoring the start step
        """
        json = super().get_json(fm)
        json["value"] = self.value
        return json

