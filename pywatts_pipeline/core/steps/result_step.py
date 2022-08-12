from typing import Optional, Dict

import pandas as pd

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.filemanager import FileManager


class ResultStep(BaseStep):
    """
    This steps fetch the correct column if the previous step provides data with multiple columns as output
    """

    def __init__(self, input_steps, buffer_element: str):
        super().__init__(input_steps=input_steps)
        self.buffer_element = buffer_element
        # TODO should have own buffer


    def get_result(self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))):
        """
        Returns the specified result of the previous step.
        """
        # TODO fill own buffer with results of previous steps

        # TODO use _pack_data
        # TODO can we get rid off get_all?
        if not return_all:
            result = list(self.input_steps.values())[0].get_result(start, self.buffer_element, minimum_data=minimum_data)
        else:
            result = {self.buffer_element: list(self.input_steps.values())[0].get_result(start, self.buffer_element,
                                                                                       minimum_data=minimum_data)}
        if result is None or self.buffer_element not in result.keys():
            return None
        else:
            return result[self.buffer_element]

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns all information for restoring the resultStep.
        """
        json_dict = super().get_json(fm)
        json_dict["buffer_element"] = self.buffer_element
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
        step = cls(inputs, stored_step["buffer_element"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step
