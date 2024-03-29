import logging
from typing import Dict

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.summary.summary_object import SummaryObject
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.steps.step import Step

logger = logging.getLogger(__name__)


class SummaryStep(Step):
    """
    This step encapsulates modules and manages all information for executing a pipeline step.
    Including fetching the input from the input and target step.

    :param module: The module which is wrapped by the step-
    :type module: Base
    :param input_step: The input_step of the module.
    :type input_step: Step
    :param file_manager: The file_manager which is used for storing data.
    :type file_manager: FileManager
    """

    def __init__(self,
                 module: BaseSummary,
                 input_steps: Dict[str, BaseStep],
                 file_manager):
        super().__init__(module, input_steps, file_manager)
        self.name: str = module.name
        self.file_manager: FileManager = file_manager
        self.module: BaseSummary = module

    def _transform(self, input_data):
        return self.module.transform(file_manager=self.file_manager, **input_data)

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load a stored step.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        step = cls(module, inputs, file_manager)
        step.inputs_steps = inputs
        step.name = stored_step["name"]
        step.file_manager = file_manager
        return step

    def get_summaries(self, start) -> [SummaryObject]:
        """
        Calculates a summary for the input data.
        :return: The summary as markdown formatted string
        :rtype: Str
        """
        input_data = self._get_inputs(self.input_steps, start)
        input_data, _  = self.temporal_align_inputs(input_data)
        return [self._transform(input_data)]
