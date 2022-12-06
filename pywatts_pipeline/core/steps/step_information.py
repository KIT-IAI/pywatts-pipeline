from typing import Union

from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.steps.selection_step import SelectionStep


class StepInformation:
    """
    This steps contains information necesary for creating a pipeline and steps by the step factory

    :param step: The step
    :param pipeline: The pipeline
    """

    def __init__(self, step: BaseStep, pipeline):
        self.step = step
        self.pipeline = pipeline

    def __getitem__(self, item: Union[str, int, slice]):
        from pywatts_pipeline.core.steps.step import Step

        if isinstance(item, str):
            if isinstance(self.step, Step):
                result_step = self.step.get_result_step(item)
                self.pipeline.add(module=result_step, input_ids=[self.step.id])
                return StepInformation(result_step, self.pipeline)
        elif isinstance(item, int) or isinstance(item, slice) or isinstance(item, list):
            selection_step = SelectionStep({"step": self.step}, item)
            self.pipeline.add(module=selection_step, input_ids=[self.step.id])
            return StepInformation(selection_step, self.pipeline)

        return self


class SummaryInformation:
    """
    The summary information contains all information about a summary step.
    """
    def __init__(self, step, pipeline):
        self.step = step
        self.pipeline = pipeline
