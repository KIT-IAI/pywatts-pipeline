from pywatts_pipeline.core.steps.base_step import BaseStep


class StepInformation:
    """
    This steps contains information necesary for creating a pipeline and steps by the step factory

    :param step: The step
    :param pipeline: The pipeline
    """

    def __init__(self, step: BaseStep, pipeline):
        if isinstance(step, BaseStep):
            self.step = step.name
        else:
            self.step = step
        self.pipeline = pipeline

    def __getitem__(self, item: str):
        return StepInformation(self.step + "__" + item, self.pipeline)


class SummaryInformation:
    """
    The summary information contains all information about a summary step.
    """
    def __init__(self, step, pipeline):
        self.step = step
        self.pipeline = pipeline
