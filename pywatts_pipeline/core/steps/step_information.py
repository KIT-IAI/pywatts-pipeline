from pywatts_pipeline.core.steps.base_step import BaseStep

class StepInformation:
    """
    This steps contains information necesary for creating a pipeline and steps by the step factory

    :param step: The step
    :param pipeline: The pipeline
    """

    def __init__(self, step: BaseStep, pipeline):
        self.step = step
        self.pipeline = pipeline

    def __getitem__(self, item: str):
        from pywatts_pipeline.core.steps.step import Step
        if isinstance(self.step, Step):
            self.step.last = False  # TODO this should be a part of the step_factory
            result_step = self.step.get_result_step(item)
            self.pipeline.add(module=result_step, input_ids=[self.step.id])
            return StepInformation(result_step, self.pipeline)
        else:
            return self


class SummaryInformation:
    def __init__(self, step, pipeline):
        self.step = step
        self.pipeline = pipeline
