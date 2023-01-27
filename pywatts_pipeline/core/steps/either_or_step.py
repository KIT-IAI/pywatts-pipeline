import pandas as pd
from pywatts_pipeline.core.steps.base_step import BaseStep


class EitherOrStep(BaseStep):
    """
    This step merges the result of multiple input steps, by choosing the first step in the input list which
    contains data for the current data.

    :param input_step: The input_steps for the either_or_step
    :type input_step: List[BaseStep]
    """

    def __init__(self, input_steps):
        super().__init__(input_steps)
        self.name = "EitherOr"

    def get_result(self, start, return_all=False, minimum_data=(0, pd.Timedelta(0))):
        if self._should_stop(start, minimum_data):
            return None
        input_data = self._get_inputs(self.input_steps, start, minimum_data)
        for key, res in input_data.items():
            self.update_buffer(res, key)
        return self._pack_data(start, return_all=return_all, minimum_data=minimum_data)

    def _get_inputs(self, input_steps, start, minimum_data=(0, pd.Timedelta(0))):
        results = []
        for step in input_steps.values():
            inp = step.get_result(start, minimum_data=minimum_data)
            if inp is not None:
                results.append(inp)
        if results:
            return {self.name: results[0]}
        return None

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load the Either or step from a stored step.

        :param stored_step: Information about the stored either or step
        :param inputs: the input steps
        :param targets: Does not exist for eitherOr
        :param module: Does not exist for either or step
        :param file_manager: The filemanager used for saving informations.
        :return: The restored eitherOrStep
        """
        step = cls(inputs)
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def _should_stop(self, start, minimum_data):
        return self._get_inputs(self.input_steps, start) is None
