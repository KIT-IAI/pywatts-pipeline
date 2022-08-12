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

    def _get_inputs(self, input_steps, start, minimum_data=(0, pd.Timedelta(0))):
        for step in input_steps.values():
            inp = step.get_result(start, minimum_data=minimum_data)
            if inp is not None:
                return {self.name: inp}
        return None

    def _post_transform(self, result):
        if not isinstance(result, dict):
            result = {self.name: result}
        for key, res in result.items():
            self.update_buffer(res, key)
        return result

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
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def _should_stop(self, start, minimum_data):
        return self._get_inputs(self.input_steps, start) is None


    def get_result(self, start: pd.Timestamp,
                   return_all=False, minimum_data=(0, pd.Timedelta(0))):
            """
            This method is responsible for providing the result of this step.
            Therefore,
            this method triggers the get_input and get_target data methods.
            Additionally, it triggers the computations and checks if all data are processed.

            :param start: The start date of the requested results of the step
            :type start: pd.Timedstamp
            :param end: The end date of the requested results of the step (exclusive)
            :type end: Optional[pd.Timestamp]
            :return: The resulting data or None if no data are calculated
            """
            # Check if step should be executed.
            if self._should_stop(start, minimum_data):
                return None

            self._compute(start, minimum_data)
            return self._pack_data(start, return_all=return_all, minimum_data=minimum_data)