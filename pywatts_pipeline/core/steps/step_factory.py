import inspect
import warnings
from typing import Tuple, Dict, Union, List, Callable

import xarray as xr

from pywatts_pipeline.core.callbacks import BaseCallback
from pywatts_pipeline.core.exceptions.step_creation_exception import (
    StepCreationException,
)
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.steps.either_or_step import EitherOrStep
from pywatts_pipeline.core.steps.pipeline_step import PipelineStep
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.steps.step_information import (
    StepInformation,
    SummaryInformation,
)
from pywatts_pipeline.core.steps.summary_step import SummaryStep
from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.transformer.base import Base


class StepFactory:
    """
    A factory for creating the appropriate step for the current sitation.
    """

    def create_step(
        self,
        module: Base,
        kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
        method,
        callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]],
        condition,
        computation_mode,
        refit_conditions,
        #  retrain_batch,
        lag,
    ):
        """
        Creates a appropriate step for the current situation.

        :param module: The module which should be added to the pipeline
        :param kwargs: The input steps for the current step
        :param targets: The target steps for the currrent step
        :param callbacks: Callbacks to use after results are processed.
        :param condition: A function returning True or False which indicates if the step should be performed
        :param computation_mode: The computation mode of the step
        :param refit_conditions: A List of methods or Base Conditions for determining if the step should be fitted at a
                                 specific timestamp.
        :param retrain_batch: Determines how much past data should be used for relearning.
        :param lag: Needed for online learning. Determines what data can be used for retraining.
                    E.g., when 24 hour forecasts are performed, a lag of 24 hours is needed, else the retraining would
                    use future values as target values.
        :return: StepInformation
        """
        if "use_inverse_transform" in kwargs:
            warnings.warn(
                "The usage of use_inverse_transform is deprecated. And will be removed in pywatts_pipeline version 0.2"
            )
            method = "inverse_transform" if kwargs["use_inverse_transform"] else None
            del kwargs["use_inverse_transform"]

        if "use_probabilistic_transform" in kwargs:
            warnings.warn(
                "The usage of use_prob_transform is deprecated. And will be removed in pywatts_pipeline version 0.2"
            )
            method = "predict_proba" if kwargs["use_probabilistic_transform"] else None
            del kwargs["use_probabilistic_transform"]

        pipeline = self._check_ins(kwargs)

        input_steps, target_steps = self._check_and_extract_inputs(
            kwargs, module, pipeline
        )

        if isinstance(module, Pipeline):
            step = PipelineStep(
                module,
                input_steps,
                pipeline.file_manager,
                targets=target_steps,
                callbacks=callbacks,
                computation_mode=computation_mode,
                condition=condition,
                refit_conditions=refit_conditions,  # retrain_batch=retrain_batch,
                lag=lag,
            )
        else:
            step = Step(
                module,
                input_steps,
                pipeline.file_manager,
                targets=target_steps,
                method=method,
                callbacks=callbacks,
                computation_mode=computation_mode,
                condition=condition,
                refit_conditions=refit_conditions,  # retrain_batch=retrain_batch,
                lag=lag,
            )

        pipeline.add(
            module=step,
            input_ids=[step.id for step in input_steps.values()],
            target_ids=[step.id for step in target_steps.values()],
        )

        if len(target_steps) > 1:
            # TODO why it is necessary to define here get_result_steps?
            step.last = False
            for target in target_steps:
                r_step = step.get_result_step(target)
                r_id = pipeline.add(module=step)
                r_step.id = r_id

        return StepInformation(step, pipeline)

    def _check_and_extract_inputs(self, kwargs, module, pipeline, set_last=True):
        transform_arguments = inspect.signature(module.transform).parameters.keys()
        fit_arguments = inspect.signature(module.fit).parameters.keys()
        if "kwargs" not in transform_arguments and not isinstance(module, Pipeline):
            in_keys = set(transform_arguments) & set(kwargs.keys())
            t_keys = (set(fit_arguments) & set(kwargs.keys())) - in_keys
            if len(in_keys) != len(transform_arguments):
                raise StepCreationException(
                    f"The module {module.name} misses the following inputs: {set(transform_arguments) - in_keys}. "
                    f"Only {kwargs} are given as input."
                    f"Add <missing argument>=<desired_input> when adding {module.name} to the pipeline.",
                    module,
                )

            input_steps: Dict[str, BaseStep] = {
                key: self._check_in(kwargs[key], pipeline=pipeline, set_last=set_last)
                for key in in_keys
            }
            target_steps: Dict[str, BaseStep] = {
                key: self._check_in(kwargs[key], pipeline=pipeline, set_last=set_last)
                for key in t_keys
            }
        else:
            input_steps: Dict[str, BaseStep] = {
                key: self._check_in(kwargs[key], pipeline=pipeline, set_last=set_last)
                for key in filter(lambda x: not x.startswith("target"), kwargs.keys())
            }
            target_steps: Dict[str, BaseStep] = {
                key: self._check_in(kwargs[key], pipeline=pipeline, set_last=set_last)
                for key in filter(lambda x: x.startswith("target"), kwargs.keys())
            }
        return input_steps, target_steps

    def _check_in(self, element, pipeline, set_last=True):
        if isinstance(element, StepInformation):
            if set_last:
                element.step.last = False
            if isinstance(element.step, PipelineStep):
                raise StepCreationException(
                    f"Please specify which result of {element.step.name} should be used, since this steps"
                    f"may provide multiple results."
                )
            return element.step
        if isinstance(element, tuple):
            return self._createEitherOrStep(element, pipeline).step
        raise Exception()

    @staticmethod
    def _createEitherOrStep(inputs: Tuple[StepInformation], pipeline):
        for input_step in inputs:
            input_step.step.last = False
        step = EitherOrStep(
            {x.step.name + f"{i}": x.step for i, x in enumerate(inputs)}
        )
        pipeline.add(module=step, input_ids=list(map(lambda x: x.step.id, inputs)))
        return StepInformation(step, pipeline)

    @staticmethod
    def _check_ins(kwargs):
        pipeline = None
        for input_step in kwargs.values():
            if isinstance(input_step, StepInformation):
                pipeline_temp = input_step.pipeline
                if len(input_step.step.targets) > 1:
                    raise StepCreationException(
                        f"The step {input_step.step.name} has multiple outputs. "
                        "Adding such a step to the pipeline is ambigious. "
                        "Specifiy the desired column of your dataset by using step[<column_name>]",
                    )
            elif isinstance(input_step, Pipeline):
                raise StepCreationException(
                    "Adding a pipeline as input might be ambigious. "
                    "Specifiy the desired column of your dataset by using pipeline[<column_name>]",
                )
            elif isinstance(input_step, tuple):
                # We assume that a tuple consists only of step informations and do not contain a pipeline.
                pipeline_temp = input_step[0].pipeline

                if len(input_step[0].step.targets) > 1:
                    raise StepCreationException(
                        f"The step {input_step.step.name} has multiple outputs. Adding such a step to the pipeline is "
                        "ambigious. "
                        "Specifiy the desired column of your dataset by using step[<column_name>]",
                    )

                for step_information in input_step[1:]:

                    if len(step_information.step.targets) > 1:
                        raise StepCreationException(
                            f"The step {input_step.step.name} has multiple outputs. Adding such a step to the pipeline"
                            " is ambigious. "
                            "Specifiy the desired column of your dataset by using step[<column_name>]",
                        )
                    if not pipeline_temp == step_information.pipeline:
                        raise StepCreationException(
                            f"A step can only be part of one pipeline. Assure that all inputs {kwargs}"
                            f"are part of the same pipeline."
                        )

            if pipeline_temp is None:
                raise StepCreationException("No Pipeline is specified.")

            if pipeline is None:
                pipeline = pipeline_temp

            if not pipeline_temp == pipeline:
                raise StepCreationException(
                    f"A step can only be part of one pipeline. Assure that all inputs {kwargs}"
                    f"are part of the same pipeline."
                )
        return pipeline

    def create_summary(
        self,
        module: BaseSummary,
        kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
    ) -> SummaryInformation:
        """
        Uses a Summary, adds it to the pipeline, creates a summary step and returns the summary information
        Args:
            module: The summary that should be added
            kwargs: Input steps

        Returns: A summary information, containing the pipeline and the new SummaryStep

        """
        pipeline = self._check_ins(kwargs)

        input_steps, target_steps = self._check_and_extract_inputs(
            kwargs, module, pipeline, set_last=False
        )

        step = SummaryStep(
            module,
            input_steps,
            pipeline.file_manager,
        )

        pipeline.add(
            module=step,
            input_ids=[step.id for step in input_steps.values()],
            target_ids=[step.id for step in target_steps.values()],
        )

        return SummaryInformation(step, pipeline)
