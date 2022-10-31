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
        pipeline
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
        # TODO needs to check that inputs are unambigious -> I.e. check that each input has only one output
        self._check_ins(kwargs)

        input_steps = self._check_and_extract_inputs(
            kwargs, module, pipeline, method
        )

        if isinstance(module, Pipeline):
            step = PipelineStep(
                module,
                input_steps,
                pipeline.file_manager,
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
                method=method,
                callbacks=callbacks,
                computation_mode=computation_mode,
                condition=condition,
                refit_conditions=refit_conditions,
                lag=lag,
            )


        return StepInformation(step, pipeline)

    def _check_and_extract_inputs(self, kwargs, module, pipeline, method=None):
        input_steps = {}
        if method is None:
            transform_arguments = \
                dict(filter(lambda x: isinstance(x[1].default, inspect._empty),
                            inspect.signature(getattr(module, list(set(module.__dir__()) & {"transform", "predict"})[
                                0])).parameters.items())
                     ).keys()
        else:
            transform_arguments = \
                dict(filter(lambda x: isinstance(x[1].default, inspect._empty),
                            inspect.signature(getattr(module, method)).parameters.items())
                     ).keys()
        
        if "kwargs" not in inspect.signature(getattr(module, list(set(module.__dir__()) & {"transform", "predict"})[0])).parameters.keys() and not isinstance(module, Pipeline):
            in_keys = set(transform_arguments) & set(kwargs.keys())
            if len(in_keys) != len(transform_arguments):
                raise StepCreationException(
                    f"The module {module.name} misses the following inputs: {set(transform_arguments) - in_keys}. "
                    f"Only {kwargs} are given as input."
                    f"Add <missing argument>=<desired_input> when adding {module.name} to the pipeline.",
                    module,
                )

            if method is None:
                transform_arguments = inspect.signature(
                    getattr(module, list(set(module.__dir__()) & {"transform", "predict"})[0])).parameters.keys()
            else:
                transform_arguments = inspect.signature(
                    getattr(module, method)).parameters.keys()
            fit_arguments = inspect.signature(module.fit).parameters.keys()
            in_keys = (set(transform_arguments) | set(fit_arguments)) & set(kwargs.keys())
            input_steps: Dict[str, BaseStep] = {key: self._check_in(kwargs[key], pipeline=pipeline) for key in kwargs}
    
        else:
            input_steps: Dict[str, BaseStep] = {key: self._check_in(kwargs[key], pipeline=pipeline)
                                            for key in kwargs.keys()}
    
    
        return input_steps
    
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
        #pipeline.add(module=step, input_ids=list(map(lambda x: x.step.id, inputs)))
        return StepInformation(step, pipeline)

    @staticmethod
    def _check_ins(kwargs):
        pipeline = None
        for key, input_step in kwargs.items():
            if isinstance(input_step, StepInformation):
                pipeline_temp = input_step.pipeline
            elif isinstance(input_step, Pipeline):
                raise StepCreationException(
                    "Adding a pipeline as input might be ambigious. "
                    "Specifiy the desired column of your dataset by using pipeline[<column_name>]",
                )
            elif isinstance(input_step, tuple):
                # We assume that a tuple consists only of step informations and do not contain a pipeline.
                pipeline_temp = input_step[0].pipeline


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

            elif isinstance(input_step, list):
                pass # TODO implement implicit FeatureUnion

            else:
                kwargs[key] = StepInformation(DummyStep(input_step), pipeline) # TODO ensure that pipeline exists..
                # TODO is DummyStep really a good idea?
                #  Should we restrict this to ForecastingHorizons?

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
        input_steps = self._check_and_extract_inputs(
            kwargs, module, pipeline
        )
        step = SummaryStep(
            module,
            input_steps,
            pipeline.file_manager,
        )

        return SummaryInformation(step, pipeline)
