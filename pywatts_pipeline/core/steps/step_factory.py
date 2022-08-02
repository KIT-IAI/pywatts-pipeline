import inspect
import warnings
from copy import deepcopy
from typing import Tuple, Dict, Union, List, Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from pywatts_pipeline.core.steps.result_step import ResultStep
from pywatts_pipeline.core.transformer.base import Base
from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.steps.either_or_step import EitherOrStep
from pywatts_pipeline.core.exceptions.step_creation_exception import StepCreationException
from pywatts_pipeline.core.steps.inverse_step import InverseStep
from pywatts_pipeline.core.pipeline import Pipeline, logger
from pywatts_pipeline.core.steps.pipeline_step import PipelineStep
from pywatts_pipeline.core.steps.probabilistic_step import ProbablisticStep
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.steps.step_information import StepInformation, SummaryInformation
from pywatts.callbacks import BaseCallback
from pywatts_pipeline.core.steps.summary_step import SummaryStep
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray


class ForecastingStep(Step):

    module : BaseForecaster

    def __init__(self, module: BaseForecaster, input_steps: Dict[str, BaseStep], file_manager, *,
                 targets: Optional[Dict[str, "BaseStep"]] = None,
                 computation_mode=ComputationMode.Default,
                 callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]] = [],
                 condition=None,
                 batch_size: Optional[None] = None,
                 refit_conditions=[],
                 retrain_batch=pd.Timedelta(hours=24),
                 lag=pd.Timedelta(hours=24),
                 fh=ForecastingHorizon(1),
                 strategy="direct"):
        super().__init__(module, input_steps,  file_manager, targets=targets, computation_mode=computation_mode, callbacks=callbacks,
                         condition=condition, batch_size=batch_size, refit_conditions=refit_conditions, retrain_batch=retrain_batch,
                         lag=lag)
        self.fh = fh
        self.strategy = strategy

    def _transform(self, input_step):
        input_data = self.temporal_align_inputs(input_step)
        for val in input_data.values():
            if val is None or len(val) == 0:
                return
        # TODO: Cast input type
        # TODO more flexible with more exogenous variables from multiple sources
        input_data = list(map(lambda it: it[1].values.reshape((len(it[1]), -1)), input_data.items()))
        result = []
        # TODO fix for rest and add option to determine if only newest data should be calculated or all
        # TODO is the a problem only for the training run?
        # TODO Strategy selection should be possible
        #      * Direct: Standard approach
        #           * Obvious approach if exogenous variables are provided.
        #           * How much values if fh has to be set? Derrive from input data (has to exist due to pywatts structure)
        #      * Explicit:
        #           * Mimic sample based approach from pyWATTS. Very expensive. Not sure if there is a use-case for it
        #      * Update
        #           * Update Strategy: Related to online learning. Could be useful..
        if self.strategy == "explicit":
            for val in input_data[0].index:
                val._set_freq("1h")
                result.append(
                    self.module.predict(ForecastingHorizon(self.fh).to_absolute(val)).values, # TODO Exogenous
                )
            result = xr.DataArray(result, dims=["time", "dim0", "dim1"], coords={"time":  input_data[0].index[:len(result)]})
        elif self.strategy == "direct":
            # Use absolute instead of relative ForecastHorizon. I would feel better, if we do this
            # TODO fix 1h
            result = numpy_to_xarray(self.module.predict(fh=ForecastingHorizon(np.arange(1, len(input_data[0]) + 1), freq="1h"), X=input_data[0]), list(self.temporal_align_inputs(input_step).values())[0])
        elif self.strategy == "update":
            # Use absolute instead of relative ForecastHorizon. I would feel better, if we do this
            # TODO fix 1h
            input_data[0] = input_data[0].asfreq("1h")
            result = xr.DataArray(self.module.update_predict(*input_data))
        return self._post_transform(result)


    def _fit(self, inputs: Dict[str, BaseStep], target_step):
        # TODO: Cast input type
        # TODO Exogenous more flexible
        inputs = list(map(lambda it: it[1].values.reshape((len(it[1]), -1)), inputs.items()))
        # TODO fix deepcopy
        target_step = list(map(lambda x: x.values, deepcopy(target_step).values()))
        self.module.fit(target_step[0], X=inputs[0])

class StepFactory:
    """
    A factory for creating the appropriate step for the current sitation.
    """

    def create_step(self,
                    module: Base,
                    kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
                    use_inverse_transform: bool, use_predict_proba: bool,
                    callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]],
                    condition,
                    batch_size,
                    computation_mode,
                    refit_conditions,
                  #  retrain_batch,
                    lag,
                    fh=None,
                    strategy="direct"):
        """
        Creates a appropriate step for the current situation.

        :param module: The module which should be added to the pipeline
        :param kwargs: The input steps for the current step
        :param targets: The target steps for the currrent step
        :param use_inverse_transform: Should inverse_transform be called instead of transform
        :param use_predict_proba: Should probabilistic_transform be called instead of transform
        :param callbacks: Callbacks to use after results are processed.
        :param condition: A function returning True or False which indicates if the step should be performed
        :param batch_size: The size of the past time range which should be used for relearning the module
        :param computation_mode: The computation mode of the step
        :param refit_conditions: A List of methods or Base Conditions for determining if the step should be fitted at a
                                 specific timestamp.
        :param retrain_batch: Determines how much past data should be used for relearning.
        :param lag: Needed for online learning. Determines what data can be used for retraining.
                    E.g., when 24 hour forecasts are performed, a lag of 24 hours is needed, else the retraining would
                    use future values as target values.
        :return: StepInformation
        """

        if isinstance(module, BaseForecaster):
            pipeline = self._check_ins(kwargs)
            input_steps, target_steps = self._split_input_target_steps(kwargs, pipeline)
            step = ForecastingStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                        callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                        batch_size=batch_size, refit_conditions=refit_conditions,  # retrain_batch=retrain_batch,
                        lag=lag, fh=fh, strategy=strategy)
        elif isinstance(module, Base):
            arguments = inspect.signature(module.transform).parameters.keys()
            if "kwargs" not in arguments and not isinstance(module, Pipeline):
                for argument in arguments:
                    if argument not in kwargs.keys():
                        raise StepCreationException(
                            f"The module {module.name} miss {argument} as input. The module needs {arguments} as input. "
                            f"{kwargs} are given as input."
                            f"Add {argument}=<desired_input> when adding {module.name} to the pipeline.",
                            module
                        )

            # TODO needs to check that inputs are unambigious -> I.e. check that each input has only one output
            pipeline = self._check_ins(kwargs)
            input_steps, target_steps = self._split_input_target_steps(kwargs, pipeline)

            if isinstance(module, Pipeline):
                step = PipelineStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                                    callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                                    batch_size=batch_size, refit_conditions=refit_conditions, #retrain_batch=retrain_batch,
                                    lag=lag)
            elif use_inverse_transform:
                step = InverseStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                                   callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                                   #retrain_batch=retrain_batch,
                                   lag=lag)
            elif use_predict_proba:
                step = ProbablisticStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                                        callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                                        #retrain_batch=retrain_batch,
                                        lag=lag)
            else:
                step = Step(module, input_steps, pipeline.file_manager, targets=target_steps,
                            callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                            batch_size=batch_size, refit_conditions=refit_conditions, #retrain_batch=retrain_batch,

                            lag=lag)

        step_id = pipeline.add(module=step,
                               input_ids=[step.id for step in input_steps.values()],
                               target_ids=[step.id for step in target_steps.values()])
       # step.id = step_id

        if len(target_steps) > 1:
            step.last = False
            for target in target_steps:
                r_step = step.get_result_step(target)
                r_id = pipeline.add(module=step, input_ids=[step_id])
                #r_step.id = r_id

        return StepInformation(step, pipeline)

    def _split_input_target_steps(self, kwargs, pipeline, set_last=True):
        input_steps: Dict[str, BaseStep] = dict()
        target_steps: Dict[str, BaseStep] = dict()
        for key, element in kwargs.items():
            if isinstance(element, StepInformation):
                if set_last:
                    element.step.last = False
                if key.startswith("target"):
                    target_steps[key] = element.step
                else:
                    input_steps[key] = element.step
                if isinstance(element.step, PipelineStep):
                    raise StepCreationException(
                        f"Please specify which result of {element.step.name} should be used, since this steps"
                        f"may provide multiple results.")
            elif isinstance(element, tuple):
                if key.startswith("target"):
                    target_steps[key] = self._createEitherOrStep(element, pipeline).step
                else:
                    input_steps[key] = self._createEitherOrStep(element, pipeline).step
        return input_steps, target_steps

    def _createEitherOrStep(self, inputs: Tuple[StepInformation], pipeline):
        for input_step in inputs:
            input_step.step.last = False
        step = EitherOrStep({x.step.name + f"{i}": x.step for i, x in enumerate(inputs)})
        pipeline.add(module=step, input_ids=list(map(lambda x: x.step.id, inputs)))
        return StepInformation(step, pipeline)

    def _check_ins(self, kwargs):
        pipeline = None
        for input_step in kwargs.values():
            if isinstance(input_step, StepInformation):
                pipeline_temp = input_step.pipeline
                if len(input_step.step.targets) > 1:  # TODO define multi-output steps?
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
                            f"The step {input_step.step.name} has multiple outputs. Adding such a step to the pipeline is "
                            "ambigious. "
                            "Specifiy the desired column of your dataset by using step[<column_name>]",
                        )
                    if not pipeline_temp == step_information.pipeline:
                        raise StepCreationException(
                            f"A step can only be part of one pipeline. Assure that all inputs {kwargs}"
                            f"are part of the same pipeline.")

            if pipeline_temp is None:
                raise StepCreationException(f"No Pipeline is specified.")

            if pipeline is None:
                pipeline = pipeline_temp

            if not pipeline_temp == pipeline:
                raise StepCreationException(f"A step can only be part of one pipeline. Assure that all inputs {kwargs}"
                                            f"are part of the same pipeline.")
        return pipeline

    def create_summary(self,
                       module: BaseSummary,
                       kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
                       ) -> SummaryInformation:
        arguments = inspect.signature(module.transform).parameters.keys()

        if "kwargs" not in arguments and not isinstance(module, Pipeline):
            for argument in arguments:
                if argument not in kwargs.keys():
                    raise StepCreationException(
                        f"The module {module.name} miss {argument} as input. The module needs {arguments} as input. "
                        f"{kwargs} are given as input."
                        f"Add {argument}=<desired_input> when adding {module.name} to the pipeline.",
                        module
                    )

        # TODO needs to check that inputs are unambigious -> I.e. check that each input has only one output
        pipeline = self._check_ins(kwargs)
        input_steps, target_steps = self._split_input_target_steps(kwargs, pipeline, set_last=False)

        step = SummaryStep(module, input_steps, pipeline.file_manager, )

        step_id = pipeline.add(module=step,
                               input_ids=[step.id for step in input_steps.values()],
                               target_ids=[step.id for step in target_steps.values()])

        return SummaryInformation(step, pipeline)
