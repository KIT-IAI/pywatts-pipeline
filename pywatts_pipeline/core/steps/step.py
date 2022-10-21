import logging
import time
import warnings
from typing import Optional, Dict, Union, Callable, List

import cloudpickle
import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.callbacks import BaseCallback
from pywatts_pipeline.core.condition.base_condition import BaseCondition
from pywatts_pipeline.core.exceptions.not_fitted_exception import NotFittedException
from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.steps.result_step import ResultStep
from pywatts_pipeline.core.transformer.base import Base, BaseEstimator
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.util.run_setting import RunSetting
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes

logger = logging.getLogger(__name__)


class Step(BaseStep):
    """
    This step encapsulates modules and manages all information for executing a pipeline step.
    Including fetching the input from the input and target step.

    :param module: The module which is wrapped by the step-
    :type module: Base
    :param input_steps: The input_step of the module.
    :type input_steps: Step
    :param file_manager: The file_manager which is used for storing data.
    :type file_manager: FileManager
    :param targets: The step against which's output the module of the current step should be fitted. (Default: None)
    :type targets: Optional[Step]
    :param computation_mode: The computation mode which should be for this step. (Default: ComputationMode.Default)
    :type computation_mode: ComputationMode
    :param callbacks: Callbacks to use after results are processed.
    :type callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]]
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.DataArray, xr.DataArray, bool]
    :param refit_conditions: A List of Callables of BaseConditions, which contains a condition that indicates if
                                 the module should be trained or not
    :type refit_conditions: List[Union[BaseCondition, Callable]]
    :param lag: Needed for online learning. Determines what data can be used for retraining.
            E.g., when 24 hour forecasts are performed, a lag of 24 hours is needed, else the retraining would
            use future values as target values.
    :type lag: pd.Timedelta
    :param retrain_batch: Needed for online learning. Determines how much data should be used for retraining.
    :type retrain_batch: pd.Timedelta
    """

    def __init__(self,
                 module: Base,
                 input_steps: Dict[str, BaseStep],
                 file_manager,
                 *,
                 targets: Optional[Dict[str, "BaseStep"]] = None,
                 method=None,
                 computation_mode=ComputationMode.Default,
                 callbacks: List[
                     Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]
                 ] = None,
                 condition=None,
                 refit_conditions:List[BaseCondition]=None,
                 retrain_batch=pd.Timedelta(hours=24),
                 lag=pd.Timedelta(hours=24)):
        self.method = method
        super().__init__(
            input_steps,
            targets,
            condition=condition,
            computation_mode=computation_mode,
            name=module.name,
        )
        self.file_manager = file_manager
        self.module = module
        self.callbacks = callbacks if callbacks is not None else []
        self.refit_conditions = refit_conditions if refit_conditions is not None else []

        if self.current_run_setting.computation_mode is not ComputationMode.Refit \
                and len(self.refit_conditions) > 0:
            message = (
                "You added a refit_condition without setting the computation_mode to refit."
                " The condition will be ignored."
            )
            warnings.warn(message)
            logger.warning(message)
        self.result_steps: Dict[str, ResultStep] = {}

        self.lag = lag
        self.retrain_batch = retrain_batch

    def get_result(self,
                   start: pd.Timestamp,
                   return_all=False,
                   minimum_data=(0, pd.Timedelta(0))):
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

    def _fit(self, inputs: Dict[str, BaseStep], target_step):
        # Fit the encapsulate module, if the input and the target is not stopped.
        start_time = time.time()
        self.module.fit(**inputs, **target_step)
        self.training_time.set_kv("", time.time() - start_time)

    def _callbacks(self):
        # plots and writs the data if the step is finished.
        for callback in self.callbacks:
            if isinstance(callback, BaseCallback):
                callback.set_filemanager(self.file_manager)
            callback(self.buffer)

    def get_summaries(self, start):
        """
        Returns the fit times as summaries.
        :return: The summary as markdown formatted string
        :rtype: Str
        """
        self._callbacks()
        return [self.transform_time, self.training_time]

    def _transform(self, input_data):
        # TODO has to be more general for sktime
        if isinstance(self.module, BaseEstimator) and not self.module.is_fitted:
            message = f"Try to call transform in {self.name} on not fitted module {self.module.name}"
            logger.error(message)
            raise NotFittedException(message, self.name, self.module.name)

        # TODO looks a bit hacky?
        if self.method is None:
            method = getattr(
                self.module,
                list(set(self.module.__dir__()) & {"transform", "predict"})[0],
            )
        else:
            method = getattr(self.module, self.method)

        start_time = time.time()
        result = method(**input_data)
        self.transform_time.set_kv("", time.time() - start_time)
        return self._post_transform(result)

    def _post_transform(self, result):
        if not isinstance(result, dict):
            result = {self.name: result}
        for key, res in result.items():
            self.update_buffer(res, key)
        return result

    @staticmethod
    def temporal_align_inputs(inputs, target=None):
        """
        Takes inputs and target time series and temporally align them
        Args:
            inputs: Input time series
            target: Target time series

        Returns: Tuple of aligned input and target time series

        """
        # TODO handle different named dims
        dims = set()
        for inp in inputs.values():
            dims.update(inp.dims)

        if target is not None:
            for inp in target.values():
                dims.update(inp.dims)
        dims.remove(_get_time_indexes(inputs)[0])
        if target is None:
            return dict(
                zip(inputs.keys(), xr.align(*list(inputs.values()), exclude=dims))
            ), {}
        result = dict(
            zip(
                list(inputs.keys()) + list(target.keys()),
                xr.align(
                    *list(inputs.values()), *list(target.values()), exclude=dims
                ),
            )
        )

        return {key: result[key] for key in inputs.keys()}, {
                key: result[key] for key in target.keys()}

    @classmethod
    def load(cls, stored_step: Dict, inputs, targets, module, file_manager):
        """
        Load a stored step.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        if stored_step["condition"]:
            with open(stored_step["condition"], "rb") as pickle_file:
                condition = cloudpickle.load(pickle_file)
        else:
            condition = None
        refit_conditions = []
        for refit_condition in stored_step["refit_conditions"]:
            with open(refit_condition, "rb") as pickle_file:
                refit_conditions.append(cloudpickle.load(pickle_file))

        callbacks = []
        for callback_path in stored_step["callbacks"]:
            with open(callback_path, "rb") as pickle_file:
                callback = cloudpickle.load(pickle_file)
            callbacks.append(callback)

        step = cls(
            module,
            inputs,
            targets=targets,
            file_manager=file_manager,
            condition=condition,
            refit_conditions=refit_conditions,
            callbacks=callbacks,
        )
        step.default_run_setting = RunSetting.load(stored_step["default_run_setting"])
        step.current_run_setting = step.default_run_setting.clone()
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        step.method = stored_step["method"]

        return step

    def _compute(self, start, minimum_data):
        input_data = self._get_inputs(self.input_steps, start, minimum_data)
        target = self._get_inputs(self.targets, start, minimum_data)
        input_data, target = self.temporal_align_inputs(input_data, target)
        if self.current_run_setting.computation_mode in [
            ComputationMode.Default,
            ComputationMode.FitTransform,
            ComputationMode.Train,
        ]:
            self._fit(input_data, target)
        elif self.module is BaseEstimator:  # TODO more general for sktime
            logger.info("%s not fitted in Step %s", self.module.name, self.name)

        self._transform(input_data)

    def _get_inputs(self, input_steps, start, minimum_data=(0, pd.Timedelta(0))):
        min_data_module = self.module.get_min_data()
        if isinstance(min_data_module, (int, np.integer)):
            minimum_data = minimum_data[0] + min_data_module, minimum_data[1]
        else:
            minimum_data = minimum_data[0], minimum_data[1] + min_data_module
        return {
            key: inp.get_result(start, minimum_data=minimum_data)
            for key, inp in input_steps.items()
        }

    def get_json(self, fm: FileManager):
        json = super().get_json(fm)
        condition_path = None
        refit_conditions_paths = []
        callbacks_paths = []
        if self.condition:
            condition_path = fm.get_path(f"{self.name}_condition.pickle")
            with open(condition_path, "wb") as outfile:
                cloudpickle.dump(self.condition, outfile)
        for refit_condition in self.refit_conditions:
            refit_conditions_path = fm.get_path(f"{self.name}_refit_conditions.pickle")
            with open(refit_conditions_path, "wb") as outfile:
                cloudpickle.dump(refit_condition, outfile)
            refit_conditions_paths.append(refit_conditions_path)
        for callback in self.callbacks:
            callback_path = fm.get_path(f"{self.name}_callback.pickle")
            with open(callback_path, "wb") as outfile:
                cloudpickle.dump(callback, outfile)
            callbacks_paths.append(callback_path)
        json.update(
            {
                "callbacks": callbacks_paths,
                "method": self.method,
                "condition": condition_path,
                "refit_conditions": refit_conditions_paths,
            }
        )
        return json

    def refit(self, start: pd.Timestamp):
        """
        Refits the module of the step.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        if self.current_run_setting.computation_mode in [ComputationMode.Refit]:
            for refit_condition in self.refit_conditions:
                # TODO self.lag has to be named differently. It handles how much values should be
                #  considered for retraining
                condition_input = {
                    key: value.step.get_result(start- self.lag)
                    for key, value in refit_condition.kwargs.items()
                }
                if list(filter(lambda x: x is None,condition_input.values())):
                    break

                if len(condition_input) > 0:
                    condition_input, _ = self.temporal_align_inputs(
                        condition_input
                    )
                if refit_condition.evaluate(**condition_input):
                    self._refit(start)
                    break


    def _refit(self, start):
        refit_input = self._get_inputs(
            self.input_steps, start - self.retrain_batch - self.lag
        )
        refit_target = self._get_inputs(
            self.targets, start - self.retrain_batch - self.lag
        )
        # Refit only if enough data are available
        if list(filter(lambda x: x is not None, refit_input.values())):
            refit_input, refit_target = self.temporal_align_inputs(
                refit_input, refit_target
            )
            self.module.refit(**refit_input, **refit_target)

    def get_result_step(self, key: str):
        """
        Returns a new step that forwards the time series from the steps buffer with the key
        Args:
            key: Specifies which time series should be forwarded/selected

        Returns: The time series specified by the key

        """
        if key not in self.result_steps:
            self.result_steps[key] = ResultStep(
                input_steps={"result": self}, buffer_element=key
            )
        return self.result_steps[key]
