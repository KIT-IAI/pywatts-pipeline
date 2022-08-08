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
    :param input_step: The input_step of the module.
    :type input_step: Step
    :param file_manager: The file_manager which is used for storing data.
    :type file_manager: FileManager
    :param target: The step against which's output the module of the current step should be fitted. (Default: None)
    :type target: Optional[Step]
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

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, *,
                 targets: Optional[Dict[str, "BaseStep"]] = None, method=None,
                 computation_mode=ComputationMode.Default,
                 callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]] = [],
                 condition=None,
                 batch_size: Optional[None] = None,
                 refit_conditions=[],
                 retrain_batch=pd.Timedelta(hours=24),
                 lag=pd.Timedelta(hours=24)):
        self.method = method
        super().__init__(input_steps, targets, condition=condition,
                         computation_mode=computation_mode, name=module.name)
        self.file_manager = file_manager
        self.module = module
        self.retrain_batch = retrain_batch
        self.callbacks = callbacks
        self.batch_size = batch_size
        if self.current_run_setting.computation_mode is not ComputationMode.Refit and len(refit_conditions) > 0:
            message = "You added a refit_condition without setting the computation_mode to refit." \
                      " The condition will be ignored."
            warnings.warn(message)
            logger.warning(message)
        self.lag = lag
        self.refit_conditions = refit_conditions
        self.result_steps: Dict[str, ResultStep] = {}

    def get_result(self, start: pd.Timestamp, return_all=False, minimum_data=(0, pd.Timedelta(0))):
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
        self.module.fit(**inputs, **target_step)

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

    def _transform(self, input_step):
        # TODO has to be more general for sktime
        if isinstance(self.module, BaseEstimator) and not self.module.is_fitted:
            message = f"Try to call transform in {self.name} on not fitted module {self.module.name}"
            logger.error(message)
            raise NotFittedException(message, self.name, self.module.name)

        input_data = self.temporal_align_inputs(input_step)
        for val in input_data.values():
            if val is None or len(val) == 0:
                return
        # TODO looks a bit hacky?
        if self.method is None:
            method = getattr(self.module, list(set(self.module.__dir__()) & {"transform", "predict"})[0])
        else:
            method = getattr(self.module, self.method)
        result = method(**input_data)
        return self._post_transform(result)

    def _post_transform(self, result):
        if not isinstance(result, dict):
            result = {self.name: result}
        for key, res in result.items():
            self.update_buffer(res, key)
        return result

    def temporal_align_inputs(self, inputs, target=None):
        # TODO handle different named dims
        # TODO move to step?
        dims = set()
        for inp in inputs.values():
            dims.update(inp.dims)

        if target is not None:
            for inp in target.values():
                dims.update(inp.dims)
        dims.remove(_get_time_indexes(inputs)[0])
        if target is None:
            return dict(zip(inputs.keys(), xr.align(*list(inputs.values()), exclude=dims)))
        else:
            result = dict(zip(list(inputs.keys()) + list(target.keys()),
                              xr.align(*list(inputs.values()), *list(target.values()), exclude=dims)))

            return {key: result[key] for key in inputs.keys()}, \
                   {key: result[key] for key in target.keys()}

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
            with open(stored_step["condition"], 'rb') as pickle_file:
                condition = cloudpickle.load(pickle_file)
        else:
            condition = None
        refit_conditions = []
        for refit_condition in stored_step["refit_conditions"]:
            with open(refit_condition, 'rb') as pickle_file:
                refit_conditions.append(cloudpickle.load(pickle_file))

        callbacks = []
        for callback_path in stored_step["callbacks"]:
            with open(callback_path, 'rb') as pickle_file:
                callback = cloudpickle.load(pickle_file)
            callback.set_filemanager(file_manager)
            callbacks.append(callback)

        step = cls(module, inputs, targets=targets, file_manager=file_manager, condition=condition,
                   refit_conditions=refit_conditions, callbacks=callbacks, batch_size=stored_step["batch_size"])
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
        if self.current_run_setting.computation_mode in [ComputationMode.Default, ComputationMode.FitTransform,
                                                         ComputationMode.Train]:
            input_data, target = self.temporal_align_inputs(input_data, target)
            start_time = time.time()
            self._fit(input_data, target)
            self.training_time.set_kv("", time.time() - start_time)
        elif self.module is BaseEstimator:
            logger.info("%s not fitted in Step %s", self.module.name, self.name)

        start_time = time.time()
        self._transform(input_data)
        self.transform_time.set_kv("", time.time() - start_time)

    def _get_inputs(self, inputs, start, minimum_data=(0, pd.Timedelta(0))):
        min_data_module = self.module.get_min_data()
        if isinstance(min_data_module, (int, np.integer)):
            minimum_data = minimum_data[0] + min_data_module, minimum_data[1]
        else:
            minimum_data = minimum_data[0], minimum_data[1] + min_data_module
        return {
            key: inp.get_result(start, minimum_data=minimum_data)
            for key, inp in inputs.items()
        }

    def get_json(self, fm: FileManager):
        json = super().get_json(fm)
        condition_path = None
        refit_conditions_paths = []
        callbacks_paths = []
        if self.condition:
            condition_path = fm.get_path(f"{self.name}_condition.pickle")
            with open(condition_path, 'wb') as outfile:
                cloudpickle.dump(self.condition, outfile)
        for refit_condition in self.refit_conditions:
            refit_conditions_path = fm.get_path(f"{self.name}_refit_conditions.pickle")
            with open(refit_conditions_path, 'wb') as outfile:
                cloudpickle.dump(refit_condition, outfile)
            refit_conditions_paths.append(refit_conditions_path)
        for callback in self.callbacks:
            callback_path = fm.get_path(f"{self.name}_callback.pickle")
            with open(callback_path, 'wb') as outfile:
                cloudpickle.dump(callback, outfile)
            callbacks_paths.append(callback_path)
        json.update({"callbacks": callbacks_paths,
                     "method": self.method,
                     "condition": condition_path,
                     "refit_conditions": refit_conditions_paths,
                     "batch_size": self.batch_size})
        return json

    def refit(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Refits the module of the step.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        if self.current_run_setting.computation_mode in [ComputationMode.Refit] and isinstance(self.module,
                                                                                               BaseEstimator):
            for refit_condition in self.refit_conditions:
                if isinstance(refit_condition, BaseCondition):
                    condition_input = {key: value.step.get_result(start) for key, value in
                                       refit_condition.kwargs.items()}
                    for val in condition_input.values():
                        if val is None:
                            return
                    if refit_condition.evaluate(**condition_input):
                        self._refit(end)
                        break
                elif isinstance(refit_condition, Callable):
                    input_data = self._get_inputs(self.input_steps,start)
                    target = self._get_inputs(self.targets, start)
                    if refit_condition(input_data, target):
                        self._refit(end)
                        break

    def _refit(self, end):
        # TODO there is something wrong with end... We need to introduce a lag here?
        refit_input = self._get_inputs(self.input_steps, end - self.retrain_batch - self.lag)
        refit_target = self._get_inputs(self.targets, end - self.retrain_batch - self.lag)
        refit_input, refit_target = self.temporal_align_inputs(refit_input, refit_target)

        self.module.refit(**refit_input, **refit_target)

    def get_result_step(self, item: str):
        if item not in self.result_steps:
            self.result_steps[item] = ResultStep(input_steps={"result": self}, buffer_element=item)
        return self.result_steps[item]
