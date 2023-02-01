"""
Module containing a pipeline
"""
import glob
import json
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union, List, Dict, Optional, Callable

import pandas as pd
import xarray as xr

from pywatts_pipeline.core.steps.either_or_step import EitherOrStep
from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.transformer.base import BaseTransformer
from pywatts_pipeline.core.steps.base_step import BaseStep
from pywatts_pipeline.core.util.run_setting import RunSetting
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.exceptions.io_exceptions import IOException
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.steps.start_step import StartStep
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.steps.step_information import StepInformation
from pywatts_pipeline.core.exceptions.wrong_parameter_exception import (
    WrongParameterException,
)
from pywatts_pipeline.core.steps.summary_step import SummaryStep
from pywatts_pipeline.utils._xarray_time_series_utils import (
    _get_time_indexes,
    get_last,
)
from pywatts_pipeline.utils._pywatts_json_encoder import PyWATTSJsonEncoder
from pywatts_pipeline.core.summary.summary_formatter import (
    SummaryMarkdown,
    SummaryFormatter,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="pywatts.log",
    level=logging.ERROR,
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger("matplotlib").setLevel(logging.WARN)


# TODO:
#  * Check that the module is copied!
#  * How does the new API work with
#       * multi target output regressors
#       * or with subpipelineing -> No problem !
class Pipeline(BaseTransformer):
    """
    The pipeline class is the central class of pyWATTS. It is responsible for
    * The interaction with the user
    * starting the execution of the pipeline
    * loading and saving the pipeline

    :param path: The path where the results of the pipeline should be stored (Default: ".")
    :type path: str
    """

    def __init__(self, path: Optional[str] = ".", name="Pipeline", steps=None):
        super().__init__(name)
        self.steps: Dict[str, BaseStep] = {}
        self.est_dict = {}
        self.result = {}
        self.start_steps = {}
        if path is None:
            self.file_manager = None
        else:
            self.file_manager = FileManager(path)
        self._pipeline_construction_informations = steps if steps is not None else []
        if not steps is None:
            self._add(steps)
        self.current_run_setting = None

    def add(self, estimator, name, input_edges,
            callbacks: List = None,
            condition: Optional = None,
            computation_mode: ComputationMode = ComputationMode.Default,
            refit_conditions: List[Union[Callable, bool]] = None,
            lag: Optional[int] = pd.Timedelta(hours=0),
            method=None):
        """
        TODO
        """
        self.reset()
        self._pipeline_construction_informations.append([
            estimator, name, input_edges, {"callbacks": callbacks if not callbacks is None else [],
                                           "condition": condition,
                                           "computation_mode": computation_mode,
                                           "refit_conditions": refit_conditions if not refit_conditions is None else [],
                                           "lag": lag,
                                           "method": method}
        ])
        return self._add(self._pipeline_construction_informations)[-1]

    def _add(self, steps):
        self.steps = {}
        self.start_steps = {}
        from pywatts_pipeline.core.steps.step_factory import StepFactory
        step_informations = []
        for transformer, name, input_edges, add_params in steps:
            if id(transformer) not in self.est_dict:
                self.est_dict[id(transformer)] = deepcopy(transformer)

            transformer = self.est_dict[id(transformer)]
            kwargs = {key: self._get_step(edge) for key, edge in input_edges.items()}
            if isinstance(transformer, BaseSummary):
                pre_steps, step, post_steps = StepFactory().create_summary(transformer, self, kwargs)
            else:
                pre_steps, step, post_steps = StepFactory().create_step(module=transformer, kwargs=kwargs, **add_params,
                                                                        pipeline=self)
            for _step in pre_steps:
                self._add_step(_step, _step.name)
            self._add_step(step, name)
            for _step in post_steps:
                self._add_step(_step, _step.name)
            step_informations.append(StepInformation(name, self))
        return step_informations

    def _get_step(self, edge):
        # TODO: We need to enable edge like "preprocessing__calendar". TODO check if this is already possible!
        #     - This selects the input "preprocessing" and selects the column "calendar"
        column = None
        if "__" in edge:
            edge, column = edge.split("__")
        if f"{edge}__{column}" in self.steps:
            return self.steps[f"{edge}__{column}"]
        if isinstance(edge, tuple):
            kwargs = {_edge: self._get_step(_edge) for _edge in edge}
            step = EitherOrStep(kwargs)
            edge = f"EitherOr_{len(self.steps)}"
            self._add_step(step=step, name=edge)
            return self.steps[edge]
        if edge in self.steps and not column is None:
            result_step = self.steps[edge].get_result_step(column)
            self._add_step(step=result_step, name=f"{edge}__{column}")
            return result_step
        if edge in self.steps and column is None:
            return self.steps[edge]
        if not edge in self.steps:
            start_step = StartStep(edge)
            self.start_steps[edge] = start_step
            self._add_step(step=start_step, name=edge)
            return self.start_steps[edge]
        raise Exception()

    def transform(self, **x: xr.DataArray) -> xr.DataArray:
        """
        Transform the input into output, by performing all the step in this pipeline.
        Moreover, this method collects the results of the last steps in this pipeline.

        Note, this method is necessary for enabling subpipelining.

        :param x: The input data
        :type x: xr.DataArray
        :return:The transformed data
        :rtype: xr.DataArray
        """
        return self._transform(x)

    def _transform(self, x):
        # New data are arrived. Thus no step is finished anymore.
        for step in self.steps.values():
            step.finished = False

        # Fill the start_step buffers
        for key, (start_step) in self.start_steps.items():
            start_step.update_buffer(x[key].copy(), start_step.index)

        # Get start date for the new calculation (last date of the previous one)
        start = None if len(self.result) == 0 else get_last(self.result) + 1
        last_steps = list(
            map(lambda x: x, filter(lambda x: x.last and not isinstance(x, SummaryStep), self.steps.values())))
        result = self._collect_results(last_steps, start)

        # Store result in self.result.
        for key in result.keys():
            if key not in self.result:
                self.result[key] = result[key]
            else:
                dim = _get_time_indexes(result, get_all=False)
                self.result[key] = xr.concat([self.result[key], result[key]], dim=dim)
        return result

    def _collect_results(self, last_steps, start):
        # Note the return value is None if none of the inputs provide a result for this step...
        result = {}
        for i, step in enumerate(last_steps):
            res = step.get_result(start, return_all=True)
            if res is not None:
                for key, value in res.items():
                    result = self._add_to_result(i, key, value, result)
        return result

    @staticmethod
    def _add_to_result(i, key, res, result):
        if key in result.keys():
            message = f"Naming Conflict: {key} is renamed to. {key}_{i}"
            warnings.warn(message)
            logger.info(message)
            result[f"{key}_{i}"] = res
        else:
            result[key] = res
        return result

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameter of a pipeline module
        :return: Dictionary containing information about this module
        :rtype: Dict
        """
        return {}

    def set_params(self, **kwargs):
        """
        Set params of pipeline module.
        """
        if "steps" in kwargs:
            self._add(kwargs.pop("steps"))
        if "path" in kwargs:
            self.file_manager = FileManager(kwargs.pop("path"))
        if "name" in kwargs:
            self.name = kwargs.pop("name")

    def test(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        summary: bool = True,
        summary_formatter: SummaryFormatter = SummaryMarkdown(),
        refit=False,
        reset=True,
    ):
        """
        Executes all modules in the pipeline in the correct order. This method call only transform on every module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :param summary: A flag indicating if an additional summary should be returned or not.
        :type summary: bool
        :param summary_formatter: Determines the format of the summary.
        :type summary_formatter: SummaryFormatter
        :return: The result of all end points of the pipeline
        :rtype: Dict[xr.DataArray]
        """
        return self._run(
            data,
            ComputationMode.Transform,
            summary,
            summary_formatter,
            refit=refit,
            reset=reset,
        )

    def train(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        summary: bool = True,
        summary_formatter: SummaryFormatter = SummaryMarkdown(),
    ):
        """
        Executes all modules in the pipeline in the correct order. This method calls fit and transform on each module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :param summary: A flag indicating if an additional summary should be returned or not.
        :type summary: bool
        :param summary_formatter: Determines the format of the summary.
        :type summary_formatter: SummaryFormatter
        :return: The result of all end points of the pipeline
        :rtype: Dict[xr.DataArray]
        """
        result = self._run(
            data, ComputationMode.FitTransform, summary, summary_formatter, reset=True
        )
        self.reset()
        return result

    def reset(self):
        for step in self.steps.values():
            self.result = {}
            step.reset()

    def _run(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        mode: ComputationMode,
        summary: bool,
        summary_formatter: SummaryFormatter,
        reset=False,
        refit=False,
    ):
        if reset:
            self.reset()

        self.current_run_setting = RunSetting(
            computation_mode=mode,
            summary_formatter=summary_formatter,
            return_summary=summary,
        )
        for step in self.steps.values():
            step.set_run_setting(self.current_run_setting)

        data = self._check_input(data)
        result = self._transform(data)

        if refit:
            start = list(data.values())[0][_get_time_indexes(data)[0]][0].values
            self.refit(start)

        if summary and self.file_manager is not None:
            summary_data = self.create_summary(summary_formatter)
            return result, summary_data

        return result

    def _check_input(self, data):
        if isinstance(data, pd.DataFrame):
            ds = data.to_xarray()
            data = {key: ds[key] for key in ds.data_vars}
            data.update({
                key: xr.DataArray(coords={key: ds.indexes[key]}, dims=[key])
                for key, index in ds.indexes.items()
            })
        elif isinstance(data, xr.Dataset):
            data = {key: data[key] for key in data.data_vars}
        elif isinstance(data, dict):
            for key in data:
                if not isinstance(data[key], xr.DataArray):
                    raise WrongParameterException(
                        "Input Dict does not contain xr.DataArray objects.",
                        "Make sure to pass Dict[str, xr.DataArray].",
                        self.name,
                    )
        else:
            raise WrongParameterException(
                "Unkown data type to pass to pipeline steps.",
                "Make sure to use pandas DataFrames, xarray Datasets, or Dict[str, xr.DataArray].",
                self.name,
            )
        return data

    def _add_step(self, step: Union[BaseStep], name):
        self.steps[name] = step
        logger.info(
            f"Add {self.steps[name]} to the pipeline. Inputs are {self._find_step_names(step.input_steps)}"
            f" and the target is {self._find_step_names(step.targets)}."
        )

    def save(self, fm: FileManager):
        """
        Saves the pipeline. Note You should not call this method from outside of pyWATTS. If you want to store your
        pipeline then you should use to_folder.
        """
        json_module = super().save(fm)
        path = os.path.join(str(fm.basic_path), self.name)
        if os.path.isdir(path):
            number = len(glob.glob(f"{path}*"))
            path = f"{path}_{number + 1}"
        self.to_folder(path)
        json_module["pipeline_path"] = path
        json_module["params"] = {}
        return json_module

    @classmethod
    def load(cls, load_information):
        """
        Loads the pipeline.  Note You should not call this method from outside of pyWATTS. If you want to store your
        pipeline then you should use from_folder.
        """
        pipeline = cls.from_folder(load_information["pipeline_path"])
        return pipeline

    def _find_step_names(self, steps):
        names = {}
        for input_key, step in steps.items():
            for k, v in self.steps.items():
                if id(step) == id(v):
                    names[input_key] = k
                    break
        return names

    def to_folder(self, path: Union[str, Path]):
        """
        Saves the pipeline in pipeline.json in the specified folder.

        :param path: path of the folder
        :return: None
        """
        save_file_manager = FileManager(path, time_mode=False)

        modules = []
        # 1. Iterate over steps and collect all modules -> With step_id to module_id
        #    Create for each step dict with information for restorage
        steps_for_storing = []
        for key, step in self.steps.items():
            step_json = step.get_json(save_file_manager)
            step_json["key"] = key
            step_json["targets"] = self._find_step_names(step.targets)
            step_json["inputs"] = self._find_step_names(step.input_steps)
            if isinstance(step, Step):
                if step.module in modules:
                    step_json["module_id"] = modules.index(step.module)
                else:
                    modules.append(step.module)
                    step_json["module_id"] = len(modules) - 1
            steps_for_storing.append(step_json)

        # 2. Iterate over all modules and create Json and save as pickle or h5 ... if necessary...
        modules_for_storing = []
        for module in modules:
            stored_module = module.save(save_file_manager)
            modules_for_storing.append(stored_module)

        # 3. Put everything together and dump it.
        stored_pipeline = {
            "name": "Pipeline",
            "id": 1,
            "version": 2,
            "modules": modules_for_storing,
            "steps": steps_for_storing,
            "path": self.file_manager.basic_path if self.file_manager else None,
        }
        file_path = save_file_manager.get_path("pipeline.json")
        with open(file_path, "w", encoding="utf8") as outfile:
            json.dump(
                obj=stored_pipeline,
                fp=outfile,
                sort_keys=False,
                indent=4,
                cls=PyWATTSJsonEncoder,
            )

    @staticmethod
    def from_folder(load_path, file_manager_path=None):
        """
        Loads the pipeline from the pipeline.json in the specified folder
        .. warning::
            Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
            Consequently, load only pipelines you trust with `from_folder`.
            For more details about pickling see https://docs.python.org/3/library/pickle.html


        :param load_path: path to the pipeline.json
        :type load_path: str
        :param file_manager_path: path for the results and outputs
        :type file_manager_path: str
        """
        if not os.path.isdir(load_path):
            logger.error("Path %s for loading pipeline does not exist", load_path)
            raise IOException(
                f"Path {load_path} does not exist"
                f"Check the path which you passed to the from_folder method."
            )

        # load json file
        file_path = os.path.join(load_path, "pipeline.json")
        with open(file_path, "r", encoding="utf8") as outfile:
            json_dict = json.load(outfile)

        # load general pipeline config
        if file_manager_path is None:
            file_manager_path = json_dict.get("path", ".")

        pipeline = Pipeline(file_manager_path)
        # 1. load all modules

        modules = {}  # create a dict of all modules with their id from the json
        for i, json_module in enumerate(json_dict["modules"]):
            modules[i] = pipeline._load_modules(json_module)
            pipeline.est_dict[id(modules[i])] = modules[i]

        # 2. Load all steps
        for step_json in json_dict["steps"]:
            step = pipeline._load_step(modules, step_json)
            pipeline._add_step(step, step_json["key"])

        pipeline.start_steps = {
            element.index: element
            for element in filter(lambda x: isinstance(x, StartStep), pipeline.steps.values())
        }

        return pipeline

    def _load_modules(self, json_module):
        mod = __import__(json_module["module"], fromlist=json_module["class"])
        klass = getattr(mod, json_module["class"])
        return klass.load(json_module)

    def _load_step(self, modules, step):
        mod = __import__(step["module"], fromlist=step["class"])
        klass = getattr(mod, step["class"])
        module = None
        if isinstance(klass, Step) or issubclass(klass, Step):
            module = modules[step["module_id"]]
        inputs = {k: self.steps[v] for k, v in step["inputs"].items()}
        targets = {k: self.steps[v] for k, v in step["targets"].items()}
        loaded_step = klass.load(
            step,
            inputs=inputs,
            targets=targets,
            module=module,
            file_manager=self.file_manager,
        )
        return loaded_step

    def __getitem__(self, item: str):
        """
        Returns the step_information for the start step corresponding to the item
        """
        if item not in self.start_steps.keys():
            start_step = StartStep(item)
            self.start_steps[item] = start_step
            self._add_step(step=start_step, name=item)
        return StepInformation(step=self.start_steps[item], pipeline=self)

    def create_summary(self, summary_formatter: SummaryFormatter = SummaryMarkdown(), start=None):
        summaries = self._get_summaries(start)
        return summary_formatter.create_summary(summaries, self.file_manager)

    def _get_summaries(self, start):
        summaries = []
        for step in filter(lambda x: isinstance(x, Step), self.steps.values()):
            summaries.extend(step.get_summaries(start))
        return summaries

    def refit(self, start):
        """
        Refits all steps inside of the pipeline.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        for step in self.steps.values():
            # A lag is needed, since if we have a 24 hour forecast we can evaluate the forecast not until 24 hours
            # are gone, since before not all target variables are available
            if isinstance(step, Step):
                step.refit(start)
