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

from pywatts_pipeline.core.steps.inverse_step import InverseStep
from pywatts_pipeline.core.steps.probabilistic_step import ProbablisticStep
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
from pywatts_pipeline.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts_pipeline.core.steps.summary_step import SummaryStep
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes, get_last, get_start
from pywatts_pipeline.utils._pywatts_json_encoder import PyWATTSJsonEncoder
from pywatts_pipeline.core.summary.summary_formatter import SummaryMarkdown, SummaryJSON, SummaryFormatter

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='pywatts.log',
                    level=logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARN)


class Pipeline(BaseTransformer):
    """
    The pipeline class is the central class of pyWATTS. It is responsible for
    * The interaction with the user
    * starting the execution of the pipeline
    * loading and saving the pipeline

    :param path: The path where the results of the pipeline should be stored (Default: ".")
    :type path: str
    """

    def __init__(self, path: Optional[str] = ".", name="Pipeline"):
        # TODO integrate Miraes API
        super().__init__(name)
        self.assembled_steps = []
        self.result = {}
        self.start_steps = dict()
        self.steps: List[BaseStep] = []
        self.step_counter = 0
        if path is None:
            self.file_manager = None
        else:
            self.file_manager = FileManager(path)

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
        for step in self.assembled_steps:
            step.finished = False

        # Fill the start_step buffers
        for key, (start_step, _) in self.start_steps.items():
            start_step.update_buffer(x[key].copy())

        # Get start date for the new calculation (last date of the previous one)
        start = None if len(self.result) == 0 else get_last(self.result)
        last_steps = list(filter(lambda x: x.last and not isinstance(x, SummaryStep), self.assembled_steps))
        result = self._collect_results(last_steps, start)

        # TODO could we combine the following lines with _collect_results and _add_to_result?
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
        result = dict()
        for i, step in enumerate(last_steps):
            res = step.get_result(start, return_all=True)
            if res is not None:
                for key, value in res.items():
                    result = self._add_to_result(i, key, value, result)
        return result

    def _add_to_result(self, i, key, res, result):
        if key in result.keys():
            message = f"Naming Conflict: {key} is renamed to. {key}_{i}"
            warnings.warn(message)
            logger.info(message)
            result[f"{key}_{i}"] = res
        else:
            result[key] = res
        return result

    parameters = {
        "steps": list,
        "assembled_steps": list,
        "estimators": list,
        "outputs": list,
    }

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameter of a pipeline module
        :return: Dictionary containing information about this module
        :rtype: Dict
        """
        return {}

    def make_list(self, obj):
        """Make obj a list if it isn't already."""
        if isinstance(obj, list):
            return obj
        return [obj]

    def set_params(self, **kwargs):
        """
        Set params of pipeline module.
        """
        for parameter, value in kwargs.items():
            if parameter == "assembled_step":
                self.step_counter = 0
                for step in parameter:
                    self.assembled_steps.append(step)
                setattr(self, parameter, value)

            elif parameter in self.parameters.keys():#TODO and type(value) in self.make_list(
            #    self.parameters[parameter]
            #):
                setattr(self, parameter, value)
            else:
                raise ValueError(
                    "".join(
                        [
                            f"{type(self).__name__} doesn't have a parameter ",
                            f"{parameter} which accepts input of type",
                            f"{type(value).__name__}. ",
                            f"Valid options and types are {self.parameters}",
                        ]
                    )
                )

    def test(self, data: Union[pd.DataFrame, xr.Dataset], summary: bool = True,
             summary_formatter: SummaryFormatter = SummaryMarkdown(), refit=False, reset=False):
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
        # TODO rename to transform?
        return self._run(data, ComputationMode.Transform, summary, summary_formatter, refit=refit, reset=reset)

    def train(self, data: Union[pd.DataFrame, xr.Dataset], summary: bool = True,
              summary_formatter: SummaryFormatter = SummaryMarkdown(), reset=True):
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
        # TODO rename to fit?
        return self._run(data, ComputationMode.FitTransform, summary, summary_formatter, reset=reset)

    def _run(self, data: Union[pd.DataFrame, xr.Dataset], mode: ComputationMode, summary: bool,
             summary_formatter: SummaryFormatter, reset=False, refit=False):

        if reset:
            for step in self.assembled_steps:
                self.result = {}
                step.reset()

        self.current_run_setting = RunSetting(computation_mode=mode,
                                              summary_formatter=summary_formatter,
                                              return_summary=summary)
        for step in self.assembled_steps:
            step.set_run_setting(self.current_run_setting)

        data = self.check_input(data)

        result = self._transform(data)

        if refit:
            self.refit(start=get_start(data), end=get_last(data))

        # TODO handle callbacks
        if summary:
            self._create_summary(summary_formatter)
            return result, summary

        return result

    def check_input(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_xarray()
            data = {key: data[key] for key in data.data_vars}
        elif isinstance(data, xr.Dataset):
            data = {key: data[key] for key in data.data_vars}
        elif isinstance(data, dict):
            for key in data:
                if not isinstance(data[key], xr.DataArray):
                    raise WrongParameterException(
                        "Input Dict does not contain xr.DataArray objects.",
                        "Make sure to pass Dict[str, xr.DataArray].",
                        self.name)
        else:
            raise WrongParameterException(
                "Unkown data type to pass to pipeline steps.",
                "Make sure to use pandas DataFrames, xarray Datasets, or Dict[str, xr.DataArray].",
                self.name)
        return data

    def add(self, *,
            module: Union[BaseStep],
            input_ids: List[int] = None,
            target_ids: List[int] = None):
        """
        Add a new module with all of it's inputs to the pipeline.

        :param target_ids: The target determines the module which provides the target value.
        :param module: The module which should be added
        :param input_ids: A list of modules, whose input is needed for this steps
        :return: None
        """
        # TODO extend it to Miraes API

        # register modules in the pipeline
        self._register_step(module, set_assembled=False)

        #logger.info(
         #   f"Add {self.steps[-1]} to the pipeline. Inputs are {[step for step in self.get_steps_by_ids(input_ids)]}"
         #   f"{'.' if not input_ids else f' and the target is {[step for step in self.get_steps_by_ids(target_ids)]}.'}"
        #)

    def get_step(self, edge, steps):
        for step in steps:
            if step.name == edge:
                return StepInformation(step=step, pipeline=self)
        if edge not in self.start_steps.keys():
            start_step = StartStep(edge)
            self.start_steps[edge] = start_step, StepInformation(step=start_step, pipeline=self)
            start_step.name = edge
            self.add(module=start_step)
            self.steps.append(start_step)
            params = self.assemble(self.steps)
            self.set_params(**params)
            return self.start_steps[edge][-1]



    def add_new_api(self, estimator, name, input_edges, use_inverse_transform: bool = False,
                    use_prob_transform: bool = False,
                    callbacks: List = [],
                    condition: Optional = None,
                    computation_mode: ComputationMode = ComputationMode.Default,
                    batch_size: Optional[pd.Timedelta] = None,
                    refit_conditions: List[Union[Callable, bool]] = [],
                    lag: Optional[int] = pd.Timedelta(hours=0),
                    retrain_batch: Optional[int] = pd.Timedelta(hours=24), ):

        # @mirae Thanks for your api proposal.
        kwargs = {key: self.get_step(edge, self.assembled_steps) for key, edge in input_edges.items()}

        # TODO Check that it does the same as GraphNode.
        from pywatts_pipeline.core.steps.step_factory import StepFactory
        if isinstance(estimator, BaseSummary):
            step = StepFactory().create_summary(module=estimator, kwargs=kwargs)
        else:
            step = StepFactory().create_step(module=estimator, kwargs=kwargs,  use_inverse_transform=use_inverse_transform,
                                         use_predict_proba=use_prob_transform,
                                         condition=condition,
                                         callbacks=callbacks,
                                         computation_mode=computation_mode, batch_size=batch_size,
                                         refit_conditions=refit_conditions if isinstance(refit_conditions, list) else [
                                             refit_conditions
                                         ],
                                    #     retrain_batch=retrain_batch,
                                         lag=lag)
        self.add(module=step.step)
        step.step.name = name
        self.steps.append(step.step)
        params = self.assemble(self.steps)
        self.set_params(**params)

    def assemble(self, steps: list, outputs=None):
        estimators = [node.module for node in filter(lambda x: isinstance(x, Step), steps)]
        est_dict = {}
        for est in estimators:
            est_dict[id(est)] = deepcopy(est)
        assembled_steps = []
        for node in steps:
            from pywatts_pipeline.core.steps.step_factory import StepFactory

            # TODO Check that it does the same as GraphNode.__init__

            # TODO add ids only for assembled_steps?
            if isinstance(node, StartStep):
                step = node
            elif isinstance(node, SummaryStep):
                # TODO this is not working. node.input_steps is related to self.steps and not to self.assembled_steps
                inputs = {key: StepInformation(self.get_step(value.name, assembled_steps).step, self) for key, value in node.input_steps.items()}
                inputs.update({key: StepInformation(self.get_step(value.name, assembled_steps).step, self)  for key, value in node.targets.items()})
                step = StepFactory().create_summary(module=est_dict[id(node.module)], kwargs=inputs).step
            else:
                inputs = {key: StepInformation(self.get_step(value.name, assembled_steps).step, self) for key, value in node.input_steps.items()}
                inputs.update({key: StepInformation(self.get_step(value.name, assembled_steps).step, self)  for key, value in node.targets.items()})
                step = StepFactory().create_step(module=est_dict[id(node.module)], kwargs=inputs,
                                             use_predict_proba=True if isinstance(node, ProbablisticStep) else False,
                                             use_inverse_transform=True if isinstance(node, InverseStep) else False,
                                             callbacks=node.callbacks,
                                             condition=node.condition,
                                             batch_size=node.batch_size,
                                             computation_mode=node.default_run_setting.computation_mode,
                                             refit_conditions=node.refit_conditions,
                                             #retrain_batch =node.retrain_batch,
                                             lag=node.lag
                                             ).step
            step.id = node.id
            step.name = node.name
            assembled_steps.append(step)
        params = {
            "steps": steps,
            "assembled_steps": assembled_steps,
            "estimators": estimators,
            "outputs": outputs,
        }
        return params

    def get_steps_by_ids(self, ids: List[int]):
        return list(filter(lambda x: x.id in ids, self.assembled_steps))

    def _register_step(self, step, set_assembled=True):
        """
        Registers the step in the pipeline.

        :param step: the step to be registered
        """
        step_id = self.step_counter
        self.step_counter += 1
        #if set_assembled:
        #    self.assembled_steps.append(step)
        step.id = step_id

    def save(self, fm: FileManager):
        """
        Saves the pipeline. Note You should not call this method from outside of pyWATTS. If you want to store your
        pipeline then you should use to_folder.
        """
        json_module = super().save(fm)
        path = os.path.join(str(fm.basic_path), self.name)
        if os.path.isdir(path):
            number = len(glob.glob(f'{path}*'))
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

    def to_folder(self, path: Union[str, Path]):
        """
        Saves the pipeline in pipeline.json in the specified folder.

        :param path: path of the folder
        :return: None
        """
        if not isinstance(path, Path):
            path = Path(path)
        save_file_manager = FileManager(path, time_mode=False)

        modules = []
        # 1. Iterate over steps and collect all modules -> With step_id to module_id
        #    Create for each step dict with information for restorage
        steps_for_storing = []
        for step in self.assembled_steps:
            step_json = step.get_json(save_file_manager)
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
            "version": 1,
            "modules": modules_for_storing,
            "steps": steps_for_storing,
            "path": self.file_manager.basic_path if self.file_manager else None,
        }
        file_path = save_file_manager.get_path('pipeline.json')
        with open(file_path, 'w') as outfile:
            json.dump(obj=stored_pipeline, fp=outfile, sort_keys=False, indent=4, cls=PyWATTSJsonEncoder)

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
            raise IOException(f"Path {load_path} does not exist"
                              f"Check the path which you passed to the from_folder method.")

        # load json file
        file_path = os.path.join(load_path, 'pipeline.json')
        with open(file_path, 'r') as outfile:
            json_dict = json.load(outfile)

        # load general pipeline config
        if file_manager_path is None:
            file_manager_path = json_dict.get('path', ".")

        pipeline = Pipeline(file_manager_path)
        # 1. load all modules
        modules = {}  # create a dict of all modules with their id from the json
        for i, json_module in enumerate(json_dict["modules"]):
            modules[i] = pipeline._load_modules(json_module)

        # 2. Load all steps
        for step in json_dict["steps"]:
            step = pipeline._load_step(modules, step)
            pipeline.steps.append(step)
            params = pipeline.assemble(pipeline.steps)
            pipeline.assembled_steps = pipeline.steps
            pipeline.set_params(**params)

        # TODO what happens with steps and assembled steps if the pipeline is loaded?

        pipeline.start_steps = {element.index: (element, StepInformation(step=element, pipeline=pipeline))
                                for element in filter(lambda x: isinstance(x, StartStep), pipeline.assembled_steps)}

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
        loaded_step = klass.load(step,
                                 inputs={key: self.get_steps_by_ids([int(step_id)])[0] for step_id, key in
                                         step["input_ids"].items()},
                                 targets={key: self.get_steps_by_ids([int(step_id)])[0] for step_id, key in
                                          step["target_ids"].items()},
                                 module=module,
                                 file_manager=self.file_manager)
        return loaded_step

    def __getitem__(self, item: str):
        """
        Returns the step_information for the start step corresponding to the item
        """
        # TODO seems to be a bit weird...
        if item not in self.start_steps.keys():
            start_step = StartStep(item)
            self.start_steps[item] = start_step, StepInformation(step=start_step, pipeline=self)
            self.add(module=start_step, input_ids=[], target_ids=[])
        return self.start_steps[item][-1]

    def _create_summary(self, summary_formatter, start=None):
        summaries = []
        for step in self.assembled_steps:
            summaries.extend(step.get_summaries(start))
        return summary_formatter.create_summary(summaries, self.file_manager)

    def refit(self, start, end):
        """
        Refits all steps inside of the pipeline.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        for step in self.assembled_steps:
            # A lag is needed, since if we have a 24 hour forecast we can evaluate the forecast not until 24 hours
            # are gone, since before not all target variables are available
            if isinstance(step, Step):
                # TODO in the new online setting we need no lag or do we?
                step.refit(start - step.lag, end - step.lag)
