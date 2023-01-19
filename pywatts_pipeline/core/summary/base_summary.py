# pylint: disable=W0233
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import xarray as xr

from pywatts_pipeline.core.transformer.base import Base
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.steps.step_information import SummaryInformation
from pywatts_pipeline.core.summary.summary_object import SummaryObject

if TYPE_CHECKING:
    pass


class BaseSummary(Base, ABC):
    """
    This is the base class of the modules. It manages the basic functionality of modules. BaseTransformer and
    BaseEstimator inherit from this class.

    :param name: Name of the module
    :type name: str
    """

    def fit(self, **kwargs):
        """
        Dummy method of fit, which does nothing
        :return:
        """

    @abstractmethod
    def transform(
        self, file_manager: FileManager, **kwargs: xr.DataArray
    ) -> SummaryObject:
        """
        Transform method. Here the summary should be calculated.
        :param file_manager: The filemanager, it can be used to store data that corresponds to the summary as a file.
        :type: file_manager: FileManager
        :param kwargs: The input data for which a summary should be calculated.
        :type kwargs: xr.DataArray
        :return: A markdown formatted string that contains the summary.
        :rtype: SummaryObject
        """

    def __call__(self, **kwargs) -> SummaryInformation:
        """
        Adds this module to pipeline by creating step and step information

        :param inputs: The input for the current step. If the input is a pipeline, then the corresponding module and
                       step is a starting step in the pipeline. If inputs is a list then the elements of the list have
                       to be a StepInformation or a tuple of Stepinformations. If it is a StepInformation then the input
                       has to be provided for calculating the next step. If it is a tuple, at least the result of one of
                       the steps in the tuple of step information must be provided for calculating the next step. The
                       tuples can be used for merging to path after an if statement.
        :type inputs: Union[Pipeline, List[Union[StepInformation, Tuple[StepInformation]]]

        :rtype: SummaryInformation
        """

        non_supported_kwargs = [
            "use_inverse_transform",
            "refit_conditions",
            "callbacks",
            "condition",
            "computation_mode",
            "batch_size",
        ]

        for kwa in non_supported_kwargs:
            if kwa in kwargs:
                warnings.warn(
                    f"{kwa} is set for {self.name}. However, {self.name} is a SummaryModule and the"
                    f" corresponding step do not support {kwa}."
                )

        pipeline = self._extract_pipeline(kwargs)

        self.name = f"{self.name}_{len(pipeline.steps)}"
        edges = {k : v.step.name for k, v in kwargs.items()}
        return pipeline.add(
            self,
            name=self.name,
            input_edges=edges,
        )
