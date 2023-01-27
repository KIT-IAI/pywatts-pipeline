import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.steps.summary_step import SummaryStep


class TestSummaryStep(unittest.TestCase):
    def setUp(self) -> None:
        self.module_mock = MagicMock()
        self.module_mock.transform.return_value = "#I AM MARKDOWN"
        self.module_mock.name = "test"
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        self.result_dummy = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        self.step_mock = MagicMock()
        self.step_mock.get_result.return_value = self.result_dummy
        self.step_mock.id = 2
        self.fm_mock = MagicMock()
        self.summary = SummaryStep(self.module_mock, {"input": self.step_mock}, self.fm_mock)

    def tearDown(self) -> None:
        self.module_mock = None
        self.step_mock = None
        self.summary = None
        self.fm_mock = None

    def test_get_summary(self):
        self.module_mock.get_min_data.return_value = pd.Timedelta(hours=1)
        result = self.summary.get_summaries(None)

        self.step_mock.get_result.assert_called_once_with(None, minimum_data=(0, pd.Timedelta(hours=1)))
        self.assertEqual(self.module_mock.transform.call_args[1]["file_manager"], self.fm_mock)
        xr.testing.assert_equal(self.module_mock.transform.call_args[1]["input"], self.result_dummy)
        self.assertEqual(result, ["#I AM MARKDOWN"])

    def test_load(self):
        stored_step = {
            "target_ids": {},
            "input_ids": {2: "input"},
            "id": -1,
            'module': 'pywatts_pipeline.core.steps.summary_step',
            "class": "SummaryStep",
            "name": "test"}

        summary = SummaryStep.load(stored_step, {"input": self.step_mock}, {}, self.module_mock, self.fm_mock)
        self.assertEqual(len(summary.input_steps), 1)
        self.assertEqual(summary.input_steps["input"], self.step_mock)
        self.assertEqual(summary.module, self.module_mock)
        self.assertEqual(summary.file_manager, self.fm_mock)

    def test_store(self):
        fm_mock = MagicMock()
        json = self.summary.get_json(fm_mock)

        self.assertEqual(json, {'callbacks': [],
                                'class': 'SummaryStep',
                                'condition': None,
                                'default_run_setting': {'computation_mode': 4},
                                'last': True,
                                'method': None,
                                'module': 'pywatts_pipeline.core.steps.summary_step',
                                'name': 'test',
                                'refit_conditions': [],
                                }, json)
