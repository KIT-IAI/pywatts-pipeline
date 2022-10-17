import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.util.run_setting import RunSetting


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.module_mock = MagicMock()
        self.module_mock.transform
        self.module_mock.name = "test"
        self.step_mock = MagicMock()
        self.step_mock.id = 2

    def tearDown(self) -> None:
        self.module_mock = None
        self.step_mock = None

    def test_fit(self):
        input_dict = {'input_data': None}
        step = Step(self.module_mock, self.step_mock, file_manager=MagicMock())
        step._fit(input_dict, {})
        self.module_mock.fit.assert_called_once_with(**input_dict)

    @patch("builtins.open")
    @patch("pywatts_pipeline.core.steps.step.cloudpickle")
    def test_store_load_of_step_with_condition(self, cloudpickle_mock, open_mock):
        condition_mock = MagicMock()
        step = Step(self.module_mock, self.step_mock, None, condition=condition_mock)
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("folder", "test_condition.pickle")
        json = step.get_json(fm_mock)
        reloaded_step = Step.load(json, [self.step_mock], targets=None, module=self.module_mock,
                                  file_manager=MagicMock())

        # One call in load and one in save
        open_mock.assert_has_calls(
            [call(os.path.join("folder", "test_condition.pickle"), "wb"),
             call(os.path.join("folder", "test_condition.pickle"), "rb")],
            any_order=True)
        self.assertEqual(json, {
            "target_ids": {},
            'method': None,
            "input_ids": {},
            "id": -1,
            'default_run_setting': {'computation_mode': 4},
            "refit_conditions": [],
            "module": "pywatts_pipeline.core.steps.step",
            "class": "Step",
            "name": "test",
            'callbacks': [],
            "last": True,
            'condition': os.path.join("folder", "test_condition.pickle")}, json)

        self.assertEqual(reloaded_step.module, self.module_mock)
        self.assertEqual(reloaded_step.input_steps, [self.step_mock])
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)
        cloudpickle_mock.dump.assert_called_once_with(condition_mock, open_mock().__enter__.return_value)

    @patch("builtins.open")
    @patch("pywatts_pipeline.core.steps.step.cloudpickle")
    def test_store_load_of_step_with_refit_conditions(self, cloudpickle_mock, open_mock):
        refit_conditions_mock = MagicMock()
        step = Step(self.module_mock, self.step_mock, None, refit_conditions=[refit_conditions_mock])
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("folder", "test_refit_conditions.pickle")
        json = step.get_json(fm_mock)
        reloaded_step = Step.load(json, [self.step_mock], targets=None, module=self.module_mock,
                                  file_manager=MagicMock())

        # One call in load and one in save
        open_mock.assert_has_calls(
            [call(os.path.join("folder", "test_refit_conditions.pickle"), "wb"),
             call(os.path.join("folder", "test_refit_conditions.pickle"), "rb")],
            any_order=True)
        self.assertEqual(json, {
            "target_ids": {},
            # Same as for test_load.
            "input_ids": {},
            "id": -1,
            'method': None,
            'default_run_setting': {'computation_mode': 4},
            "refit_conditions": [os.path.join("folder", "test_refit_conditions.pickle")],
            "module": "pywatts_pipeline.core.steps.step",
            "class": "Step",
            "name": "test",
            'callbacks': [],
            "last": True,
            'condition': None}, json),

        self.assertEqual(reloaded_step.module, self.module_mock)
        self.assertEqual(reloaded_step.input_steps, [self.step_mock])
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)
        cloudpickle_mock.dump.assert_called_once_with(refit_conditions_mock, open_mock().__enter__.return_value)

    @patch("pywatts_pipeline.core.steps.base_step._get_time_indexes", return_value="time")
    @patch("pywatts_pipeline.core.steps.base_step.xr")
    def test_transform_batch_with_existing_buffer(self, xr_mock, *args):
        # Check that data in batch learning are concatenated
        input_step = MagicMock()
        input_step._should_stop.return_value = False
        time = pd.date_range('2000-01-01', freq='1D', periods=7)
        time2 = pd.date_range('2000-01-14', freq='1D', periods=7)
        time3 = pd.date_range('2000-01-01', freq='1D', periods=14)
        input_step.get_result.return_value = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                                               coords={'time': time2})

        self.module_mock.transform.return_value = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                                               coords={'time': time2})
        self.module_mock.get_min_data.return_value = 2
        step = Step(self.module_mock, {"x": input_step}, file_manager=MagicMock())
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        step.buffer = {"test": da}
        xr_mock.concat.return_value = xr.DataArray([2, 3, 4, 3, 3, 1, 2, 2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                                   coords={'time': time3})

        # TODO: I think get results awaits list of time indexes?
        step.get_result(pd.Timestamp("2000.01.07"))

        # Two calls, once in should_stop and once in _transform
        input_step.get_result.assert_has_calls(
            [call(pd.Timestamp('2000-01-07 00:00:00'),
                  minimum_data=(2, pd.Timedelta(0))),
             call(pd.Timestamp('2000-01-07 00:00:00'),
                  minimum_data=(2, pd.Timedelta(0)))])
        xr_mock.concat.assert_called_once()

        xr.testing.assert_equal(da, list(xr_mock.concat.call_args_list[0])[0][0][0])
        xr.testing.assert_equal(xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                             coords={'time': time2}), list(xr_mock.concat.call_args_list[0])[0][0][1])
        assert {'dim': 'time'} == list(xr_mock.concat.call_args_list[0])[1]

    def test_get_result(self):
        input_step = MagicMock()

        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        input_step.get_result.return_value = da
        input_step._should_stop.return_value = False

        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        self.module_mock.transform.return_value = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                                               coords={'time': time})
        step = Step(self.module_mock, {"x": input_step}, file_manager=MagicMock())
        # TODO: I think get results awaits list of time indexes?
        step.get_result(pd.Timestamp("2000.01.01"), )

        # Two calls, once in should_stop and once in _transform
        input_step.get_result.assert_has_calls(
            [call(pd.Timestamp("2000-01-01"),
                  minimum_data=(0, self.module_mock.get_min_data().__radd__())),
             call(pd.Timestamp('2000-01-01 '),
                  minimum_data=(0, self.module_mock.get_min_data().__radd__()))])

        xr.testing.assert_equal(da, self.module_mock.transform.call_args[1]["x"])


    def test_input_finished(self):
        input_dict = {'input_data': None}
        step = Step(self.module_mock, self.step_mock, file_manager=MagicMock())
        step._fit(input_dict, {})

        self.module_mock.fit.assert_called_once_with(**input_dict)

    def test_fit_with_targets(self):
        input_dict = {'input_data': None}
        target_mock = MagicMock()

        step = Step(self.module_mock, self.step_mock, MagicMock(), targets={"target": MagicMock()})
        step._fit(input_dict, {"target": target_mock})

        self.module_mock.fit.assert_called_once_with(**input_dict, target=target_mock)

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        input_dict = {'input_data': da}
        step = Step(self.module_mock, {"x": self.step_mock}, None)
        step._fit(input_dict, {})
        step._transform(input_dict)
        xr.testing.assert_equal(
            self.module_mock.transform.call_args[1]["input_data"],
            da
        )

    def test_load(self):
        step_config = {
            "target_ids": {},
            "input_ids": {2: 'x'},
            'default_run_setting': {'computation_mode': 3},
            "id": -1,
            'method': None,
            "module": "pywatts_pipeline.core.steps.step",
            "class": "Step",
            "condition": None,
            "refit_conditions": [],
            'callbacks': [],
            "name": "test",
            "last": False,
        }
        step = Step.load(step_config, {'x': self.step_mock}, None, self.module_mock, None)

        self.assertEqual(step.name, "test")
        self.assertEqual(step.id, -1)
        self.assertEqual(step.get_json("file"), step_config)
        self.assertEqual(step.module, self.module_mock)

    def test_get_json(self):
        step = Step(self.module_mock, self.step_mock, None)
        json = step.get_json("file")
        self.assertEqual({'callbacks': [],
                          'class': 'Step',
                          'default_run_setting': {'computation_mode': 4},
                          'condition': None,
                          'id': -1,
                          'input_ids': {},
                          'last': True,
                          'method': None,
                          'module': 'pywatts_pipeline.core.steps.step',
                          'name': 'test',
                          'target_ids': {},
                          'refit_conditions': []}, json)

    def test_set_run_setting(self):
        step = Step(MagicMock(), MagicMock(), MagicMock())
        step.set_run_setting(RunSetting(ComputationMode.FitTransform))

        assert step.current_run_setting.computation_mode == ComputationMode.FitTransform

        step.set_run_setting(RunSetting(ComputationMode.Transform))
        assert step.current_run_setting.computation_mode == ComputationMode.Transform

    def test_set_computation_mode_specified(self):
        step = Step(MagicMock(), MagicMock(), MagicMock(), computation_mode=ComputationMode.FitTransform)
        assert step.current_run_setting.computation_mode == ComputationMode.FitTransform

        step.set_run_setting(RunSetting(computation_mode=ComputationMode.Transform))
        assert step.current_run_setting.computation_mode == ComputationMode.FitTransform

    def test_reset(self):
        step = Step(MagicMock(), MagicMock(), MagicMock())
        step.buffer = MagicMock()
        step.current_run_setting = RunSetting(computation_mode=ComputationMode.Transform)
        step.finished = True
        step.reset()

        self.assertIsNone(None)
        assert step.current_run_setting.computation_mode == ComputationMode.Default
        # TODO: None type does not support index access
        assert step._should_stop(None, (0, pd.Timedelta("0h"))) == False
        assert step.finished == False

    @patch('pywatts_pipeline.core.steps.step.Step._get_inputs', return_value={"x": 2})
    def test_refit_refit_conditions_false(self, get_inputs_mock):
        step = Step(self.module_mock, {"x": self.step_mock}, file_manager=None, refit_conditions=[lambda x, y: False],
                    computation_mode=ComputationMode.Refit)
        step.refit(pd.Timestamp("2000.01.01"), pd.Timestamp("2020.01.01"))
        self.module_mock.refit.assert_not_called()

    @patch('pywatts_pipeline.core.steps.step.Step._get_inputs')
    def test_refit_refit_conditions_true(self, get_inputs_mock):
        step = Step(self.module_mock, {"x": self.step_mock},
                    targets={"target": self.step_mock}, file_manager=None, refit_conditions=[lambda x, y: True],
                    computation_mode=ComputationMode.Refit)

        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        get_inputs_mock.side_effect = [{"x": da},  {"target": da}, {"x": da},  {"target": da}]

        step.refit(pd.Timestamp("2000.01.01"), pd.Timestamp("2020.01.01"))

        xr.testing.assert_equal(
            self.module_mock.refit.call_args[1]["x"],
            da
        )
        xr.testing.assert_equal(
            self.module_mock.refit.call_args[1]["target"],
            da
        )

    @patch('pywatts_pipeline.core.steps.step.Step._get_inputs', )
    def test_multiple_refit_conditions(self, get_inputs_mock):
        step = Step(self.module_mock, {"x": self.step_mock},
                    targets={"target": self.step_mock}, file_manager=None,
                    refit_conditions=[lambda x, y: False, lambda x, y: True],
                    computation_mode=ComputationMode.Refit)
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        get_inputs_mock.side_effect = [{"x": da}, {"target": da}, {"x": da}, {"target": da}, {"x": da}, {"target": da}]

        step.refit(pd.Timestamp("2000.01.01"), pd.Timestamp("2020.01.01"))
        xr.testing.assert_equal(
            self.module_mock.refit.call_args[1]["x"],
            da
        )
        xr.testing.assert_equal(
            self.module_mock.refit.call_args[1]["target"],
            da
        )