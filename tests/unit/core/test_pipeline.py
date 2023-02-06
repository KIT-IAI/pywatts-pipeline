import os
import unittest
from unittest.mock import mock_open, patch, call, MagicMock

import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.steps.start_step import StartStep
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.util.run_setting import RunSetting
from pywatts.modules import MissingValueDetector, SKLearnWrapper
from pywatts.summaries import RMSE

pipeline_json = {'id': 1,
                 'name': 'Pipeline',
                 'modules': [{'class': 'SKLearnWrapper',
                              'is_fitted': False,
                              'module': 'pywatts.modules.wrappers.sklearn_wrapper',
                              'name': 'StandardScaler',
                              'sklearn_module': os.path.join('test_pipeline', 'StandardScaler.pickle'),
                              'targets': []},
                             {'class': 'SKLearnWrapper',
                              'is_fitted': False,
                              'module': 'pywatts.modules.wrappers.sklearn_wrapper',
                              'name': 'LinearRegression',
                              'sklearn_module': os.path.join('test_pipeline', 'LinearRegression.pickle'),
                              'targets': []}],
                 'steps': [{'class': 'StartStep',
                            'default_run_setting': {'computation_mode': 4},
                            'index': 'input',
                            'inputs': {},
                            'key': 'input',
                            'last': False,
                            'module': 'pywatts_pipeline.core.steps.start_step',
                            'name': 'input',
                            'targets': {}},
                           {'callbacks': [],
                            'class': 'Step',
                            'condition': None,
                            'default_run_setting': {'computation_mode': 4},
                            'inputs': {'input': 'input'},
                            'key': 'StandardScaler',
                            'last': False,
                            'method': None,
                            'module': 'pywatts_pipeline.core.steps.step',
                            'module_id': 0,
                            'name': 'StandardScaler',
                            'refit_conditions': [],
                            'targets': {}},
                           {'callbacks': [],
                            'class': 'Step',
                            'condition': None,
                            'default_run_setting': {'computation_mode': 4},
                            'inputs': {'x': 'StandardScaler'},
                            'key': 'LinearRegression',
                            'last': True,
                            'method': None,
                            'module': 'pywatts_pipeline.core.steps.step',
                            'module_id': 1,
                            'name': 'LinearRegression',
                            'refit_conditions': [],
                            'targets': {}}],
                 'version': 2}


class TestPipeline(unittest.TestCase):

    @patch("pywatts_pipeline.core.pipeline.FileManager")
    def setUp(self, fm_mock) -> None:
        self.fm_mock = fm_mock()
        self.pipeline = Pipeline()

    def tearDown(self) -> None:
        self.pipeline = None

    # TODO does something similar
    def test_add_input_as_positional(self):
        # Should fail with an better error message
        SKLearnWrapper(LinearRegression())(x=self.pipeline["input"])

    def test_add_only_module(self):
        SKLearnWrapper(LinearRegression())(x=self.pipeline["input"])
        # nodes 1 plus startstep
        self.assertEqual(len(self.pipeline.steps), 2)

    def test_add_module_which_is_not_in_a_list(self):
        wrapper = SKLearnWrapper(LinearRegression())(input=self.pipeline["input"])
        SKLearnWrapper(LinearRegression())(x=wrapper)
        # nodes 1 plus startstep
        self.assertEqual(len(self.pipeline.steps), 3)

    def test_add_non_step_information(self):
        # This should raise an exception since pipeline might get multiple columns in the input dataframe
        with self.assertRaises(Exception) as context:
            SKLearnWrapper(StandardScaler())(x=self.pipeline)  # This should fail
        self.assertEqual(
            "The input step has an invalid type: <class 'pywatts_pipeline.core.pipeline.Pipeline'>",
            str(context.exception))

    def test_add_module_with_inputs(self):
        scaler1 = SKLearnWrapper(StandardScaler())(x=self.pipeline["x"])
        scaler2 = SKLearnWrapper(StandardScaler())(x=self.pipeline["test1"])
        SKLearnWrapper(LinearRegression())(input_1=scaler1, input_2=scaler2)

        # Three modules plus start step and one collect step
        self.assertEqual(5, len(self.pipeline.steps))

    def test_add_module_with_one_input_without_a_list(self):
        scaler = SKLearnWrapper(StandardScaler())(input=self.pipeline["test"])
        SKLearnWrapper(LinearRegression())(input=scaler)

        # Three modules plus start step and one collect step
        self.assertEqual(3, len(self.pipeline.steps))

    @patch('pywatts_pipeline.core.pipeline.FileManager')
    @patch('pywatts_pipeline.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_to_folder(self, mock_file, json_mock, fm_mock):
        scaler = SKLearnWrapper(StandardScaler())(input=self.pipeline["input"])
        SKLearnWrapper(LinearRegression())(x=scaler)
        fm_mock_object = MagicMock()
        fm_mock.return_value = fm_mock_object
        fm_mock_object.get_path.side_effect = [
            os.path.join('test_pipeline', 'StandardScaler.pickle'),
            os.path.join('test_pipeline', 'LinearRegression.pickle'),
            os.path.join('test_pipeline', 'pipeline.json'),
        ]

        self.pipeline.to_folder("test_pipeline")

        calls_open = [call(os.path.join('test_pipeline', 'StandardScaler.pickle'), 'wb'),
                      call(os.path.join('test_pipeline', 'LinearRegression.pickle'), 'wb'),
                      call(os.path.join('test_pipeline', 'pipeline.json'), 'w', encoding="utf8")]
        mock_file.assert_has_calls(calls_open, any_order=True)
        args, kwargs = json_mock.dump.call_args
        assert kwargs["obj"]["id"] == pipeline_json["id"]
        assert kwargs["obj"]["name"] == pipeline_json["name"]

        assert kwargs["obj"]["modules"] == pipeline_json["modules"]
        assert kwargs["obj"]["steps"] == pipeline_json["steps"]

    @patch('pywatts_pipeline.core.pipeline.FileManager')
    @patch('pywatts.modules.sklearn_wrapper.cloudpickle')
    @patch('pywatts_pipeline.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    @patch('pywatts_pipeline.core.pipeline.os.path.isdir')
    def ltest_from_folder(self, isdir_mock, mock_file, json_mock, pickle_mock, fm_mock):
        scaler = StandardScaler()
        linear_regression = LinearRegression()

        isdir_mock.return_value = True
        json_mock.load.return_value = pipeline_json

        pickle_mock.load.side_effect = [scaler, linear_regression]

        pipeline = Pipeline.from_folder("test_pipeline")
        calls_open = [call(os.path.join("test_pipeline", "StandardScaler.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "LinearRegression.pickle"), "rb"),
                      call(os.path.join("test_pipeline", "pipeline.json"), "r")]

        mock_file.assert_has_calls(calls_open, any_order=True)

        json_mock.load.assert_called_once()
        assert pickle_mock.load.call_count == 2

        isdir_mock.assert_called_once()
        self.assertEqual(3, len(pipeline.id_to_step))

    def test_module_naming_conflict(self):
        # This test should check, that modules with the same name do not lead to an error
        # What should this test?        
        # self.fail()
        pass

    def test_add_with_target(self):
        SKLearnWrapper(LinearRegression())(input=self.pipeline["input"], target=self.pipeline["target"])
        self.assertEqual(3, len(self.pipeline.steps))

    def test_multiple_same_module(self):
        reg_module = SKLearnWrapper(module=LinearRegression())
        reg_one = reg_module(x=self.pipeline["test"], target=self.pipeline["target"])
        reg_two = reg_module(x=self.pipeline["test2"], target=self.pipeline["target"])
        detector = MissingValueDetector()
        detector(dataset=reg_one)
        detector(dataset=reg_two)

        # Three start steps (test, test2, target), two regressors two detectors
        self.assertEqual(7, len(self.pipeline.steps))
        modules = []
        for element in self.pipeline.steps.values():
            if isinstance(element, Step) and not element.module in modules:
                modules.append(element.module)
        # One sklearn wrappers, one missing value detector
        self.assertEqual(2, len(modules))

        self.pipeline.train(
            pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2], "target": [2, 2, 4, 4, -5]},
                         index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))),
        summary=False)

    @patch('pywatts_pipeline.core.pipeline.Pipeline.create_summary')
    @patch('pywatts_pipeline.core.pipeline.FileManager')
    def test_add_pipeline_to_pipeline_and_train(self, fm_mock, create_summary_mock):
        sub_pipeline = Pipeline()

        detector = MissingValueDetector()

        detector(dataset=sub_pipeline["regression"])

        regressor = SKLearnWrapper(LinearRegression(), name="regression")(x=self.pipeline["test"],
                                                                          target=self.pipeline["target"])
        sub_pipeline(regression=regressor)

        summary_formatter_mock = MagicMock()
        self.pipeline.train(pd.DataFrame({"test": [24, 24], "target": [12, 24]}, index=pd.to_datetime(
            ['2015-06-03 00:00:00', '2015-06-03 01:00:00'])), summary_formatter=summary_formatter_mock)

        create_summary_mock.assert_called_once_with(summary_formatter_mock)

    def test_add_pipeline_to_pipeline_and_test(self):
        # Add some steps to the pipeline
        transformer = MagicMock()
        transformer.name = "magic_transformer"

        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        subpipeline = Pipeline()
        subpipeline.add(transformer, name="magic_transformer", input_edges={"x": "foo"})
        subpipeline(foo=self.pipeline["foo"])

        _, summary = self.pipeline.test(ds)

        self.assertIn("magic_transformer", summary)

    def test_if_cloned_instance_in_pipeline(self):
        wrapped_lr = SKLearnWrapper(LinearRegression(), name="lr")
        wrapped_lr(x=self.pipeline["test"])

        self.assertTrue(id(wrapped_lr) != self.pipeline.steps["lr"].module)
        self.assertTrue(isinstance(self.pipeline.steps["lr"].module, SKLearnWrapper))
        self.assertTrue(isinstance(wrapped_lr, SKLearnWrapper))
        self.assertEquals(wrapped_lr.get_params(), self.pipeline.steps["lr"].module.get_params())

    @patch("pywatts_pipeline.core.pipeline.FileManager")
    @patch('pywatts_pipeline.core.pipeline.json')
    @patch("builtins.open", new_callable=mock_open)
    def test_add_pipeline_to_pipeline_and_save(self, open_mock, json_mock, fm_mock):
        sub_pipeline = Pipeline()

        detector = MissingValueDetector()
        detector(dataset=sub_pipeline["regressor"])

        regressor = SKLearnWrapper(LinearRegression())(x=self.pipeline["test"])
        sub_pipeline(regression=regressor)

        self.pipeline.to_folder(path="path")

        self.assertEqual(json_mock.dump.call_count, 2)

    def create_summary_in_subpipelines(self):
        assert False

    @patch('pywatts_pipeline.core.pipeline.FileManager')
    def test__collect_batch_results_naming_conflict(self, fm_mock):
        step_one = MagicMock()
        step_one.name = "step"
        step_two = MagicMock()
        step_two.name = "step"
        result_step_one = MagicMock()
        result_step_two = MagicMock()
        merged_result = {
            "one": result_step_one,
            "two": result_step_two
        }

        step_one.get_result.return_value = {"step": result_step_one}
        step_two.get_result.return_value = {"step_1": result_step_two}

        result = self.pipeline._collect_results({"one" : step_one, "two": step_two}, None)

        # Assert that steps are correclty called.
        step_one.get_result.assert_called_once_with(None, return_all=True)
        step_two.get_result.assert_called_once_with(None, return_all=True)

        # Assert return value is correct
        self.assertEqual(merged_result, result)

    @patch("pywatts_pipeline.core.pipeline.FileManager")
    def test_get_params(self, fm_mock):
        result = Pipeline().get_params()
        self.assertEqual(result, {"steps":[], "model_dict":{}})

    def test_set_params(self):
        self.pipeline.set_params()
        self.assertEqual(self.pipeline.get_params(),
                         {"steps":[], "model_dict":{}})

    def test__collect_batch_results(self):
        step_one = MagicMock()
        step_one.name = "step_one"
        step_two = MagicMock()
        step_two.name = "step_two"
        result_step_one = MagicMock()
        result_step_two = MagicMock()
        merged_result = {
            "step_one": result_step_one,
            "step_two": result_step_two
        }

        step_one.get_result.return_value = {"step_one": result_step_one}
        step_two.get_result.return_value = {"step_two": result_step_two}

        result = self.pipeline._collect_results({"step_one" : step_one, "step_two": step_two}, None)

        # Assert that steps are correclty called.
        step_one.get_result.assert_called_once_with(None, return_all=True)
        step_two.get_result.assert_called_once_with(None, return_all=True)

        # Assert return value is correct
        self.assertEqual(merged_result, result)


    @patch('pywatts_pipeline.core.pipeline.FileManager')
    @patch("pywatts_pipeline.core.pipeline._get_time_indexes", return_value=["time"])
    def test_transform_pipeline(self, get_time_indexes_mock, fm_mock):
        input_mock = MagicMock()
        input_mock.indexes = {"time": ["20.12.2020"]}
        step_two = MagicMock()
        result_mock = MagicMock()
        step_two.name = "mock"
        step_two.get_result.return_value = {"mock": result_mock}
        self.pipeline._add_step(step=step_two, name="step_two")
        self.pipeline.current_run_setting = RunSetting(computation_mode=ComputationMode.Transform)

        result = self.pipeline.transform(x=input_mock)

        step_two.get_result.assert_called_once_with(None, return_all=True)
        self.assertEqual({"step_two": result_mock}, result)


    @patch('pywatts_pipeline.core.pipeline.FileManager')
    @patch("pywatts_pipeline.core.pipeline._get_time_indexes", return_value="time")
    def test_transform_pipeline_multiples(self, get_time_indexes_mock, fm_mock):
        input_mock = MagicMock()
        input_mock.indexes = {"time": ["20.12.2020"]}
        step_two = MagicMock()
        time = pd.date_range('2000-01-01', freq='1T', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})
        time2 = pd.date_range('2000-01-08', freq='1T', periods=7)
        da2 = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time2})
        step_two.name = "mock"
        step_two.get_result.return_value = {"mock": da}
        self.pipeline._add_step(step=step_two, name="step_two")
        self.pipeline.current_run_setting = RunSetting(computation_mode=ComputationMode.Transform)

        self.pipeline.transform(x=input_mock)
        step_two.get_result.assert_called_once_with(None, return_all=True)
        self.pipeline.transform(x=input_mock)
        step_two.get_result.has_calls([
            call(None, return_all=True), call("20.12.2020", return_all=True)
        ], any_order=True)
        get_time_indexes_mock.assert_called_once()
        xr.testing.assert_equal(get_time_indexes_mock.call_args[0][0]["step_two"],
                                da)
        self.assertFalse(get_time_indexes_mock.call_args[1]["get_all"])
    @patch("pywatts_pipeline.core.pipeline.FileManager")
    @patch("pywatts_pipeline.core.pipeline.Pipeline.from_folder")
    def test_load(self, from_folder_mock, fm_mock):
        created_pipeline = MagicMock()
        from_folder_mock.return_value = created_pipeline
        pipeline = Pipeline.load({'name': 'Pipeline',
                                  'class': 'Pipeline',
                                  'module': 'pywatts_pipeline.core.pipeline',
                                  'pipeline_path': 'save_path'})

        from_folder_mock.assert_called_once_with("save_path")
        self.assertEqual(created_pipeline, pipeline)

    @patch("pywatts_pipeline.core.pipeline.FileManager")
    @patch("pywatts_pipeline.core.pipeline.Pipeline.to_folder")
    @patch("pywatts_pipeline.core.pipeline.os")
    def test_save(self, os_mock, to_folder_mock, fm_mock):
        os_mock.path.join.return_value = "save_path"
        os_mock.path.isdir.return_value = False
        sub_pipeline = Pipeline()
        detector = MissingValueDetector()
        detector(dataset=sub_pipeline["test"])
        fm_mock = MagicMock()
        fm_mock.basic_path = "path_to_save"
        result = sub_pipeline.save(fm_mock)

        to_folder_mock.assert_called_once_with("save_path")
        os_mock.path.join.assert_called_once_with("path_to_save", "Pipeline")
        self.assertEqual({'name': 'Pipeline',
                          'class': 'Pipeline',
                          'module': 'pywatts_pipeline.core.pipeline',
                          'params': {},
                          'pipeline_path': 'save_path'}, result)


    @patch('pywatts_pipeline.core.pipeline.FileManager')
    def test_test(self, fm_mock):
        # Add some steps to the pipeline

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        first_step.get_result.return_value = {"first": da}
        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"Second": da}

        self.pipeline._add_step(step=first_step, name="first")
        self.pipeline._add_step(step=second_step, name="second")

        self.pipeline.test(pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                                        index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5))),
                           summary=False)

        first_step.get_result.assert_called_once_with(None, return_all=True)
        second_step.get_result.assert_called_once_with(None, return_all=True)

        first_step.set_run_setting.assert_called_once()
        self.assertEqual(first_step.set_run_setting.call_args[0][0].computation_mode, ComputationMode.Transform)
        second_step.set_run_setting.assert_called_once()
        self.assertEqual(second_step.set_run_setting.call_args[0][0].computation_mode, ComputationMode.Transform)

        first_step.reset.assert_called_once()
        second_step.reset.assert_called_once()

    @patch("builtins.open")
    @patch('pywatts_pipeline.core.pipeline.FileManager')
    def test_train(self, fmmock, open_mock):
        # Add some steps to the pipeline
        time = pd.date_range('2000-01-01', freq='1H', periods=7)

        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        first_step.get_result.return_value = {"first": da}

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"second": da}

        self.pipeline._add_step(step=first_step, name="first")
        self.pipeline._add_step(step=second_step, name="second")

        data = pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                            index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5)))
        result, summary = self.pipeline.train(data, summary=True)

        self.assertEqual(self.pipeline.result, {})

        first_step.set_run_setting.assert_called_once()
        self.assertEqual(first_step.set_run_setting.call_args[0][0].computation_mode, ComputationMode.FitTransform)
        second_step.set_run_setting.assert_called_once()
        self.assertEqual(second_step.set_run_setting.call_args[0][0].computation_mode, ComputationMode.FitTransform)

        first_step.get_result.assert_called_once_with(None, return_all=True)
        second_step.get_result.assert_called_once_with(None, return_all=True)
        self.assertEqual(first_step.reset.call_count, 2)
        self.assertEqual(second_step.reset.call_count, 2)

        xr.testing.assert_equal(result["second"], da)

    @patch('pywatts_pipeline.core.pipeline.FileManager')
    def test_train_return_no_summary(self, fmmock):
        # Add some steps to the pipeline
        time = pd.date_range('2000-01-01', freq='1H', periods=7)

        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        first_step.get_result.return_value = {"first": da}

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"second": da}

        self.pipeline._add_step(step=first_step, name="first")
        self.pipeline._add_step(step=second_step, name="second")

        data = pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                            index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5)))
        result = self.pipeline.train(data, summary=False)

        assert not isinstance(result, tuple)
        xr.testing.assert_equal(result["second"], da)
    def test_pipeline_path_none(self):
        pipeline = Pipeline(None)

        # Add some steps to the pipeline
        time = pd.date_range('2000-01-01', freq='1H', periods=7)

        da = xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"], coords={'time': time})

        # Assert that the computation is set to fit_transform if the ComputationMode was default
        first_step = MagicMock()
        first_step.computation_mode = ComputationMode.Default
        first_step.finished = False
        first_step.get_result.return_value = {"first": da}

        second_step = MagicMock()
        second_step.computation_mode = ComputationMode.Train
        second_step.finished = False
        second_step.get_result.return_value = {"second": da}

        pipeline._add_step(first_step, name="First")
        pipeline._add_step(second_step, name="Second")

        data = pd.DataFrame({"test": [1, 2, 2, 3, 4], "test2": [2, 2, 2, 2, 2]},
                            index=pd.DatetimeIndex(pd.date_range('2000-01-01', freq='24H', periods=5)))
        result = pipeline.train(data)
        assert not isinstance(result, tuple)
        assert pipeline.file_manager is None


    @patch("builtins.open", new_callable=mock_open)
    def test_horizon_greater_one_regression_inclusive_summary_file(self, open_mock):
        lin_reg = LinearRegression()
        self.fm_mock.get_path.return_value = "summary_path"

        multi_regressor = SKLearnWrapper(lin_reg)(foo=self.pipeline["foo"], target=self.pipeline["target"],
                                                  target2=self.pipeline["target2"])
        rmse = RMSE()
        rmse(y=self.pipeline["target"], prediction=multi_regressor["target"])

        time = pd.date_range('2000-01-01', freq='24H', periods=5)

        foo = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={'time': time})
        target = xr.DataArray([[2, 3], [2, 4], [2, 5], [2, 6], [2, 7]], dims=["time", "horizon"],
                              coords={'time': time, "horizon": [1, 2]})
        target2 = xr.DataArray([3, 3, 3, 3, 3], dims=["time"], coords={'time': time})

        ds = xr.Dataset({'foo': foo, "target": target, "target2": target2})

        result, summary = self.pipeline.train(ds, summary=True)

        self.assertTrue("Training Time" in summary)
        self.assertTrue("RMSE" in summary)

        self.fm_mock.get_path.assert_called_once_with("summary.md")
        open_mock().__enter__.return_value.write.assert_called_once_with(summary)

        self.assertTrue("LinearRegression__target" in result.keys())

    @patch('pywatts_pipeline.core.pipeline.isinstance', return_value=True)
    def test_refit(self, isinstance_mock):
        first_step = MagicMock()
        first_step.lag = pd.Timedelta("1d")

        self.pipeline._add_step(step=first_step, name="FOO")
        self.pipeline.refit(pd.Timestamp("2000.01.02"))

        first_step.refit.assert_called_once_with(pd.Timestamp("2000.01.02"))
