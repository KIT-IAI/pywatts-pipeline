# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

# Other modules required for the pipeline are imported
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# From pyWATTS the pipeline is imported
from pywatts.callbacks import LinePlotCallback
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.bats import BATS
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.var import VAR

from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules import CalendarExtraction, CalendarFeature, ClockShift, LinearInterpolater, SKLearnWrapper, Sampler
from pywatts.summaries import RMSE
from sktime.forecasting.naive import NaiveForecaster

# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="../results")

    # Extract dummy calendar features, using holidays from Germany
    # NOTE: CalendarExtraction can't return multiple features.
    calendar = CalendarExtraction(continent="Europe", country="Germany", features=[CalendarFeature.month_sine,
                                                                                   CalendarFeature.month_cos,
                                                                                   CalendarFeature.weekday,
                                                                                   CalendarFeature.weekend,
                                                                                   CalendarFeature.hour_cos,
                                                                                   CalendarFeature.hour_sine]
                                  )
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time", name="imputer_power")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")

    regressor_power_statistics = SKLearnWrapper(
        module=LinearRegression(fit_intercept=True)
    )
    rmse = RMSE()

    pipeline.add_new_api(calendar, "calendar", {"x": "load_power_statistics"})
    pipeline.add_new_api(imputer_power_statistics, "imputer_power", {"x": "load_power_statistics"})
    pipeline.add_new_api(power_scaler, "power_scaler", {"x": "imputer_power"})
    pipeline.add_new_api(regressor_power_statistics, "regressor_power_statistics",
                         {"cal": "calendar", "target": "power_scaler"},
                         callbacks=[LinePlotCallback("linear_regression")], )
    pipeline.add_new_api(power_scaler, "power_scaler_inverse",
                         {"x": "regressor_power_statistics"},
                         callbacks=[LinePlotCallback("Rescaler LR")], method="inverse_transform")

    # TODO make 24 hour forecasts here...
    pipeline.add_new_api(AutoETS(), "ETS", {"y": "load_power_statistics", "fh":ForecastingHorizon(np.arange(1, 24), freq="1h"),
                                            "cv" : ExpandingWindowSplitter(fh=ForecastingHorizon(np.arange(1, 24), freq="1h"))
                                            }, #method="update_predict",
                         #["load_power_statistics", "calendar"]},
                         callbacks=[LinePlotCallback("ETS")])
    # TODO use cv instead of ForecastingHorizon here?
    #pipeline.add_new_api(ARIMA(), "ARIMA", {"target": "load_power_statistics", "calendar": "calendar"},
    #                     callbacks=[LinePlotCallback("linear_regression")], fh=np.arange(1, 2760), strategy="direct")
    #pipeline.add_new_api(TrendForecaster(), "TrendForecaster", {"target": "load_power_statistics", "calendar": "calendar"},
    #                     callbacks=[LinePlotCallback("linear_regression")], fh=np.arange(1, 2760), strategy="direct")
    pipeline.add_new_api(power_scaler, "power_scaler_inverse", {"x": "regressor_power_statistics"},
                         computation_mode=ComputationMode.Transform,
                         use_inverse_transform=True, callbacks=[LinePlotCallback("rescale")])



    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")
    data = data.asfreq(pd.infer_freq(data.index))

    train = data.iloc[:6000, :]
    assert id(rmse) != id(pipeline.assembled_steps[-1].module)

    pipeline.train(data=train)
    # TODO addition of modules after training will reset the whole pipeline...
    pipeline.add_new_api(rmse, "rmse", {"y_hat": "power_scaler_inverse",
                                        "y": "load_power_statistics",
                                        "ETS": "ETS",
                                       # "ARIMA": "ARIMA",
                                       # "TrendForecaster": "TrendForecaster"
                                        })
    pipeline.assembled_steps[-3].method = "predict"
    test = data.iloc[6000:6100, :]
    result = pipeline.test(data=test, reset=True)

    # Save the pipeline to a folder
    pipeline.to_folder("./pipe_getting_started")

    print("Execute second pipeline")
    # Load the pipeline as a new instance
    pipeline2 = Pipeline.from_folder("./pipe_getting_started", file_manager_path="../pipeline2_results")
    #       WARNING
    #       Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
    #       Consequently, load only pipelines you trust with from_folder.
    #       For more details about pickling see https://docs.python.org/3/library/pickle.html

    result2 = pipeline2.test(test)
    print("Finished")
