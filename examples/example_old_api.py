# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

# Other modules required for the pipeline are imported
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# From pyWATTS the pipeline is imported
from pywatts.callbacks import LinePlotCallback
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules import CalendarExtraction, CalendarFeature, ClockShift, LinearInterpolater, SKLearnWrapper, Sampler
from pywatts.summaries import RMSE

# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="../results")

    # Extract dummy calendar features, using holidays from Germany
    # NOTE: CalendarExtraction can't return multiple features.
    calendar = CalendarExtraction(continent="Europe", country="Germany", features=[CalendarFeature.month,
                                                                                   CalendarFeature.weekday,
                                                                                   CalendarFeature.weekend]
                                  )
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time", name="imputer_power")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")

    regressor_power_statistics = SKLearnWrapper(
        module=LinearRegression(fit_intercept=True)
    )
    rmse = RMSE()

    cal = calendar(x=pipeline["load_power_statistics"])
    imputed = imputer_power_statistics(x=pipeline["load_power_statistics"])
    scaled = power_scaler(x=imputed)
    res = regressor_power_statistics(cal=cal, target=scaled, callbacks=[LinePlotCallback("linear_regression")], )
    inv_scaled = power_scaler(x =res, computation_mode=ComputationMode.Transform,
                         use_inverse_transform=True, callbacks=[LinePlotCallback("rescale")])
    rmse(y_hat=inv_scaled, y=pipeline["load_power_statistics"])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")
    train = data.iloc[:6000, :]

    assert id(rmse) != id(pipeline.assembled_steps[-1].module)
    pipeline.train(data=train)

    test = data.iloc[6000:, :]
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
