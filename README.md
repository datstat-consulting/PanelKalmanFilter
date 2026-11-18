# PanelKalmanFilter

Estimate a shared linear Gaussian state-space model for a panel of entities. Each entity has its own latent trajectory and can have its own coefficients for exogenous regressors. The package provides maximum-likelihood fitting, filtered and smoothed state estimates, likelihood contributions, and observation forecasts.

The package grew from work on macroeconomic credit risk. The same panel structure applies to longitudinal data in economics and biostatistics. The filter uses the preceding state estimate to predict the next state and updates it after observing the current measurements. SciPy fits the shared model parameters by maximizing the Gaussian likelihood. The smoother then uses the full observation history to revise earlier state estimates.

## Installation

~~~bash
python -m pip install git+https://github.com/datstat-consulting/PanelKalmanFilter.git
~~~

Run `python -m pip install .` from a checkout to install locally. The package uses NumPy, pandas, and SciPy.

## Panel data

Supply a pandas DataFrame with one row per entity and columns in chronological order. Each time block contains `stateDim * obsDim` consecutive measurements. Within a block, group measurements by their state loading. Each exogenous regressor is a separate wide DataFrame with matching entity and measurement labels.

Use `NaN` for missing observations. The filter updates from available measurements and uses the state prediction alone when all measurements for a period are missing. Regressors must be finite. Omit `exog` when the model uses only endogenous measurements.

## Estimation on the bundled panel

The example uses [training](dummy_training.csv) and [testing](dummy_testing.csv) CSV files from the repository. Both have three entities. Training covers periods 1–24, and the held-out data cover periods 25–36. The example fits one latent state to `Y` with five exogenous variables.

~~~python
import pandas as pd
from PanelKalmanFilter import PanelKalmanFilter

training = pd.read_csv("dummy_training.csv")
testing = pd.read_csv("dummy_testing.csv")
regressors = ["sX", "fX", "pX1", "pX2", "pX3"]

def wide(frame, variable):
    return frame.pivot(index="idvar", columns="timevar", values=variable)

endog = wide(training, "Y")
exog = [wide(training, name) for name in regressors]
model = PanelKalmanFilter(stateDim=1, obsDim=1, endog=endog, exog=exog).fit()
print(model.optimization_result_.success, model.optimization_result_.message)

filtered = model.filter()
smoothed = model.smooth()
per_entity_log_likelihood = filtered.log_likelihood.sum(axis=1)

future_exog = [wide(testing, name) for name in regressors]
predictions = model.forecast(exog=future_exog)
observed = wide(testing, "Y")
mae_by_entity = (predictions - observed).abs().mean(axis=1)
~~~

`filter()` and `smooth()` return a `FilterResult` with `means` of shape `(entities, periods, stateDim)`, `covariances` of shape `(entities, periods, stateDim, stateDim)`, and per-period `log_likelihood` contributions. `smooth()` uses Rauch–Tung–Striebel smoothing and returns the fitted model's filtering likelihood alongside the smoothed estimates.

The held-out forecast above is conditional on the supplied regressor values. When future regressors are unknown, pass a separately constructed scenario or omit them to forecast with zero-valued regressors. With future regressor tables, the forecast takes its horizon and column labels from the tables. Otherwise, set the horizon with `model.forecast(steps=12)`.

The original simulation used R's `prodest::panelSim` with the following settings.

~~~r
library(prodest)
set.seed(123)
dummy_panel <- panelSim(
    N = 3, T = 360, alphaL = .6, alphaK = .4, DGP = 1,
    rho = .7, sigeps = .1, sigomg = .3, rholnw = .3
)
~~~

## Existing parameter-based workflows

`paramInit()` returns the starting parameter vector. `fit(params_init=..., method="BFGS")` selects a different optimizer and still records its SciPy result. The earlier `wrapMinimize`, `sampleKF`, and `wrapForecast` methods remain available for code that passes a parameter vector and the last filtered states explicitly.

## References

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hu, B., Qiu, W., Xu, C., & Wang, J. (2020). Integration of a Kalman filter in the geographically weighted regression for modeling the transmission of hand, foot and mouth disease. *BMC Public Health*, 20, 1–15.
- Kerbl, S., & Sigmund, M. (2011). What drives aggregate credit risk? *Oesterreichische Nationalbank Financial Stability Report*, 22.
