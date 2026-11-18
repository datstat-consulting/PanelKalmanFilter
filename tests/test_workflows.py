import numpy as np
import pandas as pd
from scipy.stats import norm

from PanelKalmanFilter import PanelKalmanFilter


def test_filter_includes_all_likelihood_increments_and_handles_missing_data():
    panel = pd.DataFrame([[1., np.nan, 2.]], index=["a"])
    model = PanelKalmanFilter(1, 1, panel)
    params = np.array([1., 0., 1., 0.])
    result = model.filter(params)
    np.testing.assert_allclose(result.means[0, :, 0], [0.5, 0.5, 11 / 7])
    np.testing.assert_allclose(result.log_likelihood[0], [
        norm.logpdf(1, loc=0, scale=np.sqrt(2)), 0,
        norm.logpdf(2, loc=0.5, scale=np.sqrt(3.5)),
    ])
    assert model.sampleKF(params, model.init_exp, model.init_var)[0] == -result.log_likelihood.sum()
    assert (model.smooth(params).covariances <= result.covariances + 1e-12).all()


def test_panel_exog_alignment_and_forecast():
    panels = pd.DataFrame([[1., 2., 3.], [2., 3., 5.]], index=["x", "y"])
    exog = pd.DataFrame(1.0, index=panels.index, columns=panels.columns)
    model = PanelKalmanFilter(1, 1, panels, [exog.iloc[::-1, ::-1]])
    params = np.array([1., 0., 1., 0., 2., 3.])
    future = pd.DataFrame(2.0, index=["y", "x"], columns=[3, 4])
    forecast = model.wrapForecast(params, [[1.], [4.]], [future])
    np.testing.assert_allclose(forecast, [[5, 5], [10, 10]])
    assert forecast.index.tolist() == ["x", "y"]
    model.fit(options={"maxiter": 3})
    assert model.forecast(2).shape == (2, 2)
