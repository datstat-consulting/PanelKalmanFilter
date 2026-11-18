"""Maximum-likelihood estimation and prediction for independent panel units."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

__all__ = ["PanelKalmanFilter", "FilterResult"]


@dataclass(frozen=True)
class FilterResult:
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: np.ndarray


class PanelKalmanFilter:
    """Shared linear state-space model with optional unit-specific regressors.

    Endog is a wide DataFrame with one row per entity. Columns are ordered
    time-major, with stateDim * obsDim consecutive observations per period.
    Each exog DataFrame follows the same layout.
    """

    def __init__(self, stateDim, obsDim, endog, exog=None):
        if not all(isinstance(n, Integral) and n > 0 for n in (stateDim, obsDim)):
            raise ValueError("State and observation dimensions must be positive integers.")
        if not isinstance(endog, pd.DataFrame) or endog.empty:
            raise ValueError("endog must be a nonempty wide DataFrame.")
        if not endog.index.is_unique or not endog.columns.is_unique:
            raise ValueError("Panel row and column labels must be unique.")
        self.stateDim, self.obsDim = stateDim, obsDim
        self.observation_size = stateDim * obsDim
        self.N, width = endog.shape
        self.T, remainder = divmod(width, self.observation_size)
        if remainder:
            raise ValueError("Each period must contain stateDim * obsDim observations.")
        self.endog = endog.copy()
        self.exog = tuple(exog) if exog is not None else ()
        self.K = len(self.exog)
        self._observations = self.endog.to_numpy(dtype=float).reshape(
            self.N, self.T, self.observation_size
        )
        if np.isinf(self._observations).any():
            raise ValueError("Use NaN for missing observations.")
        self._exogenous = self._exog_array(self.exog, endog.columns)
        self.init_exp = np.zeros(stateDim)
        self.init_var = np.eye(stateDim)
        self.post_exp = []
        self.obsDict = [
            list(endog.columns[start:start + self.observation_size])
            for start in range(0, width, self.observation_size)
        ]
        self.defaultInitParams = self.paramInit()

    def _exog_array(self, frames, columns):
        if not frames:
            return None
        for frame in frames:
            if (not isinstance(frame, pd.DataFrame) or not frame.index.is_unique
                    or not frame.columns.is_unique
                    or set(frame.index) != set(self.endog.index)
                    or set(frame.columns) != set(columns)):
                raise ValueError("Exogenous tables must match the entity and observation labels.")
        values = np.stack([frame.reindex(index=self.endog.index, columns=columns).to_numpy(dtype=float)
                           for frame in frames], axis=-1)
        if not np.isfinite(values).all():
            raise ValueError("Exogenous values must be finite.")
        return values.reshape(self.N, -1, self.observation_size, len(frames))

    @property
    def n_parameters(self):
        return self.stateDim**2 + self.stateDim + 2 * self.observation_size + self.N * self.K

    def unpackParams(self, params):
        values = np.asarray(params, dtype=float)
        n, m = self.stateDim, self.observation_size
        if (values.ndim != 1 or len(values) < self.n_parameters
                or not np.isfinite(values).all()):
            raise ValueError(f"Provide at least {self.n_parameters} finite parameters.")
        A = values[:n*n].reshape(n, n)
        log_q = values[n*n:n*n+n]
        loading = values[n*n+n:n*n+n+m]
        log_r = values[n*n+n+m:n*n+n+2*m]
        if np.any(np.abs(np.r_[log_q, log_r]) > 100):
            raise ValueError("Log variances are outside the numerical range.")
        Q, R = np.diag(np.exp(log_q)), np.diag(np.exp(log_r))
        C = np.zeros((m, n))
        C[np.arange(m), np.repeat(np.arange(n), self.obsDim)] = loading
        offset = n*n + n + 2*m
        B = values[offset:offset + self.N*self.K].reshape(self.K, self.N).T if self.K else None
        legacy_variance = values[self.n_parameters] if self.K and len(values) > self.n_parameters else None
        return [A, Q, C, R, B, legacy_variance]

    def _step(self, model, mean, covariance, observation, exogenous, *, transition=True):
        A, Q, C, R = model
        if transition:
            mean = A @ mean
            covariance = A @ covariance @ A.T + Q
        keep = ~np.isnan(observation)
        if not keep.any():
            return mean, covariance, 0.
        design = C[keep]
        noise = R[np.ix_(keep, keep)]
        innovation = observation[keep] - (C @ mean + exogenous)[keep]
        cross_cov = covariance @ design.T
        innovation_cov = design @ cross_cov + noise
        factor = cho_factor(innovation_cov, lower=True)
        gain = cho_solve(factor, cross_cov.T).T
        mean = mean + gain @ innovation
        update = np.eye(self.stateDim) - gain @ design
        covariance = update @ covariance @ update.T + gain @ noise @ gain.T
        covariance = (covariance + covariance.T) / 2
        log_likelihood = -0.5 * (
            keep.sum() * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum()
            + innovation @ cho_solve(factor, innovation)
        )
        return mean, covariance, float(log_likelihood)

    def incrementKF(self, i, params, post_exp, post_var, new_exog, new_obs):
        A, Q, C, R, B, _ = self.unpackParams(params)
        adjusted = 0. if B is None else np.asarray(new_exog) @ B[i]
        return list(self._step((A, Q, C, R), post_exp, post_var, np.asarray(new_obs), adjusted))

    def _unit(self, model, B, unit, initial_mean, initial_covariance):
        means = np.empty((self.T, self.stateDim))
        covariances = np.empty((self.T, self.stateDim, self.stateDim))
        log_likelihood = np.empty(self.T)
        offsets = np.zeros(self.T) if B is None else self._exogenous[unit] @ B[unit]
        mean, covariance = initial_mean, initial_covariance
        for t, observation in enumerate(self._observations[unit]):
            mean, covariance, contribution = self._step(
                model, mean, covariance, observation, offsets[t], transition=t > 0
            )
            means[t], covariances[t], log_likelihood[t] = mean, covariance, contribution
        return means, covariances, log_likelihood

    def indivKF(self, params, init_exp, init_var, i):
        A, Q, C, R, B, _ = self.unpackParams(params)
        means, _, likelihood = self._unit((A, Q, C, R), B, i, init_exp, init_var)
        return [float(likelihood.sum()), means[-1]]

    def filter(self, params=None, *, initial_mean=None, initial_covariance=None):
        """Return state estimates, covariances and every Gaussian likelihood increment."""
        params = self.params_ if params is None else params
        A, Q, C, R, B, _ = self.unpackParams(params)
        mean = self.init_exp if initial_mean is None else np.asarray(initial_mean, dtype=float)
        covariance = self.init_var if initial_covariance is None else np.asarray(
            initial_covariance, dtype=float
        )
        n = self.stateDim
        if (mean.shape != (n,) or covariance.shape != (n, n)
                or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance).min() < -1e-12):
            raise ValueError("Provide a finite initial mean and positive semidefinite covariance.")
        units = [self._unit((A, Q, C, R), B, unit, mean, covariance) for unit in range(self.N)]
        return FilterResult(*[np.stack(items) for items in zip(*units)])

    def smooth(self, params=None):
        """Rauch–Tung–Striebel smoothed posterior state means and covariances."""
        params = self.params_ if params is None else params
        filtered = self.filter(params)
        A, Q, *_ = self.unpackParams(params)
        means = filtered.means.copy()
        covariances = filtered.covariances.copy()
        for unit in range(self.N):
            for t in range(self.T - 2, -1, -1):
                filtered_cov = filtered.covariances[unit, t]
                cross = filtered_cov @ A.T
                prior = A @ cross + Q
                gain = cho_solve(cho_factor(prior), cross.T).T
                means[unit, t] += gain @ (means[unit, t + 1] - A @ filtered.means[unit, t])
                posterior = filtered_cov + gain @ (covariances[unit, t + 1] - prior) @ gain.T
                covariances[unit, t] = (posterior + posterior.T) / 2
        return FilterResult(means, covariances, filtered.log_likelihood)

    def sampleKF(self, params, init_exp, init_var):
        result = self.filter(params, initial_mean=init_exp, initial_covariance=init_var)
        return [-float(result.log_likelihood.sum()) / self.N, list(result.means[:, -1])]

    def wrapLoglike(self, params):
        return self.sampleKF(params, self.init_exp, self.init_var)[0]

    def fit(self, params_init=None, *, method="L-BFGS-B", options=None):
        """Fit by Gaussian maximum likelihood and retain the SciPy result."""
        initial = self.paramInit() if params_init is None else np.asarray(params_init, dtype=float)
        self.unpackParams(initial)

        def objective(params):
            try:
                with np.errstate(over="raise", invalid="raise"):
                    return self.wrapLoglike(params)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e100

        result = minimize(objective, initial[:self.n_parameters], method=method, options=options)
        filtered = self.filter(result.x)
        self.optimization_result_ = result
        self.params_ = result.x.copy()
        self.post_exp = list(filtered.means[:, -1])
        return self

    def wrapMinimize(self, params_init, algo):
        self.fit(params_init, method=algo)
        return self.params_.copy(), self.post_exp

    def paramInit(self):
        n, m = self.stateDim, self.observation_size
        return np.r_[(0.8 * np.eye(n)).ravel(), np.full(n, -2.),
                     np.ones(m), np.zeros(m), np.zeros(self.N * self.K)]

    def incrementForecast(self, i, params, new_exog, t, statevar):
        A, _, C, _, B, _ = self.unpackParams(params)
        offset = 0. if B is None or new_exog is None else np.asarray(new_exog) @ B[i]
        return C @ np.linalg.matrix_power(A, t) @ np.asarray(statevar) + offset

    def forecastKalman(self, params, statevar, newexog=None, predRange=5):
        states = np.asarray(statevar, dtype=float)
        if states.shape != (self.N, self.stateDim):
            raise ValueError("Provide the last filtered state for every entity.")
        if newexog is not None:
            newexog = tuple(newexog)
            if len(newexog) != self.K or not newexog:
                raise ValueError("Future exog must have the original number of regressor tables.")
            columns = newexog[0].columns
            if len(columns) == 0 or len(columns) % self.observation_size:
                raise ValueError("Future exog must cover whole observation periods.")
            offsets = self._exog_array(newexog, columns)
            predRange = offsets.shape[1]
        else:
            offsets = None
        if not isinstance(predRange, Integral) or predRange < 1:
            raise ValueError("predRange must be a positive integer.")
        A, _, C, _, B, _ = self.unpackParams(params)
        predictions = []
        for t in range(predRange):
            states = states @ A.T
            forecast = states @ C.T
            if offsets is not None:
                forecast += np.einsum("nmk,nk->nm", offsets[:, t], B)
            predictions.append(forecast)
        return np.stack(predictions, axis=1).tolist()

    def indivForecast(self, params, i, statevar, newexog=None, predRange=5):
        return self.forecastKalman(params, statevar, newexog, predRange)[i]

    def wrapForecast(self, params, statevar, newexog=None, predRange=5):
        values = self.forecastKalman(params, statevar, newexog, predRange)
        columns = None if newexog is None else newexog[0].columns
        return pd.DataFrame(np.asarray(values).reshape(self.N, -1),
                            index=self.endog.index, columns=columns)

    def forecast(self, steps=5, *, exog=None):
        """Forecast from the last fitted states. Omitted regressors are set to zero."""
        return self.wrapForecast(self.params_, self.post_exp, newexog=exog, predRange=steps)
