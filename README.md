# PanelKalmanFilter

For linear dynamic processes with noisy measurements, one can posit the existence of a state variable affecting it. In this state space model, a process enters different states as it evolves. The Kalman Filter seeks to estimate this state variable using expectation maximization. Estimation can incorporate exgoenous, or control, variables other than the estimated state variable. The Kalman Filter is widely used in different fields, including Economics, Engineering, Finance, and Physics.

`PanelKalmanFilter` is a Python package meant for use with panel data. This package grew from work performed for analysis of Financial data. We used Python since it allowed ease in implementing scientific computation and algorithms. As such, the Kalman Filter can now be applied easily by more end-users in different fields making use of panel or longitudinal datasets. `PanelKalmanFilter` relies heavily on `numpy` and `scipy` to handle matrix algebra and optimization. The expectation maximization algorithm was implemented in a prediction step and an update step, with Maximum Likelihood Estimation used to determine optimal coefficients.
  
While `PanelKalmanFilter` was developed for Financial data, it can be used for a wide range of fields. In particular, we can see longitudinal and panel data in Economics and Biostatistics receive extensive treatment with Kalman Filtering with our package. 

# Example



# References
