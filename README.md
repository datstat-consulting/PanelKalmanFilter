# PanelKalmanFilter

For linear dynamic processes with noisy measurements, one can posit the existence of a state variable affecting it. In this state space model, a process enters different states as it evolves. The Kalman Filter seeks to estimate this state variable using expectation maximization. Estimation can incorporate exgoenous, or control, variables other than the estimated state variable. The Kalman Filter is widely used in different fields, including Economics, Engineering, Finance, and Physics.

`PanelKalmanFilter` is a Python package meant for use with panel data. This package grew from work performed for estimating Macroeconomic Credit Risk. We used Python since it allowed ease in implementing scientific computation and algorithms. As such, the Kalman Filter can now be applied easily by more end-users in different fields making use of panel or longitudinal datasets. `PanelKalmanFilter` relies heavily on `numpy` and `scipy` to handle matrix algebra and optimization. The expectation maximization algorithm was implemented in a prediction step and an update step, with Maximum Likelihood Estimation used to determine optimal coefficients.
  
While `PanelKalmanFilter` was developed for Financial data, it can be used for a wide range of fields. In particular, we can see longitudinal and panel data in Economics and Biostatistics receive extensive treatment with Kalman Filtering with our package. 

## Installation
You can install the package using pip:
```
pip install git+https://github.com/datstat-consulting/PanelKalmanFilter
```

# Example

We load a dummy dataset to use. This dataset was generated in R with the following code:

`library(prodest)`
`set.seed(123)`
`dummy_panel <- panelSim(N = 3, T = 360, alphaL = .6, alphaK = .4, DGP = 1,
                        rho = .7, sigeps = .1, sigomg = .3, rholnw = .3)`
```
df = pd.read_csv('dummy_training.csv')
df = df.iloc[:,1:9]
```
We reshape the dataset from a long panel format to a wide panel format.
```
data = df.pivot(index='idvar', columns='timevar')
```
We set our endogenous and exogenous variables to use.
```
endog = data['Y']
exog = [data['sX'] ,data['fX'], data['pX1'], data['pX2'], data['pX3']]
```
Set random seed for reproducibility.
```
#set your own seed
np.random.seed()
```
Import the class, then set an instance. Note that you can leave the `exog` argument empty to estimate a Kalman Filter only for the endogenous variable.
```
from PanelKalmanFilter import all
pkf = PanelKalmanFilter(stateDim = 2, obsDim = 3, endog = endog, exog = exog)
```
Solve the minimization problem with BFGS.
```
params_init = pkf.paramInit() #using random initial weights needs more experimentation
MLE = wrapMinimize(params_init = params_init, algo = 'BFGS') #needs experimentation with Gradient Descent for random initial weights
```
Load the test dummy dataset and perform the forecast. For practical applications, you may need to forecast the exogenous variables first.
```
testing = pd.read_csv('dummy_testing.csv')
testing = testing.iloc[:,1:9]
testdata = testing.pivot(index='idvar', columns='timevar')
exog_test = [testdata['sX'] ,testdata['fX'], testdata['pX1'], testdata['pX2'], testdata['pX3']]

forecast = pkf.wrapForecast(MLE[0], MLE[1], exog_test)
forecast
```
# References
* Hamilton, J. D. (1994). Time series analysis. Princeton university press.
* Hu, B., Qiu, W., Xu, C., & Wang, J. (2020). Integration of a Kalman filter in the geographically weighted regression for modeling the transmission of hand, foot and mouth disease. BMC public health, 20, 1-15.
* Kerbl, S., & Sigmund, M. (2011). What drives aggregate credit risk?. Oesterreichische Nationalbank Financial Stability Report, 22.
