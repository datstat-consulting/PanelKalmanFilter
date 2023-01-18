import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

class PanelKalmanFilter():
    """Here, we initialize our dimensions and initial parameters. They are as follows:
    - `stateDim`: number of states
    - `obsDim`: number of measures per state
    - `K`: number of independent variables
    - `N`: number of groups
    - `T`: number of time periods
    - `init_exp`: our initial expected value
    - `init_var`: our initial variance
    - `params_init`: our parameter initialization
    One will have to edit the parameters for different configurations.
    """
    def __init__(self, endog, exog, stateDim, obsDim):
        self.stateDim = stateDim
        self.obsDim = obsDim
        self.post_exp = []
        #self.K = len(data.columns.levels[0]) - 1
        self.K = len(exog)
        self.N = len(endog.iloc[:,0])
        #self.T = int(len(data.columns.levels[1])/(obsDim*stateDim))
        self.T = int(len(exog[0].iloc[0,:])/(obsDim*stateDim))
        self.init_exp = np.zeros(stateDim)
        self.init_var = np.eye(stateDim)
        self.obsDict = []
        for i in range(0,stateDim*obsDim):
            self.obsDict.append(list(endog.columns[0+i*stateDim*obsDim:stateDim*obsDim*(i+1)]))

    """The following function is a helper function meant to work with other more complex functions. It reshapes a list of initial parameters into their proper forms.
    - `A`: diagonal matrix of dimension `stateDim`$\times$`stateDim`. This matrix holds parameters for the state equation.
    - `V`: diagonal matrix of dimension `stateDim`$\times$`stateDim` using initialized parameters indexed `stateDim`$^{2}$:(`stateDim`$^{2}+$`stateDim`). Holds the independent error terms for the state equation.
    - `C`: diagonal matrix of dimension `stateDim*obsDim`$\times$`stateDim`. Holds parameters for the state variable in the measurement equation.
    - `W`: diagonal matrix of dimension `stateDim*obsDim`$\times$`stateDim*obsDim`. Holds the independent error terms for the measurement equation.
    - `B`: matrix for exogenous variable parameters. It is $N \times K$, with a row for each group, and a column for each independent variable.
    - `Sigma2MLE`: initialized variance for exogenous parameters.
    """
    def unpackParams(params):
        place = 0
        A = np.reshape(params[place:(place + self.stateDim**2)], (self.stateDim, self.stateDim))
        place += stateDim**2
        V = np.diag(np.exp(params[(place):(place+self.stateDim)]))
        place += stateDim
        C = np.zeros(shape = (self.stateDim*self.obsDim,self.stateDim))
        for j in range(0,self.stateDim):
            C[(self.obsDim*(j)):self.obsDim*(j+1),j] = params[place+self.obsDim*(j):(place+(self.obsDim)*(j+1))]
        place += self.obsDim*self.stateDim
        W = np.diag(np.exp(params[(place):(place+self.obsDim*self.stateDim)]))
        place += self.obsDim*self.stateDim
        B = np.zeros(shape = (self.N,self.K))
        for j in range(0, self.K):
            B[:,j] = params[place+(self.N)*(j):place+(self.N)*(j+1)]
        place += self.N*self.K
        sigma2MLE = params[place]
        return [A,V,C,W,B, sigma2MLE]


#stateDim = 2
#obsDim = 3

#params_init = [1.71980540,-7.84056667,-4.17814607,7.18531777,-1.30677476,
#-1.18088726,-1.66055959,-1.57007816,-1.55171518,5.83622406,6.27800947,4.49955609,
#-1.51563347,-1.59148527,-1.44753769,-1.84256431,-1.69543580,1.97653226,-3.36927402,5.30981140,
#-2.57127719,1.79427074,3.76140164,2.04323101,-1.35583690,-1.30289481,3.46299721,5.46711510,
#-5.99897917,2.01711379,2.65831208,1.38660559,-3.92584823,3.47669802]

    """This function estimates the Kalman Filter on time increments, using observations per time. This code also calculates likelihood for the linear model $y = Az + Bx + w$, where $w$ is the error term, $x$ is a vector of exogenous variables, and $z$ is the state variable."""

    def incrementKF(i, params,post_exp,post_var, new_exog, new_obs):
        """ unpack parameters """
        unpacked = unpackParams(params)
        A = unpacked[0]
        V = unpacked[1]
        C = unpacked[2]
        W = unpacked[3]
        B = unpacked[4][i,:]
        sigma2MLE = unpacked[5]
        postexpHold = postexpHold()

        """ predict step"""
        prior_exp = A@post_exp
        prior_var = A@post_var@np.transpose(A) + V
        obs_prior_exp = C@prior_exp
        obs_prior_var = C@prior_var@np.transpose(C) + W
        sum_exog = new_exog@B

        """ update step """
        residual = new_obs - obs_prior_exp - sum_exog
        obs_prior_cov = prior_var@np.transpose(C)
        kalman_gain = obs_prior_cov@ np.linalg.inv(obs_prior_var)
        post_exp = prior_exp + kalman_gain@residual
        post_var = prior_var - kalman_gain@np.transpose(obs_prior_cov)

        """ step likelihood """
        dist = stats.multivariate_normal(mean = np.reshape(obs_prior_exp,(len(obs_prior_exp),)),cov = obs_prior_var)
        #dist = stats.multivariate_normal(mean = post_exp, cov = post_var)
        exodist = stats.multivariate_normal(mean = 0, cov = sigma2MLE)
        log_like = dist.logpdf(new_obs) + exodist.logpdf(residual)
        return [post_exp,post_var,log_like]#, post_exog]

    """This function estimates a Kalman filter for each unit - sectors, regions, countries, groups, etc."""

    def indivKF(params,init_exp,init_var,i):
        iData = endog.iloc[i,:]
        iExog = []
        for t in range(0,K):
            iExog.append(exog[t].iloc[i,:])
        iExog = pd.concat(iExog, axis = 1).transpose()
        """ initialization """
        post_exp = init_exp
        post_var = init_var

        init_obs = np.array(iData.loc[obsDict[0]])
        init_obs_exog = np.array(iExog[obsDict[0]])
        dist = stats.multivariate_normal(mean = init_obs, cov = 1)
        log_like = dist.logpdf(init_obs)

        for t in range(0,(T-1)):
            """ predict and update """
            new_obs = np.transpose(np.array(iData[obsDict[t+1]]))
            new_exog = np.transpose(np.array(iExog[obsDict[t+1]]))
            new_post = incrementKF(i, params,post_exp,post_var, new_exog, new_obs)
            """ replace """
            post_exp = new_post[0]
            post_var = new_post[1]

        """ log-likelihood contributions """
        log_like += new_post[2]
        return [np.sum(log_like), post_exp]

    """This function goes through each unit to sum their log-likelihood contributions over time."""

    def sampleKF(params,init_exp,init_var):
        post_exp = []
        log_like = 0.0
        for i in range(0,N):
            indivKFresults = indivKF(params,init_exp,init_var,i)
            log_like += indivKFresults[0]
            post_exp.append(indivKFresults[1])
        neg_avg_log_like = -log_like/N
        print("Current average negative log-likelihood: ",neg_avg_log_like)
        return [neg_avg_log_like, post_exp]

    """This is a wrapper function for use with our optimization algorithm."""

    def wrapLoglike(params):
        #print("current parameters:{}".format(params))
        sampler = sampleKF(params,init_exp,init_var)
        self.post_exp = post_exp = sampler[1]
        return sampler[0]

    """We use the minimize function from the scipy package to perform Maximum Likelihood Estimation. This wrapper function allows convenience. It returns our parameter estimates and our state variable, for use in forecasting."""

    def wrapMinimize(params_init, algo):
        MLE = minimize(fun = wrapLoglike, x0 = params_init, method = algo)
        return(MLE.x, self.post_exp)

    """This function forecasts the endogenous variable for time increments."""

    def incrementForecast(i, params, new_exog, t, statevar):
        """ unpack parameters """
        unpacked = unpackParams(params)
        A = unpacked[0]
        V = unpacked[1]
        C = unpacked[2]
        #W = unpacked[3]
        B = unpacked[4][i,:]
        #sigma2MLE = unpacked[5]

        """ predict step"""
        state_exp = np.linalg.matrix_power(A, t)@statevar
 
        """ forecast step """
        forecast_obs = new_exog@B + C@state_exp

        return forecast_obs

    """This function forecasts the endogenous variable per individual."""

    def indivForecast(params,i, statevar):
        iExog = []
        for t in range(0,K):
            iExog.append(exog_test[t].iloc[i,:])
        iExog = pd.concat(iExog, axis = 1).transpose()
        obsDict = []
        for t in range(0,int(len(iExog.columns)/(stateDim*obsDim))):
            obsDict.append(list(iExog.columns[0+t*stateDim*obsDim:stateDim*obsDim*(t+1)]))

        new_obs = []

        for t in range(0,len(obsDict)):
            """ predict and update """
            #reverse = list(reversed(range(len(obsDict))))
            new_exog = np.transpose(np.array(iExog[obsDict[t]]))
            forecast_endog = incrementForecast(i, params, new_exog, (t + 1), statevar[i])
            new_obs.append(forecast_endog)
        #new_obs = pd.data
        return new_obs

    """This function is a wrapper function for forecasting."""

    def forecastKalman(params, statevar):
        forecast = []
        for i in range(0,N):
            indiv_forecast = indivForecast(params,i, statevar)
            forecast.append(indiv_forecast)
        return forecast

    def wrapForecast(params, statevar):
        forecast = forecastKalman(params, statevar)
        forecast_predict = []
        for t in range(len(forecast)):
            forecast_predict.append(pd.DataFrame(np.concatenate(forecast[t])).transpose())
        forecast_dataframe = pd.concat(forecast_predict)
        return(forecast_dataframe)
