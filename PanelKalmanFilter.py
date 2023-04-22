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
    def __init__(self, stateDim, obsDim, endog, exog = None):
        self.stateDim = stateDim
        self.obsDim = obsDim
        self.endog = endog
        self.exog = exog
        self.post_exp = []
        #self.K = len(data.columns.levels[0]) - 1
        if exog!= None:
            self.K = len(exog)
        else:
            self.K = 0
        self.N = len(endog.iloc[:,0])
        #self.T = int(len(data.columns.levels[1])/(obsDim*stateDim))
        self.T = int(len(endog.iloc[0,:])/(obsDim*stateDim))
        self.init_exp = np.zeros(stateDim)
        self.init_var = np.eye(stateDim)
        self.obsDict = []
        for i in range(0,stateDim*obsDim):
            self.obsDict.append(list(endog.columns[0+i*stateDim*obsDim:stateDim*obsDim*(i+1)]))
        self.defaultInitParams = [1.69271632, -7.53703822, -4.23337738, 7.737883 ,-0.34051513, -0.9349601,
                                -1.81282109,-2.00712979,-1.87171474,5.94652967,6.35815269,4.86795731,
                                -1.58287851,-1.23747189,-1.41927504,-2.26545219,-1.5234699,2.08084558,
                                -3.76331173,4.46234078,-2.33860228,1.35042423,3.35580212,2.1897442,
                                -1.81117239, -1.83429089, 3.54230905, 4.98408931, -6.51527362, 2.03645846,
                                2.32582213, 0.85892667, -3.80231989, 4.16718004]

    """The following function is a helper function meant to work with other more complex functions. It reshapes a list of initial parameters into their proper forms.
    - `A`: diagonal matrix of dimension `stateDim`$\times$`stateDim`. This matrix holds parameters for the state equation.
    - `V`: diagonal matrix of dimension `stateDim`$\times$`stateDim` using initialized parameters indexed `stateDim`$^{2}$:(`stateDim`$^{2}+$`stateDim`). Holds the independent error terms for the state equation.
    - `C`: diagonal matrix of dimension `stateDim*obsDim`$\times$`stateDim`. Holds parameters for the state variable in the measurement equation.
    - `W`: diagonal matrix of dimension `stateDim*obsDim`$\times$`stateDim*obsDim`. Holds the independent error terms for the measurement equation.
    - `B`: matrix for exogenous variable parameters. It is $N \times K$, with a row for each group, and a column for each independent variable.
    - `Sigma2MLE`: initialized variance for exogenous parameters.
    """
    def unpackParams(self, params):
        place = 0
        A = np.reshape(params[place:(place + self.stateDim**2)], (self.stateDim, self.stateDim))
        place += self.stateDim**2
        V = np.diag(np.exp(params[(place):(place+self.stateDim)]))
        place += self.stateDim
        C = np.zeros(shape = (self.stateDim*self.obsDim,self.stateDim))
        for j in range(0,self.stateDim):
            C[(self.obsDim*(j)):self.obsDim*(j+1),j] = params[place+self.obsDim*(j):(place+(self.obsDim)*(j+1))]
        place += self.obsDim*self.stateDim
        W = np.diag(np.exp(params[(place):(place+self.obsDim*self.stateDim)]))
        place += self.obsDim*self.stateDim

        if self.exog == None:
            B = None
            sigma2MLE = None
        else:
            B = np.zeros(shape = (self.N, self.K))
            for j in range(0, self.K):
                B[:,j] = params[place+(self.N)*(j):place+(self.N)*(j+1)]
            place += self.N*self.K
            sigma2MLE = params[place]

        return [A,V,C,W,B, sigma2MLE]

    """
    This function estimates the Kalman Filter on time increments, using observations per time. 
    This code also calculates likelihood for the linear model $y = Az + Bx + w$, 
    where $w$ is the error term, $x$ is a vector of exogenous variables, and $z$ is the state variable.
    """

    def incrementKF(self, i, params,post_exp,post_var, new_exog, new_obs):
        """ unpack parameters """
        unpacked = self.unpackParams(params)
        A = unpacked[0]
        V = unpacked[1]
        C = unpacked[2]
        W = unpacked[3]
        
        #postexpHold = self.post_exp

        """ predict step"""
        prior_exp = A@post_exp
        prior_var = A@post_var@np.transpose(A) + V
        obs_prior_exp = C@prior_exp
        obs_prior_var = C@prior_var@np.transpose(C) + W
        if self.exog is None:
            sum_exog = 0
        else:
            B = unpacked[4][i,:]
            sigma2MLE = unpacked[5]
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
        if self.exog != None:
            exodist = stats.multivariate_normal(mean = 0, cov = sigma2MLE)
            log_like = dist.logpdf(new_obs) + exodist.logpdf(residual)
        else:
            log_like = dist.logpdf(new_obs)
        
        return [post_exp,post_var,log_like]#, post_exog]

    """
    This function estimates a Kalman filter for each unit - sectors, regions, countries, groups, etc.
    """

    def indivKF(self, params,init_exp,init_var,i):
        iData = endog.iloc[i,:]
        if self.exog == None:
            iExog = None
            init_obs_exog = 0
        else:
            iExog = []
            for t in range(0,self.K):
                iExog.append(self.exog[t].iloc[i,:])
            iExog = pd.concat(iExog, axis = 1).transpose()
            init_obs_exog = np.array(iExog[self.obsDict[0]])

        """ initialization """
        post_exp = init_exp
        post_var = init_var

        init_obs = np.array(iData.loc[self.obsDict[0]])
        dist = stats.multivariate_normal(mean = init_obs, cov = 1)
        log_like = dist.logpdf(init_obs)

        for t in range(0,(self.T-1)):
            """ predict and update """
            new_obs = np.transpose(np.array(iData[self.obsDict[t+1]]))
            if self.exog == None:
                new_exog = 0
            else:
                new_exog = np.transpose(np.array(iExog[self.obsDict[t+1]]))

            new_post = self.incrementKF(i, params,post_exp,post_var, new_exog, new_obs)
            """ replace """
            post_exp = new_post[0]
            post_var = new_post[1]

        """ log-likelihood contributions """
        log_like += new_post[2]
        return [np.sum(log_like), post_exp]

    """
    This function goes through each unit to sum their log-likelihood contributions over time.
    """

    def sampleKF(self, params,init_exp,init_var):
        post_exp = []
        log_like = 0.0
        for i in range(0,self.N):
            indivKFresults = self.indivKF(params,init_exp,init_var,i)
            log_like += indivKFresults[0]
            post_exp.append(indivKFresults[1])
        neg_avg_log_like = -log_like/self.N
        print("Current average negative log-likelihood: ",neg_avg_log_like)
        return [neg_avg_log_like, post_exp]

    """
    This is a wrapper function for use with our optimization algorithm.
    """

    def wrapLoglike(self, params):
        #print("current parameters:{}".format(params))
        sampler = self.sampleKF(params,self.init_exp,self.init_var)
        self.post_exp = post_exp = sampler[1]
        return sampler[0]

    """
    We use the minimize function from the scipy package to perform Maximum Likelihood Estimation. 
    This wrapper function allows convenience. 
    It returns our parameter estimates and our state variable, for use in forecasting.
    """

    def wrapMinimize(self, params_init, algo):
        MLE = minimize(fun = self.wrapLoglike, x0 = params_init, method = algo)
        return(MLE.x, self.post_exp)

    """This function forecasts the endogenous variable for time increments."""

    def incrementForecast(self, i, params, new_exog, t, statevar):
        """ unpack parameters """
        unpacked = self.unpackParams(params)
        A = unpacked[0]
        V = unpacked[1]
        C = unpacked[2]
        #W = unpacked[3]
        if type(new_exog) == list and new_exog.any() != None:
            B = unpacked[4][i,:]
            exogMultiply = new_exog@B
        else:
            exogMultiply = 0
            
            
        #sigma2MLE = unpacked[5]

        """ predict step"""
        state_exp = np.linalg.matrix_power(A, t)@statevar
 
        """ forecast step """
        forecast_obs = exogMultiply + C@state_exp

        return forecast_obs

    """This function forecasts the endogenous variable per individual."""

    def indivForecast(self, params, i, statevar, newexog, predRange):
        if newexog == None:
            iExog = None
        else:
            iExog = []
            for t in range(0, self.K):
                iExog.append(newexog[t].iloc[i, :])
            iExog = pd.concat(iExog, axis=1).transpose()

            obsDict = []
            for t in range(0,int(len(iExog.columns)/(self.stateDim*self.obsDim))):
                obsDict.append(list(iExog.columns[0+t*self.stateDim*self.obsDim:self.stateDim*self.obsDim*(t+1)]))

        new_obs = []

        if newexog == None:
            predRange = predRange
        else:
            predRange = len(obsDict)

        for t in range(0,predRange):
            """ predict and update """
            #reverse = list(reversed(range(len(obsDict))))
            if newexog != None:
                new_exog = np.transpose(np.array(iExog[obsDict[t]]))
            else:
                new_exog = None

            forecast_endog = self.incrementForecast(i, params, new_exog, (t + 1), statevar[i])
            new_obs.append(forecast_endog)
        #new_obs = pd.data
        return new_obs

    """This function is a wrapper function for forecasting."""

    def forecastKalman(self, params, statevar, newexog, predRange):
        forecast = []
        for i in range(0,self.N):
            indiv_forecast = self.indivForecast(params,i, statevar, newexog, predRange)
            forecast.append(indiv_forecast)
        return forecast

    def wrapForecast(self, params, statevar, newexog = None, predRange = 5):
        forecast = self.forecastKalman(params, statevar, newexog, predRange)
        forecast_predict = []
        for t in range(len(forecast)):
            forecast_predict.append(pd.DataFrame(np.concatenate(forecast[t])).transpose())
        forecast_dataframe = pd.concat(forecast_predict)
        return(forecast_dataframe)

    def paramInit(self):
        # Define the number of parameters depending on whether exogenous data is provided
        if self.exog is None:
            num_params = 1 + self.stateDim**2 + self.stateDim + self.stateDim*self.obsDim + self.obsDim*self.stateDim + self.stateDim*self.obsDim
        else:
            num_params = 1 + self.stateDim**2 + self.stateDim + self.stateDim*self.obsDim + self.obsDim*self.stateDim + self.stateDim*self.obsDim + self.N*self.K

        # Define an initial guess for the parameters
        initial_guess = np.random.rand(num_params)
        return initial_guess
