import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy import stats, optimize
from scipy import linalg as scilin
from itertools import product
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import sympy as sym
import logging


# We define some functions
x, y, rho, c, delta, phi, psi_sym = sym.symbols('x y rho c delta phi psi_sym')
theta, pi = sym.symbols('theta pi')

# We define the link functions.
psi_sym = phi / sym.sqrt(c + (1 + rho)) + (( 1 - phi**2) / 2 - (1 - phi**2) * theta)
a_func = rho * x / ( 1 + c * x)
alpha = psi_sym * x + (( 1 -phi**2) / 2) * x**2
b_func = delta * sym.log(1 + c * x)
beta_sym = a_func.replace(x, pi + alpha.replace(x, theta - 1)) - a_func.replace(x, pi + alpha.replace(x, theta)) 
gamma_sym = b_func.replace(x, pi + alpha.replace(x, theta-1)) - b_func.replace(x, pi + alpha.replace(x, theta))


# We create the link functions.
gamma = sym.lambdify((rho, c, delta, phi, pi, theta), gamma_sym)
beta = sym.lambdify((rho, c, phi, pi, theta), beta_sym)
psi = sym.lambdify((rho, c, phi,  theta), psi_sym)

# We define some moments.
mean = rho * x + c * delta
var = 2 * c * rho * x + c**2 * delta
mom1 = y - mean
mom2 = (y- mean) * x
mom3 = (y**2 - (var + mean**2))
mom4 = (y**2 - (var + mean**2)) * x
mom5 = (y**2 - (var + mean**2)) * x**2

# We collect those moments into a function.
vol_moments_sym = sym.Matrix([mom1,mom2, mom3, mom4, mom5])
vol_moments_lambda = sym.lambdify((x, y, c, rho, delta), vol_moments_sym)

# Setup the link function.
second_stage_moments_sym = sym.Matrix([beta_sym, c, delta, gamma_sym, phi**2, psi_sym, rho])
second_stage_moments_sym.simplify()
second_stage_moments = sym.lambdify((c, delta, phi, pi, rho, theta), second_stage_moments_sym)

# Define the gradients of the link function.

link1_grad_in = sym.lambdify((c, delta, phi, pi,rho, theta),  second_stage_moments_sym.jacobian([c,delta, rho]))


link2_grad_in = sym.lambdify((rho, c, delta, phi, pi, theta), second_stage_moments_sym.jacobian([rho, c, delta,
                                                                                                 phi, pi, theta]))

def simulate_autoregressive_gamma(rho=0, scale=1, delta=1, initial_point=None, time_dim=100,
                                  state_date='2000-01-01'):
    """
    This function provides draws from the ARG(1) process of Gourieroux & Jaiak
    
    Parameters
    --------
    rho : scalar
        AR(1) coefficient
    delta : scalar
        intercept
    scale : scalar
    Returns

    -----
    draws : dataframe
    """
    
    # If initial_point is not specified, we start at the unconditional mean.
    
    initial_point = (scale * delta) / (1 - rho)
    
    # The conditional distribution of an ARG(1) process is non-centered Gamma, which has a representation as a 
    # Poisson mixture of Gamma
    
    draws = [initial_point]
    
    for _ in tqdm_notebook(range(time_dim)):
        
        latent_var = stats.poisson.rvs(mu = rho * draws[-1] / scale)
        draws.append(stats.gamma.rvs(a=delta+latent_var, scale=scale))
    
    draws = pd.DataFrame(draws[1:], pd.date_range(start=state_date, freq='D', periods=time_dim))
    return draws


def simulate_conditional_gaussian(vol_data, rho=0, scale=.1, delta=1, phi=0, vol_price=0, equity_price=1):
    """
    This function simulates conditional Gaussian random variables with mean
    
    $$E[r_{t+1} | \sigma^2_t, \sigma^2_{t+1}] = \psi \sigma^2_{t+1} + \beta \sigma^2_t + \gamma$$
    $$Var[r_{+t} | \sigma^2_t, \sigma^2_{t+1}] = (1 - \phi^2) \sigma^2_{t+1}$$
    
    Parameters
    ----------
    vol_data : pandas dataframe
        The volatility data. It must always be positive.
    rho : scalar
    scale : scalar
    delta : scalar
    phi : scalar 
        It must be in [-1,1]
    vol_price : scalar
    equity_price : scalar
    
    Returns
    -------
    data : pandas dataframe 
        This contains both the vol_data and the return data
    """
    
    gamma_val = gamma(rho=rho, c=scale, delta=delta, phi=phi, pi=vol_price, theta=equity_price)
    beta_val = beta(rho=rho, c=scale, phi=phi, pi=vol_price, theta=equity_price)
    psi_val = psi(rho=rho, c=scale, phi=phi, theta=equity_price)
    
    mean = gamma_val + beta_val * vol_data.shift(1) + psi_val * vol_data
    var = (1 - phi**2) * vol_data
    
    draws =  mean + pd.DataFrame(stats.norm.rvs(0, scale=var.apply(np.sqrt)), index=vol_data.index)
    data = pd.concat([vol_data, draws], axis=1).dropna()
    data.columns = ['vol', 'rtn']
    
    return data


def simulate_data(equity_price=1, vol_price=0, rho=0, scale=1, delta=1, phi=0, initial_point=None, time_dim=100,
                  state_date='2000-01-01'):
    """
    This function takes the reduced-form paramters and risk prices and returns the data
    
    Parameters
    --------
    equity_price: scalar
    vol_price : scalar
    phi : scalar
        leverage
    rho : scalar
        persistence
    scale : positive scalar
    initial_point: scalar, optional
        Starting value for the volatility
    time_dim : int, optional
        number of periods
    start_date : datelike, optional
        The time to start the data from.
        
    Returns
    -----
    draws : dataframe
    
    """
    vol_data = simulate_autoregressive_gamma(rho=rho, scale=scale, delta=delta, initial_point=initial_point,
                                             state_date=pd.to_datetime(state_date) - pd.Timedelta('1 day'),
                                             time_dim=time_dim + 1)
    data = simulate_conditional_gaussian(vol_data, rho=rho, scale=scale, delta=delta, phi=phi,
                                         vol_price=vol_price, equity_price=equity_price)

    return data


def vol_moments(vol_data, scale, rho,delta):    
    """ Computes the moments for the volatility. """ 

    return pd.DataFrame(np.squeeze(vol_moments_lambda(vol_data.values[1:], vol_data.values[:-1], c=scale, rho=rho,
                                                      delta=delta)).T)


def vol_moments_grad(vol_data, scale, delta, rho):
    """ Computes the jacobian of the volatility moments. """
    x = vol_data.values[:-1]
    y = vol_data.values[:-1]

    cond_mean = scale * delta * rho * x

    row1 = np.array([-np.full(x.shape, delta), -x, -np.full(x.shape,scale)])
    row2 = x * row1 
    row3 = np.array([-2 * scale * delta - 2 * rho * x - 2 * delta * cond_mean, -2 * scale * x - 2 * x * cond_mean, -
                     scale**2 - 2 * scale * cond_mean])
    row4 = x * row3
    row5 = x**2 * row3
    
    mom_grad_in  = np.row_stack([np.mean(row1, axis=1), np.mean(row2, axis=1), np.mean(row3, axis=1),
                                 np.mean(row4, axis=1), np.mean(row5, axis=1)])

    return mom_grad_in



def compute_init_constants(vol_data):
    """ 
    Computes some guesses for the volatlity paramters that we can use to initialize the optimization.
    
    From the model, we know that intercept = $ c * \delta$. We also know that the average error variance equals $
    c^2 \delta * (2 \rho / 1 - \rho) + 1)$. Consequently, $c = error\_var / ( intercept * (2 \rho / 1 - \rho) +
    1))$, and  $\delta = \text{intercept} / c$.
      
    Paramters
    -------
    vol_data : dataframe
        The volatility data
    Returns
    -----
    dict
    """
    intercept, persistence = tsa.AR(vol_data).fit(maxlag=1).params
    error_var = tsa.AR(vol_data).fit(maxlag=1).sigma2
    tsa.AR(vol_data).fit(maxlag=1).conf_int()
    init_constants = {'rho': persistence}

    init_constants['scale'] = error_var / ( intercept * ( 2 * persistence / ( 1- persistence) + 1) )
    init_constants['delta'] = intercept / init_constants['scale']

    return init_constants


def compute_mean_square(x, data, func, weight=None):
    """
    Applies func to the data at *x, and then computes its weighted mean square error.

    Paramters
    -------
    x : iterable 
        paramters
    data : dataframe 
    func : function
    weight : 2d ndarray

    Returns
    ------
    scalar
    """

    func_data = func(data, *x)
    
    if weight is None:
        root_weight = np.eye(len(func_data.T))
    else:
        root_weight = scilin.cholesky(weight)
    
    return np.asscalar(np.mean(np.ravel((func_data @  root_weight)**2)))


def compute_vol_gmm(vol_data, init_constants, bounds=None, options=None):
    """
    This function uses GMM to compute the volatility paramters and their asymptotic covariance matrix.
    
    Paramters
    ---------
    vol_data : pandas series
    init_constants : dict
        This must contain initial guesses for the paramters as values and their names as keys.
    bounds : list, optional
    options: dict, optional
    
    Returns
    --------
    final_result : dict
    cov : ndarray
    """
    if bounds is None:
        bounds = [(-1+1e-5,1-1e-5),(1e-5, 20), (1e-5,20)]

    if options is None:
        options = {'maxiter':200}
        
    x0 = list(init_constants.values())
    
    initial_result = optimize.minimize(lambda x: compute_mean_square(x, vol_data, vol_moments),
                                       x0=x0, method="SLSQP", bounds=bounds, options=options)
    
    weight_matrix = scilin.pinv(vol_moments(vol_data, *initial_result.x).cov())
    
    final_result = optimize.minimize(lambda x: compute_mean_square(x, vol_data, vol_moments, weight_matrix),
                                      x0=initial_result.x, method="SLSQP", bounds=bounds, options=options)
    estimates = {key:val for key,val in zip(init_constants.keys(), final_result.x)}

    moment_derivative = vol_moments_grad(vol_data, **estimates)
    cov = pd.DataFrame(np.linalg.pinv(moment_derivative.T @ weight_matrix @ moment_derivative),
                       columns=list(init_constants.keys()), index=list(init_constants.keys()))
    
    if not final_result.success:
        logging.warning("Convergence results are %s.\n", final_result)

    return estimates, cov / (vol_data.size -1)


def create_est_table(estimates, truth, cov, num_se=1.96):
    """
    This function creates a table that prints the estimates, truth, and confidence interval
    
    Paramters:
    --------
    names : list of str
        The values to print
    truth : dict
        The true values
    cov : dataframe
        Covariance matrix
    num_se : positive float, optional
        The number of standard errors to use.
    Returns
    -------
    dataframe
    """
    names = set(estimates.keys()).intersection(set(truth.keys()))
    true_values = [truth[name] for name in names]
    est_values = [estimates[name] for name in names]
    
    lower_ci = [estimates[name] - num_se * np.sqrt(cov.loc[name, name]) for name in names]
    upper_ci =[estimates[name] + num_se * np.sqrt(cov.loc[name, name]) for name in names]

    return_df = pd.DataFrame(np.column_stack([true_values, est_values, lower_ci, upper_ci]),
                             columns = ['truth', 'estimate', 'lower ci', 'upper ci'], index=names)
    return_df.sort_index(inplace=True)
    return_df['in_ci'] = ((return_df['truth'] >= return_df['lower ci']) 
                          & (return_df['truth'] <= return_df['upper ci']))
    
    return return_df


def cov_to_corr(cov):
    """ 
    Converts a covariance matrix to a correlation matrix.
    """
    corr = pd.DataFrame(np.atleast_2d(np.diag(cov))**(-.5) * cov.values / np.atleast_2d(np.diag(cov)).T**.5,
                        index=cov.index,columns=cov.columns)
    
    return corr


def compute_step2(data, parameter_mapping=None):
    
    if parameter_mapping is None:
        parameter_mapping = {'vol':'psi', 'vol.shift(1)':'beta', 'intercept':'gamma'}
    
    wls_results = sm.WLS.from_formula('rtn ~ 1+ vol.shift(1) + vol', weights=data.vol**(-1), data=data).fit()
    
    estimates = wls_results.params.rename(parameter_mapping)
    
    # We force phi^2 >=0
    estimates['phi_squared'] =  np.maximum(1 - np.mean(wls_results.wresid**2), 0)
    
    phi2_cov = pd.DataFrame(np.atleast_2d(np.cov(wls_results.wresid**2)), index=['phi_squared'],
                        columns=['phi_squared'])
    return_cov = wls_results.cov_params().rename(columns=parameter_mapping).rename(parameter_mapping)
    return_cov = return_cov.merge(phi2_cov, left_index=True, right_index=True, how='outer').fillna(0)
    
    return estimates, return_cov


def link_grad_reduced(delta, equity_price, phi, rho, scale, vol_price):
    """
    This function computes the jacobian of the link function with respect to the reduced form paramters
    beta, c, delta, gamma, phi^2, psi, rho
    
    Paramters
    ---------
    scale : scalar
    equity_price : scalar
    phi : scalar
    rho : scalar
    vol_price : scalar
    
    Returns
    --------
    ndarray
    """
    return_mat = np.zeros((7,7))
    mat2 = link1_grad_in(phi=phi, rho=rho, c=scale, delta=delta, theta=equity_price, pi=vol_price)
    return_mat[0,0] = 1
    return_mat[:,1] = mat2[:,0]
    return_mat[:,2] = mat2[:,1]
    return_mat[3,3] = 1
    return_mat[4,4] = 1
    return_mat[5,5] = 1
    return_mat[:,6] = mat2[:,2]
    
    return return_mat


def link_grad_structural(rho, scale, delta, phi, equity_price, vol_price):
    """
    This function computes the jacobian of the link function with respect to the structural paramters
    phi, rho, scale, delta, equity_price, and vol_price
    
    Paramters
    ---------
    rho : scalar
    scale : scalar
    phi : scalar
    equity_price : scalar
    vol_price : scalar
    
    Returns
    --------
    ndarray
    
    """
    return_mat = link2_grad_in(phi=phi, rho=rho, c=scale, delta=delta, theta=equity_price, pi=vol_price)
    
    return return_mat


def second_criterion(structural_params, link_params, weight=None):
    """
    This function computes the weighted squared deviations as defined by the link function.
    The paramters to estimate must be in alphabetical order.
    
    Paramters
    ----------
    structural_params : ndarray
    link_params : ndarray
    
    Returns
    ------
    scalar
    
    """
    # _, beta, delta_link, gamma_val, psi, phi_squared, rho_link = link_params
    scale, delta, equity_price, phi, rho, vol_price = structural_params
    
    part1 = second_stage_moments(rho=rho, c=scale, delta=delta, theta=equity_price, phi=phi, pi=vol_price).ravel()
    part2 = np.array([link_params[key] for key in sorted(link_params)])

    diff = part1 - part2
    
    if weight is None:
        weight = np.eye(len(diff))
            
    return .5 * diff @ weight @ diff
    

def compute_stage2_weight(reduced_form_cov, structural_params):

    return  np.linalg.pinv(link_grad_structural(**structural_params).T @ link_grad1(**structural_params).T)


def est_2nd_stage(reduced_form_params, reduced_form_cov, bounds=None, opts=None):
    
    if bounds is None:
        bounds = ([0,.5], [0, np.inf], [0, 10], [-1,1], [0,1], [-50, 50])
    if opts is None:
        opts = {'maxiter':200}
        
    price_guess = {'equity_price': .5, 'vol_price': -1, 'phi': - np.sqrt(reduced_form_params['phi_squared']),
                   'rho': reduced_form_params['rho'], 'scale':reduced_form_params['scale'],
                   'delta':reduced_form_params['delta']}

    x0 = [price_guess[val] for val in sorted(price_guess.keys())]
   
    init_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params), x0=x0, method="SLSQP",  
                                    options=opts, bounds=bounds)
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), init_result.x)}
    
    weight = np.linalg.pinv(link_grad_reduced(**estimates).T @ reduced_form_cov.sort_index().T.sort_index() 
               @ link_grad_reduced(**estimates))
    
    final_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params, weight=weight),
                                     x0=init_result.x, method="SLSQP",  options=opts, bounds=bounds)
                                     
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), final_result.x)}
    
    if not final_result.success:
        logging.warning("Convergence results are %s.\n", final_result)
                  
    return estimates


def compute_2nd_stage_cov(params2, cov1, time_dim):
    """ 
    This function computes the second stage covariance matrix.

    Paramters
    -------
    params2 : ndarray
        The second-stage paramters.
    cov1: 2d ndarray
        The first-stage covariance matrix.

    Returns
    ------
    cov2 : 2d ndarray
    """
    
    # This is the optimal weight matrix. 
    weight = np.linalg.pinv(link_grad_reduced(**params2).T @ cov1.sort_index().T.sort_index() @
                            link_grad_reduced(**params2))
    
    # This is the inverse of the bread for the sandwich
    inv_Bmat = np.linalg.pinv(link_grad_structural(**params2).T @ weight @ link_grad_structural(**params2))
    
    cov2 = pd.DataFrame((inv_Bmat @ link_grad_structural(**params2).T @ weight @ link_grad_structural(**params2) @
                         inv_Bmat) / time_dim, index=params2.keys(), columns=params2.keys()) 
    return cov2
