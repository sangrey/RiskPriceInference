import numpy as np
import pandas as pd
from scipy import optimize
from scipy import linalg as scilin
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import sympy as sym
import logging
from collections import OrderedDict
from volpriceinference import _simulate_autoregressive_gamma, _threadsafe_gaussian_rvs

# We define some functions
x, y, rho, scale, delta, phi, psi_sym = sym.symbols('x y rho scale delta phi psi_sym')
theta, pi = sym.symbols('theta pi')

# We define the link functions.
psi_sym = phi / sym.sqrt(scale + (1 + rho)) + (( 1 - phi**2) / 2 - (1 - phi**2) * theta)
a_func = rho * x / ( 1 + scale * x)
alpha = psi_sym * x + (( 1 -phi**2) / 2) * x**2
b_func = delta * sym.log(1 + scale * x)
beta_sym = a_func.replace(x, pi + alpha.replace(x, theta - 1)) - a_func.replace(x, pi + alpha.replace(x, theta)) 
gamma_sym = b_func.replace(x, pi + alpha.replace(x, theta-1)) - b_func.replace(x, pi + alpha.replace(x, theta))

# We create the link functions.
gamma = sym.lambdify((delta, phi, pi, rho, scale, theta), gamma_sym)
beta = sym.lambdify((phi, pi, rho, scale, theta), beta_sym)
psi = sym.lambdify((phi, rho, scale, theta), psi_sym)

# Setup the link function.
second_stage_moments_sym = sym.Matrix([beta_sym, delta, gamma_sym, phi**2, psi_sym, rho, scale])
second_stage_moments_sym.simplify()
second_stage_moments = sym.lambdify((delta, phi, pi, rho, scale, theta), second_stage_moments_sym)

# Define the gradients of the link function.
link2_grad_in = sym.lambdify((delta, phi, pi, rho, scale, theta), 
                             second_stage_moments_sym.jacobian([delta, phi, pi, rho, scale, theta]))


def simulate_autoregressive_gamma(delta=1, rho=0, scale=1, initial_point=None, time_dim=100,
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
    
    draws = _simulate_autoregressive_gamma(delta=delta,rho=rho,scale=scale,initial_point=initial_point,
                                           time_dim=time_dim)
    
    draws = pd.DataFrame(draws, pd.date_range(start=state_date, freq='D', periods=time_dim))

    return draws


def simulate_data(equity_price=1, vol_price=0, rho=0, scale=1, delta=1, phi=0, initial_point=None, time_dim=100,
                  state_date='2000-01-01'):
    """
    This function takes the reduced-form paramters and risk prices and returns the data
    
    Parameters
    --------
    equity_price: scalar
    vol_price : scalar
    phi : scalar
        leverage. It must lie in [-1,1]
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

    gamma_val = gamma(rho=rho, scale=scale, delta=delta, phi=phi, pi=vol_price, theta=equity_price)
    beta_val = beta(rho=rho, scale=scale, phi=phi, pi=vol_price, theta=equity_price)
    psi_val = psi(rho=rho, scale=scale, phi=phi, theta=equity_price)
    
    mean = gamma_val + beta_val * vol_data.shift(1) + psi_val * vol_data
    var = (1 - phi**2) * vol_data
    
    draws =  mean + np.sqrt(var) * pd.DataFrame(_threadsafe_gaussian_rvs(var.size), index=vol_data.index)
    data = pd.concat([vol_data, draws], axis=1).dropna()
    data.columns = ['vol', 'rtn']

    return data


def vol_moments(vol_data, delta, rho, scale):    
    """ Computes the moments for the volatility. """ 
    x = vol_data.values[:-1]                                                                                       
    y = vol_data.values[1:]                                                                                       

    mean = rho * x + scale * delta
    var = 2 * scale * rho * x + scale**2 * delta

    row1 = y - mean
    row2 = row1 * x
    row3 = (y**2 - (var + mean**2))
    row4 = row3 * x
    row5 = row3 * x**2

    returndf = pd.DataFrame(np.column_stack([row1, row2, row3, row4, row5]))

    return returndf

def vol_moments_grad(vol_data, delta, rho, scale):                                                                 
    """ Computes the jacobian of the volatility moments. """                                       
    x = vol_data.values[:-1]                                                                                       
    y = vol_data.values[1:]                                                                                       
                                                                                                                   
    mean = rho * x + scale * delta                                                                            
                                                                                                                   
    row1 = np.column_stack([np.full(x.shape, scale), x, np.full(x.shape, delta)])                                         
    row2 = (row1.T * x).T
    row3 = np.column_stack([scale**2  + 2 * scale * mean, 2 * scale * x + 2 * x * mean,
                            2 * rho * x + 2 * scale * delta + 2 * delta * mean])
    row4 = (row3.T * x).T                                                                                                
    row5 = (row3.T * x**2).T 
                                                                                                                   
    mom_grad_in  = -np.row_stack([np.mean(row1, axis=0), np.mean(row2, axis=0), np.mean(row3, axis=0),
                                    np.mean(row4, axis=0), np.mean(row5, axis=0)])                                    
                                                                                                                   
    return pd.DataFrame(mom_grad_in, columns=['delta', 'rho','scale'])  



def compute_init_constants(vol_data):
    """ 
    Computes some guesses for the volatlity paramters that we can use to initialize the optimization.
    
    From the model, we know that intercept = $ scale * \delta$. We also know that the average error variance equals $
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
    model = tsa.AR(vol_data).fit(maxlag=1) 
    intercept, persistence = model.params
    error_var = model.sigma2

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

    func_data = np.mean(func(data, *x), axis=0)
    
    if weight is None:
        weight = np.eye(len(func_data.T))
    
    return np.asscalar(func_data @ weight @ func_data.T)


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
        bounds = [(-1+1e-5,20),(-1, 1), (1e-5,.5)]

    if options is None:
        options = {'maxiter':200}
        
    init_constants = OrderedDict(sorted(init_constants.items(), key=lambda t: t[0]))

    x0 = list(init_constants.values())

    initial_result = optimize.minimize(lambda x: compute_mean_square(x, vol_data, vol_moments),
                                       x0=x0, method="SLSQP", bounds=bounds, options=options)
    
    moment_cov = vol_moments(vol_data, *initial_result.x).cov()
    
    # final_result = optimize.minimize(lambda x: compute_mean_square(x, vol_data, vol_moments, weight_matrix),
    #                                   x0=initial_result.x, method="SLSQP", bounds=bounds, options=options)
    final_result = initial_result
    estimates = {key:val for key,val in zip(init_constants.keys(), final_result.x)}

    # weight_matrix = scilin.pinv(vol_moments(vol_data, **estimates).cov())
    moment_derivative = vol_moments_grad(vol_data, **estimates)
    # cov = pd.DataFrame(np.linalg.pinv(moment_derivative.T @ weight_matrix @ moment_derivative),
    #                    columns=list(init_constants.keys()), index=list(init_constants.keys()))
    GprimeG = scilin.inv(moment_derivative.T @ moment_derivative)
    inner_part = moment_derivative.T @ moment_cov @ moment_derivative

    cov = pd.DataFrame(np.linalg.inv(GprimeG @ inner_part @ GprimeG.T), columns=list(init_constants.keys()),
                       index=list(init_constants.keys()))
    
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
        parameter_mapping = {'vol':'psi', 'vol.shift(1)':'beta', 'Intercept':'gamma'}
    
    wls_results = sm.WLS.from_formula('rtn ~ 1+ vol.shift(1) + vol', weights=data.vol**(-1), data=data).fit()
    
    estimates = wls_results.params.rename(parameter_mapping)
    
    # We force phi^2 >=0
    estimates['phi_squared'] =  np.maximum(1 - np.mean(wls_results.wresid**2), 0)
    
    phi2_cov = pd.DataFrame(np.atleast_2d(np.cov(wls_results.wresid**2)), index=['phi_squared'],
                        columns=['phi_squared'])
    return_cov = wls_results.cov_params().rename(columns=parameter_mapping).rename(parameter_mapping)
    return_cov = return_cov.merge(phi2_cov, left_index=True, right_index=True, how='outer').fillna(0)
    
    return dict(estimates), return_cov


def link_grad_reduced():
    """
    This function computes the jacobian of the link function with respect to the reduced form paramters
    beta, delta, gamma, phi^2, psi, rho, scale
    
    Returns
    --------
    ndarray
    """

    # The link function is of the form reduced_form_paramter - g(structural_paramter) and so its derivative with
    # respect to the reduced form paramters is the identify function.
    return_df = pd.DataFrame(np.eye(7), columns=['beta', 'delta', 'gamma', 'phi_squared', 'psi', 'rho', 'scale'],
                             index=['beta', 'delta', 'gamma', 'phi_squared', 'psi', 'rho', 'scale'])
   
    return return_df


def link_grad_structural(delta, equity_price, phi, rho, scale, vol_price):
    """
    This function computes the jacobian of the link function with respect to the structural paramters
    phi, rho, scale, delta, equity_price, and vol_price
    
    Paramters
    ---------
    equity_price : scalar
    delta : scalar
    phi : scalar
    rho : scalar
    scale : scalar
    vol_price : scalar
    
    Returns
    --------
    ndarray
    
    """
    return_mat = link2_grad_in(delta=delta, phi=phi, pi=vol_price, rho=rho, scale=scale, theta=equity_price)
    return_df = pd.DataFrame(return_mat, columns=['delta', 'phi', 'vol_price', 'rho', 'scale', 'equity_price'])

    return return_df.sort_index(axis=1)


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
    delta, equity_price, phi, rho, scale, vol_price = structural_params
    
    part1 = second_stage_moments(rho=rho, scale=scale, delta=delta, theta=equity_price, phi=phi,
                                 pi=vol_price).ravel()
    part2 = np.array([link_params[key] for key in sorted(link_params)])

    diff = part1 - part2
    
    if weight is None:
        weight = np.eye(len(diff))
            
    return .5 * diff @ weight @ diff
    

def compute_stage2_weight(reduced_form_cov, structural_params):

    return  np.linalg.pinv(link_grad_structural(**structural_params).T @ link_grad1(**structural_params).T)


def est_2nd_stage(reduced_form_params, reduced_form_cov, bounds=None, opts=None):
    
    if bounds is None:
        bounds = ([0, 20], [0, 5], [-1, 1], [-1,1], [0,.5], [-10, 10])
    if opts is None:
        opts = {'maxiter':200}
        
    price_guess = {'equity_price': .5, 'vol_price': -1, 'phi': - np.sqrt(reduced_form_params['phi_squared']),
                   'rho': reduced_form_params['rho'], 'scale':reduced_form_params['scale'],
                   'delta':reduced_form_params['delta']}

    x0 = [price_guess[val] for val in sorted(price_guess.keys())]
   
    init_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params), x0=x0, method="SLSQP",  
                                    options=opts, bounds=bounds)
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), init_result.x)}
    
    weight = np.linalg.pinv(link_grad_reduced().T @ reduced_form_cov.sort_index().T.sort_index() @
                            link_grad_reduced())
    
    final_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params, weight=weight),
                                     x0=init_result.x, method="SLSQP",  options=opts, bounds=bounds)
                                     
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), final_result.x)}
    
    if not final_result.success:
        logging.warning("Convergence results are %s.\n", final_result)

    cov = compute_2nd_stage_cov(estimates, reduced_form_cov)
                  
    return estimates, cov


def compute_2nd_stage_cov(params2, cov1):
    """ 
    This function computes the second stage covariance matrix.

    Paramters
    -------
    params2 : ndarray
        The second-stage paramters.
    cov1: 2d ndarray
        The first-stage covariance matrix. It should be divided through by the sample size.

    Returns
    ------
    cov2 : 2d ndarray
    """
    
    # This is the optimal weight matrix. 
    sorted_cov = cov1.sort_index().T.sort_index() 
    link_struct_diff = link_grad_structural(**params2)
    link_reduced_diff = link_grad_reduced()
    
    weight = np.linalg.pinv(link_reduced_diff.T @ sorted_cov @ link_reduced_diff)

    # This is the bread for the sandwich
    inv_Bmat = np.linalg.pinv(link_struct_diff.T @ weight @ link_struct_diff)
    
    # We do not need to divide through by the sample size, because sorted_cov is.
    cov2 = pd.DataFrame(inv_Bmat @ link_struct_diff.T @ weight @ link_struct_diff @ inv_Bmat,
                        index=params2.keys(), columns=params2.keys()) 
    return cov2


def estimate_params(data, vol_estimates=None, vol_cov=None):
    """ 
    We estimate the model in one step:
    
    Paramters
    ---------
    data : ndarray
        Must contain rtn and vol columns.
    vol_estimates : dict
        The volatility estimates.
    vol_cov : dataframe
        The volatility asymptotic covariance matrix.

    Returns
    ------
    params_2nd_stage : dict
    cov_2nd_stage: dataframe
    """
    
    if vol_estimates is None or vol_cov is None:
        # First we compute the volatility paramters.
        init_constants = compute_init_constants(data.vol)
        vol_estimates, vol_cov = compute_vol_gmm(data.vol, init_constants=init_constants)
    
    # Then we compute the reduced form paramters.
    reduced_form_estimates, cov_1st_stage2 = compute_step2(data)
    reduced_form_estimates.update(vol_estimates) 
    reduced_form_cov = vol_cov.merge(cov_1st_stage2, left_index=True, right_index=True,
                                     how='outer').fillna(0).sort_index(axis=1).sort_index(axis=0)
    
    # I now compute the 2nd stage.
    params_2nd_stage, cov_2nd_stage  = est_2nd_stage(reduced_form_params=reduced_form_estimates,
                                                     reduced_form_cov=reduced_form_cov)
    return params_2nd_stage, cov_2nd_stage
