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
x, y, rho, scale, delta, psi_val, zeta = sym.symbols('x y rho scale delta psi_val zeta')
theta, pi = sym.symbols('theta pi')

# We define the link functions.
psi_sym = sym.sqrt(1-zeta) / sym.sqrt(scale + (1 + rho)) + (zeta / 2 - zeta * theta)
a_func = rho * x / ( 1 + scale * x)
alpha = psi_sym * x + (zeta / 2) * x**2
b_func_in = 1 + scale * x
beta_sym = a_func.replace(x, pi + alpha.replace(x, theta - 1)) - a_func.replace(x, pi + alpha.replace(x, theta)) 
gamma_sym = delta * sym.log(b_func_in.replace(x, pi + alpha.replace(x, theta-1)) 
                            / b_func_in.replace(x, pi + alpha.replace(x, theta)))

# We create the link functions.
gamma = sym.lambdify((delta, pi, rho, scale, theta), gamma_sym)
beta = sym.lambdify((pi, rho, scale, theta, zeta), beta_sym)
psi = sym.lambdify((rho, scale, theta, zeta), psi_sym)

# Setup the link function.
second_stage_moments_sym = sym.Matrix([beta_sym, delta, gamma_sym, psi_sym, rho, scale, zeta])
second_stage_moments_sym.simplify()
second_stage_moments = sym.lambdify((delta, pi, rho, scale, theta, zeta), second_stage_moments_sym)

# Define the gradients of the link function.
link2_grad_in = sym.lambdify((delta, pi, rho, scale, theta, zeta), second_stage_moments_sym.jacobian(
    [delta, pi, rho, scale, theta, zeta]))

# We now setup the link functions for the robust inference. 
link_pi_sym = sym.Matrix([beta_sym, gamma_sym]).replace(theta, zeta**(-1) * (psi_val - (-sym.sqrt(1-zeta)/
                                                                                        sym.sqrt(1 + rho) +
                                                                                        zeta/2)))

link_pi_in = sym.lambdify((delta, pi, psi_val, rho, scale, zeta), link_pi_sym)
link_pi_grad_in = sym.lambdify((delta, pi, psi_val, rho, scale, zeta), link_pi_sym.jacobian((delta, psi_val, rho,
                                                                                             scale, zeta)))


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


def simulate_data(equity_price=1, vol_price=0, rho=0, scale=1, delta=1, zeta=1, initial_point=None, time_dim=100,
                  state_date='2000-01-01'):
    """
    This function takes the reduced-form paramters and risk prices and returns the data
    
    Parameters
    --------
    equity_price: scalar
    vol_price : scalar
    rho : scalar
        persistence
    scale : positive scalar
    initial_point: scalar, optional
        Starting value for the volatility
    time_dim : int, optional
        number of periods
    start_date : datelike, optional
        The time to start the data from.
    zeta : scalar
        1 - leverage**2. It must lie in (0,1)
        
    Returns
    -----
    draws : dataframe
    
    """
    vol_data = simulate_autoregressive_gamma(rho=rho, scale=scale, delta=delta, initial_point=initial_point,
                                             state_date=pd.to_datetime(state_date) - pd.Timedelta('1 day'),
                                             time_dim=time_dim + 1)

    gamma_val = gamma(rho=rho, scale=scale, delta=delta, pi=vol_price, theta=equity_price, zeta=zeta)
    beta_val = beta(rho=rho, scale=scale, pi=vol_price, theta=equity_price, zeta=zeta)
    psi_val = psi(rho=rho, scale=scale, theta=equity_price, zeta=zeta)
    
    mean = gamma_val + beta_val * vol_data.shift(1) + psi_val * vol_data
    var = zeta * vol_data
    
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

    returndf = pd.DataFrame(np.column_stack([row1, row2, row3, row4, row5]), index=vol_data.index[1:])

    return returndf

def vol_moments_grad(vol_data, delta, rho, scale):                                                                 
    """ Computes the jacobian of the volatility moments. """                                       
    x = vol_data.values[:-1]                                                                                       
                                                                                                                   
    mean = rho * x + scale * delta                                                                            
                                                                                                                   
    row1 = np.column_stack([np.full(x.shape, scale), x, np.full(x.shape, delta)])                                         
    row2 = (row1.T * x).T
    row3 = np.column_stack([scale**2  + 2 * scale * mean, 2 * scale * x + 2 * x * mean, 2 * rho * x + 2 * scale *
                            delta + 2 * delta * mean])
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
    weight_matrix = scilin.pinv(vol_moments(vol_data, *initial_result.x).cov());
    
    final_result = optimize.minimize(lambda x: compute_mean_square(x, vol_data, vol_moments, weight_matrix),
                                      x0=initial_result.x, method="SLSQP", bounds=bounds, options=options)
    estimates = {key:val for key,val in zip(init_constants.keys(), final_result.x)}

    weight_matrix = scilin.pinv(vol_moments(vol_data, **estimates).cov())
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
        parameter_mapping = {'vol':'psi', 'vol.shift(1)':'beta', 'Intercept':'gamma'}
    
    wls_results = sm.WLS.from_formula('rtn ~ 1+ vol.shift(1) + vol', weights=data.vol**(-1), data=data).fit()
    
    estimates = wls_results.params.rename(parameter_mapping)
    
    estimates['zeta'] = np.mean(wls_results.wresid**2)
    
    zeta_cov = pd.DataFrame(np.atleast_2d(np.cov(wls_results.wresid**2)), index=['zeta'], columns=['zeta'])
    return_cov = wls_results.cov_params().rename(columns=parameter_mapping).rename(parameter_mapping)
    return_cov = return_cov.merge(zeta_cov, left_index=True, right_index=True, how='outer').fillna(0)
    return_cov = return_cov.sort_index(axis=1).sort_index(axis=0)
    
    return dict(estimates), return_cov


def link_grad_structural(delta, equity_price, rho, scale, vol_price, zeta):
    """
    This function computes the jacobian of the link function with respect to the structural paramters
    rho, scale, delta, equity_price, and vol_price, zeta
    
    Paramters
    ---------
    equity_price : scalar
    delta : scalar
    rho : scalar
    scale : scalar
    vol_price : scalar
    zeta : scalar
    
    Returns
    --------
    ndarray
    
    """
    return_mat = link2_grad_in(delta=delta, pi=vol_price, rho=rho, scale=scale, theta=equity_price, zeta=zeta)
    return_df = pd.DataFrame(return_mat, columns=['delta', 'vol_price', 'rho', 'scale', 'equity_price', 'zeta'])

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
    delta, equity_price, rho, scale, vol_price, zeta = structural_params
    
    part1 = second_stage_moments(rho=rho, scale=scale, delta=delta, theta=equity_price, pi=vol_price, 
                                zeta=zeta).ravel()
    part2 = np.array([link_params[key] for key in sorted(link_params)])

    diff = part1 - part2
    
    if weight is None:
        weight = np.eye(len(diff))
            
    return .5 * diff @ weight @ diff
    

def est_2nd_stage(reduced_form_params, reduced_form_cov, bounds=None, opts=None):
    
    if bounds is None:
        bounds = ([0, 20], [0, 5], [-1, 1], [-1,1], [0,.5], [-10, 10])
    if opts is None:
        opts = {'maxiter':200}
        
    price_guess = {'equity_price': .5, 'vol_price': -1, 'zeta': reduced_form_params['zeta'],
                   'rho': reduced_form_params['rho'], 'scale':reduced_form_params['scale'],
                   'delta':reduced_form_params['delta']}

    x0 = [price_guess[val] for val in sorted(price_guess.keys())]
   
    init_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params), x0=x0, method="SLSQP",  
                                    options=opts, bounds=bounds)
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), init_result.x)}
    
    weight = np.linalg.pinv(reduced_form_cov.sort_index().T.sort_index())
    
    final_result = optimize.minimize(lambda x: second_criterion(x, reduced_form_params, weight=weight),
                                     x0=init_result.x, method="SLSQP",  options=opts, bounds=bounds)
                                     
    estimates = {key:val for key, val in zip(sorted(price_guess.keys()), final_result.x)}
    
    if not final_result.success:
        logging.warning("Convergence results are %s.\n", final_result)

    link_derivative = link_grad_structural(**estimates)
    cov = pd.DataFrame(scilin.pinv(link_derivative.T @ weight @ link_derivative) , index=estimates.keys(),
                       columns=estimates.keys()) 
                  
    return estimates, cov


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


def link_pi(omega, pi):
    """ Computes the link function relevent for estimating pi. """

    returnval = np.squeeze(link_pi_in(delta=omega['delta'], psi=omega['psi'], rho=omega['rho'],
                                      scale=omega['scale'], pi=pi, zeta=omega['zeta']))
    return returnval


def covariance_kernel(omega, vol_price1, vol_price2, reduced_form_cov):
    """ This function computes the covariance kernel """
    
    left_bread = link_pi_grad_in(delta=omega['delta'], pi=vol_price1, psi=omega['psi'],
                                 rho=omega['rho'], scale=omega['scale'], zeta=omega['zeta']).T
    right_bread = link_pi_grad_in(delta=omega['delta'], pi=vol_price2, psi=omega['psi'],
                                  rho=omega['rho'], scale=omega['scale'], zeta=omega['zeta']).T
    
    return np.squeeze(left_bread.T @ reduced_form_cov @ right_bread)


def compute_omega(data, vol_estimates=None, vol_cov=None):
    """ 
    We compute the reduced-form paramters and their covariance matrix. 
    
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
    omega_est : dict
    omega_cov: dataframe
    """
    
    if vol_estimates is None or vol_cov is None:
        # First we compute the volatility paramters.
        init_constants = compute_init_constants(data.vol)
        vol_estimates, vol_cov = vl.compute_vol_gmm(data.vol, init_constants=init_constants)
    
    # Then we compute the reduced form paramters.
    reduced_form_estimates, cov_1st_stage2 = compute_step2(data)
    reduced_form_estimates.update(vol_estimates) 
    reduced_form_cov = vol_cov.merge(cov_1st_stage2, left_index=True, right_index=True,
                                     how='outer').fillna(0).sort_index(axis=1).sort_index(axis=0)
   
    omega_names = ['beta', 'gamma', 'delta', 'zeta', 'psi', 'rho', 'scale']

    omega_cov = reduced_form_cov.loc[omega_names, omega_names].sort_index(axis=0).sort_index(axis=1)
    
    return {name:reduced_form_estimates[name] for name in omega_names}, omega_cov


# def projection_residual_process(omega, vol_price, vol_price_true, reduced_form_cov):
#     """ This function computes the j
#     link1 = link_pi(omega=omega, vol_price=vol_price) 
#     link2 = link_pi(omega=omega, vol_price=vol_price_true) 

#     cov1 = covariance_kernel(omega=omega, vol_price1=vol_price, vol_price2=vol_price_true,
#                              reduced_form_cov=reduced_form_cov)
#     cov2 = covariance_kernel(omega=omega, vol_price1=vol_price_true, vol_price2=vol_price_true,
#                              reduced_form_cov=reduced_form_cov)

#     return np.squeeze(link1 - cov1 @ np.linalg.solve(cov2, link2))


def qlr_stat(omega, vol_price_true, omega_cov, time_dim, bounds=None):
    """ This function computes the qlr_stat given the omega_estimates and covariance matrix.
    
    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    vol_price_true : scalar
    time_dim : scalar
        number of period.
    bounds : iterable
        bounds on pi

    Returns
    ------
    scalar 
    """
    bounds = [(bounds[0], bounds[1])] if bounds is not None else [(-50, 0)]
    
    cov_true_true = covariance_kernel(omega=omega, vol_price1=vol_price_true, vol_price2=vol_price_true,
                                      omega_cov=omega_cov)
    
    def qlr_in(pi):
        link_pi_in = link_pi(omega, pi=pi)
        cov_pi = covariance_kernel(omega=omega, vol_price1=pi, vol_price2=pi, omega_cov=omega_cov)
        
        return np.asscalar(time_dim * link_pi_in.T @ np.linalg.solve(cov_pi, link_pi_in))
    
    result = optimize.minimize(lambda x: qlr_in(x), x0=vol_price_true, method="SLSQP", bounds=bounds)

    return qlr_in(vol_price_true) - result.fun


def qlr_sim(omega, vol_price_true, reduced_form_cov, time_dim):
    
    cov_true_true = covariance_kernel(omega=omega, vol_price1=vol_price_true, vol_price2=vol_price_true,
                                      reduced_form_cov=reduced_form_cov)
    
#     # We start by simulating v_{moment_conds}
    upsilon_star = stats.multivariate_normal.rvs(mean=np.zeros(2), cov=cov_true_true)
    
    def link_star(pi):
        link_pi_in = link_pi(omega=omega, pi=pi)
        cov_pi_true =  covariance_kernel(omega=omega, vol_price1=pi, vol_price2=vol_price_true,
                                         reduced_form_cov=reduced_form_cov)
        
        # We combine computing h(pi, omega) and g_star into one step.
        link_star_in = link_pi_in - cov_pi_true @ np.linalg.solve(cov_true_true,
                                                                  link_pi(omega=omega,
                                                                          pi=vol_price_true)-upsilon_star)
        
        return link_star_in
    
    def qlr_in_star(pi):
        link_pi_in = link_star(pi=pi)
        cov_pi_pi = covariance_kernel(omega=omega, vol_price1=pi, vol_price2=pi, reduced_form_cov=reduced_form_cov)
        
        return np.asscalar(time_dim * link_pi_in.T @ np.linalg.solve(cov_pi_pi, link_pi_in))   
    
    result = optimize.minimize(lambda x: qlr_in_star(x), x0=vol_price_true, method="SLSQP", bounds=[(-20, 0)])
    
    return qlr_in_star(vol_price_true) - result.fun
    
