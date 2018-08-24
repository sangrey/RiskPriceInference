import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy import linalg as scilin
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import sympy as sym
import logging
from collections import OrderedDict
from functools import partial
from itertools import product
from volpriceinference import _simulate_autoregressive_gamma, _threadsafe_gaussian_rvs
import tqdm
from multiprocessing import Pool

# We define some functions
x, y, beta, gamma, rho, scale, delta, psi, zeta = sym.symbols('x y beta gamma rho scale delta psi zeta')
theta, pi = sym.symbols('theta pi')

# We define the link functions.
psi_sym = -sym.sqrt((1-zeta) / (scale * (1 + rho))) + zeta / 2 - zeta * theta
theta_sym = -zeta**(-1) * (psi + sym.sqrt((1-zeta) / (scale * (1 + rho))) -  1/2)
a_func = rho * x / ( 1 + scale * x)
alpha = psi * x + (zeta / 2) * x**2
b_func_in = 1 + scale * x
beta_sym = a_func.replace(x, pi + alpha.replace(x, theta - 1)) - a_func.replace(x, pi + alpha.replace(x, theta)) 
gamma_sym = delta * sym.log(b_func_in.replace(x, pi + alpha.replace(x, theta-1)) 
                            / b_func_in.replace(x, pi + alpha.replace(x, theta)))

# We create the link functions.
compute_gamma = sym.lambdify((delta, pi, rho, scale, theta, zeta), gamma_sym.replace(psi, psi_sym),
                             modules='numpy')
compute_beta = sym.lambdify((pi, rho, scale, theta, zeta), beta_sym.replace(psi, psi_sym), modules='numpy')
compute_psi = sym.lambdify((rho, scale, theta, zeta), psi_sym, modules='numpy')

# We now setup the link functions for the robust inference. 
link_func_sym = sym.Matrix([beta - beta_sym, gamma - gamma_sym, (1-zeta) * (theta - theta_sym)])
link_func_sym.simplify()
compute_link = sym.lambdify((beta, delta, gamma, pi, psi, rho, scale, theta, zeta), 
                       link_func_sym.replace(zeta, sym.Min(zeta,1)), modules='numpy')

link_func_grad_sym = sym.Matrix([link_func_sym.jacobian([beta, delta, gamma, psi, rho, scale, zeta])])
link_func_grad_sym.simplify()
compute_link_grad = sym.lambdify((beta, delta, gamma, pi, psi, rho, scale, theta, zeta),
                               link_func_grad_sym.replace( zeta, sym.Min(zeta,1)), modules='numpy')

link_moments_grad_sym = sym.Matrix([link_func_sym.jacobian([pi, theta])])
link_moments_grad_sym.simplify()
compute_link_price_grad = sym.lambdify((beta, delta, gamma, pi, psi, rho, scale, theta, zeta),
                               link_moments_grad_sym.replace(zeta, sym.Min(zeta,1)), modules='numpy')

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

    gamma_val = compute_gamma(rho=rho, scale=scale, delta=delta, pi=vol_price, theta=equity_price, zeta=zeta)
    beta_val = compute_beta(rho=rho, scale=scale, pi=vol_price, theta=equity_price, zeta=zeta)
    psi_val = compute_psi(rho=rho, scale=scale, theta=equity_price, zeta=zeta)
    
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
                                       x0=x0, method="SLSQP", options=options, bounds=bounds)
    
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


def estimate_zeta(data, parameter_mapping=None):
    
    if parameter_mapping is None:
        parameter_mapping = {'vol':'psi', 'vol.shift(1)':'beta', 'Intercept':'gamma'}
    
    wls_results = sm.WLS.from_formula('rtn ~ 1+ vol.shift(1) + vol', weights=data.vol**(-1), data=data).fit()
    
    estimates = wls_results.params.rename(parameter_mapping)
    
    estimates['zeta'] = np.mean(wls_results.wresid**2)
    
    zeta_cov = pd.DataFrame(np.atleast_2d(np.cov(wls_results.wresid**2)) / data.shape[0],
                            index=['zeta'], columns=['zeta'])
    return_cov = wls_results.cov_params().rename(columns=parameter_mapping).rename(parameter_mapping)
    return_cov = return_cov.merge(zeta_cov, left_index=True, right_index=True, how='outer').fillna(0)
    return_cov = return_cov.sort_index(axis=1).sort_index(axis=0)
    
    return dict(estimates), return_cov


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
    estimates : dict
    covariance : dataframe
    """
    
    if vol_estimates is None or vol_cov is None:
        # First we compute the volatility paramters.
        init_constants = compute_init_constants(data.vol)
        vol_estimates, vol_cov = compute_vol_gmm(data.vol, init_constants=init_constants)
    
    # Then we compute the reduced form paramters.
    estimates, cov_1st_stage2 = estimate_zeta(data)
    estimates.update(vol_estimates) 
    covariance = vol_cov.merge(cov_1st_stage2, left_index=True, right_index=True,
                                     how='outer').fillna(0).sort_index(axis=1).sort_index(axis=0)
    
    return estimates, covariance


def covariance_kernel(omega, vol_price1, vol_price2, equity_price1, equity_price2, omega_cov):
    """ This function computes the covariance kernel """
    
    left_bread = compute_link_grad(pi=vol_price1, theta=equity_price1,  **omega).T
    right_bread = compute_link_grad(pi=vol_price2, theta=equity_price2, **omega).T
    
    return np.squeeze(left_bread.T @ omega_cov @ right_bread)


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
        vol_estimates, vol_cov = compute_vol_gmm(data.vol, init_constants=init_constants)
    
    # Then we compute the reduced form paramters.
    reduced_form_estimates, cov_1st_stage2 = estimate_zeta(data)
    reduced_form_estimates.update(vol_estimates) 
    reduced_form_cov = vol_cov.merge(cov_1st_stage2, left_index=True, right_index=True,
                                     how='outer').fillna(0).sort_index(axis=1).sort_index(axis=0)
   
    omega_names = ['beta', 'gamma', 'delta', 'zeta', 'psi', 'rho', 'scale']

    omega_cov = reduced_form_cov.loc[omega_names, omega_names].sort_index(axis=0).sort_index(axis=1)
    
    return {name:reduced_form_estimates[name] for name in omega_names}, omega_cov


def qlr_stat(true_prices, omega, omega_cov):
    """ 
    This function computes the qlr_stat given the omega_estimates and covariance matrix.
    
    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    true_prices : dict

    Returns
    ------
    scalar 
    """

    equity_true = true_prices[0]
    vol_true = true_prices[1]
    
    cov_true_true = covariance_kernel(omega=omega, vol_price1=vol_true, vol_price2=vol_true,
                                      equity_price1=equity_true, equity_price2=equity_true, omega_cov=omega_cov)
    
    def qlr_in(prices):
        link_in = np.squeeze(compute_link(pi=prices[1], theta=prices[0], **omega)) 
        cov_pi = covariance_kernel(omega=omega, vol_price1=prices[1], vol_price2=prices[1],
                                   equity_price1=prices[0], equity_price2=prices[0], omega_cov=omega_cov)
        returnval = np.asscalar(link_in.T @ np.linalg.solve(cov_pi, link_in))
        # Values that give nan's or infinities mean that the value given is extremely bad, which since I am 
        # minimzing this mean it is extremely large.
        link_prices_grad = compute_link_price_grad(pi=prices[1], theta=prices[0], **omega) 
        jac = 2 * link_in.T @ np.linalg.solve(cov_pi, link_prices_grad)
        
        if np.isfinite(returnval):
            return returnval, jac
        else:
            return np.inf,jac

    result = optimize.minimize(qlr_in, x0=true_prices, jac=True)
    returnval = qlr_in(true_prices)[0] - result.fun

    # If the differrence above is not a number, I want the qlr_in(true_prices) to be worse.
    if np.isnan(returnval):
        returnval = np.inf
    
    return true_prices[0], true_prices[1], returnval


def qlr_sim(true_prices, omega, omega_cov, innov_dim=10):
    """ 
    This function simulates the qlr_stat given the omega_estimates and covariance matrix, redrawing the error
    relevant for computing the moments projected onto (theta,pi) many times.
    
    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    true_prices : dict

    Returns
    ------
    scalar 
    """
    equity_true = true_prices[0]
    vol_true = true_prices[1]
    
    cov_true_true = covariance_kernel(omega=omega, vol_price1=vol_true, vol_price2=vol_true,
                                      equity_price1=equity_true, equity_price2=equity_true, omega_cov=omega_cov)
    link_true = np.squeeze(compute_link(pi=vol_true, theta=equity_true, **omega)) 

    def cov_params_true(prices):
        return covariance_kernel(omega=omega, vol_price1=prices[1], vol_price2=vol_true,
                                      equity_price1=prices[0], equity_price2=equity_true, omega_cov=omega_cov)
    
    def project_resid(prices):
            link_in = np.squeeze(compute_link(pi=prices[1], theta=prices[0], **omega)) 
            
            return link_in - cov_params_true(prices) @ np.linalg.solve(cov_true_true, link_true)

    # Draw the innovation for the moments
    innovations = stats.multivariate_normal.rvs(mean=np.zeros(link_true.shape[0]), size=innov_dim)
    
    def link_star(prices, innov):
        returnval =  (project_resid(prices) + cov_params_true(prices) 
                @ np.linalg.solve(np.linalg.cholesky(cov_true_true), innov))
        if not np.any(np.isfinite(returnval)):
            np.place(returnval, ~np.isfinite(returnval), np.inf) 
            return returnval 
        else: 
            return returnval
        
    def qlr_in_star(prices, innov):
        link_in = link_star(prices=prices,innov=innov)
        cov_prices = covariance_kernel(omega=omega, vol_price1=prices[1], vol_price2=prices[1],
                                   equity_price1=prices[0], equity_price2=prices[0], omega_cov=omega_cov)
        
        # We do not need to pre-multiply by $T$ because we are using scaled versions of the covariances.
        return np.asscalar(link_in.T @ np.linalg.solve(cov_prices, link_in))   
    
    results1 = []
    results2 = [true_prices]
    for innov in innovations:
        x0 = np.mean(results2, axis=0) 

        val = optimize.minimize(lambda x: qlr_in_star(x, innov=innov), x0)

        results1.append(qlr_in_star(true_prices, innov=innov) - val.fun)
        results2.append(val.x)

    # Since we are only redrawing the second part.
    returnval = np.percentile([val if not np.isnan(val) else -np.inf for val in results1], 95)
    
    return true_prices[0], true_prices[1], returnval


def compute_qlr_stats(omega, omega_cov, equity_dim=20, vol_dim=20, vol_min=-20, vol_max=0, equity_min=0,
                      equity_max=2,  use_tqdm=True):
    """ 
    This function computes the qlr statistics and organizes them into a dataframe. 

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estimates' covariance matrix.
    equity_dim : positive scalar, optional
        The number of grid points for the equity price.
    vol_dim : positive scalar, optional
        The number of grid points for the vol price.
    vol_min : scalar, optional
        The minimum grid point for the volatility price
    vol_max : scalar, optional
        The maximum grid point for the volatility price
    equity_min : scalar, optional
        The minimum grid point for the equity price
    equity_max : scalar, optional
        The maximum grid point for the equity price
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe
    """
    
    it = product(np.linspace(equity_min, equity_max, equity_dim), np.linspace(vol_min, vol_max, vol_dim))
       
    qlr_stat_in = partial(qlr_stat, omega=omega, omega_cov=omega_cov)

    with Pool(8) as pool:
        if use_tqdm:
            draws = list(tqdm.tqdm_notebook(pool.imap_unordered(qlr_stat_in, it), total=vol_dim*equity_dim))
        else: 
            draws = list(pool.imap_unordered(qlr_stat_in, it))
    
    draws_df = pd.DataFrame.from_records(draws, columns=['equity', 'vol', 'qlr'])
    
    return draws_df.pivot(index='vol', columns='equity',values='qlr').sort_index(axis=0).sort_index(axis=1)


def compute_qlr_sim(omega, omega_cov, equity_dim=20, vol_dim=20, vol_min=-20, vol_max=0, equity_min=0,
                    equity_max=2,  innov_dim=10, use_tqdm=True):
    """ 
    This function computes the qlr statistics and organizes them into a dataframe. 

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estimates' covariance matrix.
    equity_dim : positive scalar, optional
        The number of grid points for the equity price.
    vol_dim : positive scalar, optional
        The number of grid points for the vol price.
    vol_min : scalar, optional
        The minimum grid point for the volatility price
    vol_max : scalar, optional
        The maximum grid point for the volatility price
    equity_min : scalar, optional
        The minimum grid point for the equity price
    equity_max : scalar, optional
        The maximum grid point for the equity price
    innov_dim : scalar, optional 
        The number of draws to inside the simulation.
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe
    """

    
    it = product(np.linspace(equity_min, equity_max, equity_dim), np.linspace(vol_min, vol_max, vol_dim))
    
    qlr_sim_in = partial(qlr_sim, omega=omega, omega_cov=omega_cov, innov_dim=innov_dim)
    
    with Pool(8) as pool:
        if use_tqdm:
            draws = list(tqdm.tqdm_notebook(pool.imap_unordered(qlr_sim_in, it), total=vol_dim*equity_dim))
        else: 
            draws = list(pool.imap_unordered(qlr_sim_in, it))
    
    draws_df = pd.DataFrame.from_records(draws, columns=['equity', 'vol', 'qlr'])
    
    return draws_df.pivot(index='vol', columns='equity',values='qlr').sort_index(axis=0).sort_index(axis=1)


def compute_strong_id(omega, omega_cov):
    """ 
    This function computes the estimates and covariances for the strongly identified solution.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estiamtes' covariance matrix.

    Returns
    -------
    rtn_prices : dict
        price estimates
    return_cov : dataframe
        Their covariance matrix
    """

    prices_init = [.5, -7]
    
    def qlr_in(prices, inv_weight=np.eye(3)):
        link_in = np.squeeze(compute_link(pi=prices[1], theta=prices[0], **omega)) 
        returnval = np.asscalar(link_in.T @ np.linalg.solve(inv_weight, link_in))

        return returnval
        

    rtn_prices = optimize.minimize(qlr_in, x0=prices_init).x
    bread = compute_link_grad(pi=rtn_prices[1], theta=rtn_prices[0],  **omega).T
    inner_cov  = bread.T @ omega_cov @ bread
    
    # rtn_prices = optimize.minimize(lambda x: qlr_in(x, inv_weight=inner_cov), x0=prices_init).x
    outer_bread = compute_link_price_grad(pi=rtn_prices[1], theta=rtn_prices[0], **omega)
    outer_bread_inv = np.linalg.inv(outer_bread.T @ outer_bread)

    names = ['equity_price', 'vol_price'] 
    return_cov = pd.DataFrame(outer_bread_inv @ outer_bread.T @ inner_cov @ outer_bread @ outer_bread_inv.T, 
                              columns=names, index=names)
    
    return {'equity_price': rtn_prices[0], 'vol_price': rtn_prices[1]}, return_cov


def estimate_params_strong_id(data, vol_estimates=None, vol_cov=None):
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
    estimates : dict
    covariance : dataframe
    """
     
    estimates, covariance = estimate_params(data, vol_estimates, vol_cov)
    
    price_estimates, price_cov = compute_strong_id(omega=estimates, omega_cov=covariance)
    # Then we compute the reduced form paramters.
    estimates.update(price_estimates) 
    covariance = covariance.merge(price_cov, left_index=True, right_index=True,
                                  how='outer').sort_index(axis=1).sort_index(axis=0)
    
    return estimates, covariance
