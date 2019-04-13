"""This module contains the functions that are used to simulate and estimate the model and perform inference."""
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import minimize
from scipy.signal import windows
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import sympy as sym
import logging
from collections import OrderedDict
from functools import partial
from itertools import product
from libvolpriceinference import _simulate_autoregressive_gamma
from libvolpriceinference import link_total, link_jacobian, _covariance_kernel
from libvolpriceinference import compute_beta, compute_gamma, compute_psi
from tqdm.auto import tqdm
from multiprocessing import Pool

# We define some functions
_x, _y, beta, gamma, psi = sym.symbols('_x _y beta gamma psi', real=True, positive=True)
logit_rho, log_scale, log_both, zeta = sym.symbols("""logit_rho log_scale
                                                   log_both zeta""", real=True,
                                                   positive=True)
theta, pi, phi, pi1, pi2, theta1, theta2, phi1, phi2 = sym.symbols("""theta pi
                                                                   phi pi1 pi2
                                                                   theta1
                                                                   theta2 phi1
                                                                   phi2""")

#  We define the functions that specify the model.
_logit = sym.log(_x) - sym.log(1 - _x)
_logistic = 1 / (1 + sym.exp(-1 * _x))

_psi_sym = (phi / sym.sqrt(2 * sym.exp(log_scale))) - (1 - phi**2) / 2 + (1 - phi**2) * theta
_theta_sym = sym.solveset(psi - _psi_sym, theta).args[0]
_B_func_in = 1 + sym.exp(log_scale) * _x
_A_func = _logistic.xreplace({_x: logit_rho}) * _x / _B_func_in
_C_func = psi * _x - ((1 - phi**2) / 2) * _x**2

_beta_sym = (_A_func.xreplace({_x: pi + _C_func.xreplace({_x: theta - 1})}) -
             _A_func.xreplace({_x: pi + _C_func.xreplace({_x: theta})}))

# We create the functions that define the nonlinear constraint implied by the argument of the logarithm needing to
# be positive.
_constraint_sym = _B_func_in.xreplace({_x: pi + _C_func})
_gamma_sym = sym.exp(log_both - log_scale) * (sym.log(_constraint_sym.xreplace({_x: theta - 1})) -
                                              sym.log(_constraint_sym.xreplace({_x: theta})))
_constraint1 = sym.lambdify((phi, pi, theta, log_scale, logit_rho, psi),
                            _constraint_sym.xreplace({_x: theta - 1}), modules='numpy')
_constraint2 = sym.lambdify((phi, pi, theta, log_scale, logit_rho, psi),
                            _constraint_sym.xreplace({_x: theta}), modules='numpy')

# We create a function to initialize the paramters with reasonable guesses in the optimization algorithms.
compute_theta = sym.lambdify((psi, logit_rho, log_scale, zeta),
                             _theta_sym.xreplace({phi: sym.Min(0, -sym.sqrt(1 - zeta))}),
                             modules='numpy')
_pi_from_gamma_in = _B_func_in.xreplace({_x: pi + _C_func})
_pi_from_gamma = sym.powsimp(sym.expand(sym.solveset(sym.exp(gamma / sym.exp(log_both - log_scale)) -
                                                     (_pi_from_gamma_in.xreplace({_x: theta - 1}) /
                                                      _pi_from_gamma_in.xreplace({_x: theta})),
                                                     pi).args[0].args[0]))

compute_pi = sym.lambdify((gamma, log_both, log_scale, phi, psi, logit_rho, theta), _pi_from_gamma, modules='numpy')

# We create the functions to jointly specify the links.
_link_sym = sym.powsimp(sym.expand(sym.Matrix([beta - _beta_sym, gamma - _gamma_sym, psi - _psi_sym,
                                               1 - (zeta + phi**2)])))

# We define the moments used to estimate the volatility paramters.
_mean = _logistic.xreplace({_x: logit_rho}) * _x + sym.exp(log_both)
_var = 2 * sym.exp(log_scale) * _logistic.xreplace({_x: logit_rho}) * _x + sym.exp(log_scale + log_both)

# I now compute the heteroskedasticity-adjusted moments.
_row1 = (_y - _mean)
_row3 = ((_y - _mean)**2 - _var)

_vol_moments = sym.Matrix([_row1, _row1 * _x, _row3, _row3 * _x, _row1 * _x**2])

compute_vol_moments = sym.lambdify([_x, _y, log_both, log_scale, logit_rho], _vol_moments, modules='numpy')
compute_vol_moments_grad = sym.lambdify([_x, _y, log_both, log_scale, logit_rho],
                                        _vol_moments.jacobian([log_both, log_scale, logit_rho]), modules='numpy')

# Define the gradient of the link function with respect to the structural paramters.
_link_price_grad_sym = sym.powsimp(sym.expand(sym.Matrix([_link_sym.jacobian([phi, pi, theta])])))

_link_price_grad_in = sym.lambdify((phi, pi, theta, beta, gamma, log_both,
                                    log_scale, psi, logit_rho, zeta),
                                   _link_price_grad_sym, modules='numpy')


def constraint(prices, omega):
    """Compute the constraint implied by logarithm's argument in the second link function being postiive."""
    constraint1 = _constraint1(*prices, logit_rho=omega['logit_rho'],
                               log_scale=omega['log_scale'], psi=omega['psi'])
    constraint2 = _constraint2(*prices, logit_rho=omega['logit_rho'],
                               log_scale=omega['log_scale'], psi=omega['psi'])

    return np.minimum(constraint1, constraint2)


def compute_moments(log_both, logit_rho, log_scale, phi, pi, theta, psi):
    """Compute the means and variances implied by the paramters."""
    rho = special.expit(logit_rho)

    vol_mean = np.exp(log_both) / (1 - rho)
    vol_var = ((2 * np.exp(log_scale) * rho * vol_mean + np.exp(log_scale)**2
                * np.exp(log_both - log_scale)) / (1 - rho**2))

    psi = compute_psi(log_scale=log_scale, phi=phi, theta=theta)
    beta = compute_beta(logit_rho=logit_rho, log_scale=log_scale, phi=phi, pi=pi, theta=theta, psi=psi)
    gamma = compute_gamma(log_both=log_both, psi=psi, log_scale=log_scale,
                          phi=phi, pi=pi, theta=theta)

    return_mean = psi * vol_mean + beta * vol_mean + gamma
    return_var = psi**2 * vol_var + beta**2 * vol_var + (1 - phi**2) * vol_mean

    return {'return_mean': return_mean, 'return_var': return_var, 'vol_mean': vol_mean, 'vol_var': vol_var}


def compute_constraint_prices(omega, omega_cov, bounds):
    """Compute the slackness in the nonlinear constraint."""
    phi_init = -np.sqrt(1 - omega['zeta']) if omega['zeta'] < 1 else 0
    theta_init = compute_theta(psi=omega['psi'], log_scale=omega['log_scale'],
                               logit_rho=omega['logit_rho'], zeta=omega['zeta'])

    vals = -1 * stats.truncexpon.rvs(loc=-bounds['pi']['max'], b=-bounds['pi']['min'], size=50)
    arg_list = [(val, _qlr_in([phi_init, val, theta_init], omega, omega_cov)) for val in vals]

    pi_est = compute_pi(log_both=omega['log_both'], gamma=omega['gamma'],
                        psi=omega['psi'], logit_rho=omega['logit_rho'],
                        log_scale=omega['log_scale'], theta=theta_init,
                        phi=phi_init)

    if np.isfinite(pi_est):
        arg_list.append((pi_est, _qlr_in([phi_init, pi_est, theta_init], omega, omega_cov)))

    pi_init = pd.DataFrame(arg_list).sort_values(1).iloc[0, 0]

    prices_init = [phi_init, pi_init, theta_init]
    constraint_in = partial(constraint, omega=omega)
    constraint_dict = {'type': 'ineq', 'fun': constraint_in}

    # I ensure that the constraint is satisfied at the initial point.
    if constraint_in(prices_init) < 0:
        prices_init[1] = -.1

    if constraint_in(prices_init) < 0:
        prices_init[2] = .1

    return constraint_dict, prices_init


def compute_hac_num_lags(data, kernel="parzen"):
    """
    Compute the number of lags for an AR(1) process.

    It uses the plug-in formula developed by Andrews (1991).  (It uses a weight equal to 1, which is
    irrelevant because we are assuming univariate data.

    Parameters
    --------
    data : dataframe
        The AR(1) coeffiicent
    kernel: str
        the kernel to use
    Returns
    ------
    num_lags : positive integer

    """
    data = pd.DataFrame(data)
    data.columns = np.arange(data.shape[1])
    # This is Andrews (1991) Eq. 6.4.
    slopes_and_vars = []
    for _, col in data.items():

        data_in = col.to_frame()
        data_in.columns = ['name']
        model = tsa.AR(data_in).fit(maxlag=1)
        intercept, slope = model.params
        innov_var = model.sigma2
        slopes_and_vars.append([slope, innov_var])

    slope_and_var_df = pd.DataFrame(slopes_and_vars, columns=['slope', 'var'])

    summands = np.array([[(4 * row.slope**2 * row.var**4) / (1 - row.slope)**8,
                          row.var**4 / (1 - row.slope**4)]
                         for row in slope_and_var_df.itertuples()])

    alpha_2 = np.mean(summands[:, 0]) / np.mean(summands[:, 1])
    time_dim = data.shape[0]

    if kernel == "parzen":
        bandwidth = 2.6614 * (alpha_2 * time_dim)**.2
    elif kernel == "tukey":
        bandwidth = 1.7452 * (alpha_2 * time_dim)**.2
    elif kernel == "quadratic-spectral":
        bandwidth = 1.3221 * (alpha_2 * time_dim)**.2
    else:
        raise NotImplementedError("We only support the parzen, tukey, and quadratic-spectral kernels.")

    # We do not want to average over subsequences that are more than the square-root of the sample size.
    # This is essentially changing the constant because it is creating a maximum value for alpha_2)
    return np.int(max(bandwidth, time_dim**.25))


def compute_names():
    """Compute the names."""
    return ['phi', 'pi', 'theta']


def sliding_window_view(arr, window):
    """
    Compute an effiicent rolling window function.

    Paramters
    -------
    arr : arraylike
        The array to roll over
    window : positive scalar
        The length of the window
    Returns
    ------
    iterator

    """
    shape = (window, arr.shape[1])
    arr = np.asarray(arr)
    output_len = arr.shape[0] - window + 1

    new_shape = (output_len,) + shape
    new_strides = (arr.strides[0],) + arr.strides
    return_data = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides,
                                                  subok=False, writeable=False)

    return return_data


def compute_link(prices, omega):
    """Compute the link function."""
    return link_total(*prices, **omega)


def compute_link_grad(prices, omega):
    """Compute the gradient of the link function with respect to the reduced-form paramters."""
    result = link_jacobian(phi=prices[0], pi=prices[1], theta=prices[2],
                           log_both=omega['log_both'],
                           log_scale=omega['log_scale'],
                           logit_rho=omega['logit_rho'], psi=omega['psi'])

    return result


def compute_link_price_grad(prices, omega):
    """Compute the gradient of the link function with respect to the structural parameters."""
    return np.atleast_2d(_link_price_grad_in(*prices, **omega))


def covariance_kernel(prices1, prices2, omega, omega_cov):
    """Compute the covarinace of the implied gaussian process as a function of the structural paramters."""
    result = _covariance_kernel(*prices1, *prices2, psi=omega['psi'],
                                log_both=omega['log_both'],
                                log_scale=omega['log_scale'],
                                logit_rho=omega['logit_rho'],
                                omega_cov=omega_cov)
    return result


def simulate_autoregressive_gamma(log_both=0, logit_rho=0, log_scale=0, initial_point=None, time_dim=100,
                                  start_date='2000-01-01'):
    """
    Provide draws from the ARG(1) process of Gourieroux & Jaisak.

    Parameters
    --------
    logit_rho : scalar
        AR(1) coefficient
    log_both : scalar
        intercept
    log_scale : scalar
    Returns

    -----
    draws : dataframe

    """
    # If initial_point is not specified, we start at the unconditional mean.

    initial_point = ((sym.exp(log_both)) / (1 - special.expit(logit_rho)) if initial_point is None
                     else initial_point)

    draws = _simulate_autoregressive_gamma(delta=np.exp(log_both - log_scale), rho=special.expit(logit_rho),
                                           scale=np.exp(log_scale), initial_point=initial_point, time_dim=time_dim)

    draws_df = pd.Series(draws, pd.date_range(start=start_date, freq='D', periods=len(draws))).to_frame()

    return draws_df


def simulate_data(theta=1, pi=0, logit_rho=0, log_scale=0, log_both=0, phi=0, initial_point=None, time_dim=100,
                  start_date='2000-01-01'):
    """
    Take the reduced-form paramters and risk prices and returns the data.

    Parameters
    --------
    theta: scalar
    pi : scalar
    logit_rho : scalar
        persistence
    log_scale : positive scalar
    initial_point: scalar, optional
        Starting value for the volatility
    time_dim : int, optional
        number of periods
    start_date : datelike, optional
        The time to start the data from.
    phi : scalar
        The leverage effect. It must lie in (0,1)

    Returns
    -----
    draws : dataframe

    """
    vol_data = simulate_autoregressive_gamma(logit_rho=logit_rho, log_scale=log_scale, log_both=log_both,
                                             initial_point=initial_point, time_dim=time_dim + 1,
                                             start_date=pd.to_datetime(start_date) - pd.Timedelta('1 day'))

    psi_val = compute_psi(log_scale=log_scale, theta=theta, phi=phi)
    gamma_val = compute_gamma(log_both=log_both, psi=psi_val, log_scale=log_scale, phi=phi, pi=pi, theta=theta)
    beta_val = compute_beta(logit_rho=logit_rho, log_scale=log_scale, pi=pi, theta=theta, phi=phi, psi=psi_val)

    price_in = (phi, pi, theta)
    if constraint(prices=price_in, omega={'psi': psi_val, 'logit_rho': logit_rho, 'log_scale': log_scale}) < 0:
        raise ValueError(f"""The set of paramters given conflict with each other. No process exists with those
                             paramters. You might want to make the volatility price smaller in magnitude.""")

    mean = gamma_val + beta_val * vol_data.shift(1) + psi_val * vol_data
    var = (1 - phi**2) * vol_data

    draws = mean + np.sqrt(var) * pd.DataFrame(np.random.standard_normal(var.size), index=vol_data.index)
    data = pd.concat([vol_data, draws], axis=1).dropna()
    data.columns = ['vol', 'rtn']

    return data


def vol_moments(vol_data, log_both, log_scale, logit_rho):
    """Compute the moments of the volatility process."""
    x = vol_data.values[:-1]
    y = vol_data.values[1:]

    moments = np.squeeze(compute_vol_moments(x, y, log_both=log_both, log_scale=log_scale, logit_rho=logit_rho)).T

    return pd.DataFrame(moments)


def vol_moments_grad(vol_data, log_both, log_scale, logit_rho):
    """Compute the jacobian of the volatility moments."""
    grad = np.mean([compute_vol_moments_grad(x, y, log_both=log_both, log_scale=log_scale, logit_rho=logit_rho)
                    for x, y in zip(vol_data.values[:-1], vol_data.values[1:])], axis=0)

    return pd.DataFrame(grad, columns=['log_both', 'log_scale', 'logit_rho'])


def compute_init_constants(vol_data):
    """
    Compute guesses for the volatlity paramters that we use to initialize the optimization.

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

    init_constants = {'log_both': max(np.log(intercept), -10)}
    init_constants['log_scale'] = max(np.log(np.var(vol_data)
                                             * (1 - persistence**2))
                                      - np.log(2 * persistence *
                                               np.mean(vol_data) + intercept),
                                      -10)
    init_constants['logit_rho'] = max(special.logit(persistence), -10)

    return init_constants


def compute_mean_square(x, data, func, weight=None):
    """
    Apply func to the data at *x, and then computes its weighted mean square error.

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


def compute_vol_gmm(vol_data, init_constants=None, options=None):
    """
    Use GMM to compute the volatility paramters and their asymptotic covariance matrix.

    Paramters
    ---------
    vol_data : pandas series
    init_constants : dict, None
        This must contain initial guesses for the paramters as values and their names as keys.
    options: dict, optional

    Returns
    --------
    final_result : dict
    cov : ndarray

    """
    if options is None:
        options = {'maxiter': 200}

    if init_constants is None:
        init_constants = compute_init_constants(vol_data)

    x0 = list(init_constants.values())
    init_constants = OrderedDict(sorted(init_constants.items(), key=lambda t: t[0]))

    initial_result = minimize(
        lambda x: vol_data.shape[0]**2 * compute_mean_square(x, vol_data, vol_moments),
        x0=x0, method='BFGS', options=options)

    if not initial_result['success']:
        logging.warning(initial_result)

    vol_moments_data1 = vol_moments(vol_data, *initial_result.x)
    num_lags = compute_hac_num_lags(vol_moments_data1, kernel='parzen')
    win = windows.parzen(M=2 * num_lags, sym=True) / np.sum(windows.parzen(M=2 * num_lags, sym=True))
    sliding_it1 = sliding_window_view(vol_moments_data1, window=len(win))
    moment_cov1 = np.mean([(x.T * win).dot(x) for x in sliding_it1], axis=0)
    weight_matrix = np.linalg.pinv(moment_cov1)

    final_result = minimize(
        lambda x: compute_mean_square(x, vol_data, vol_moments, weight_matrix),
        x0=initial_result.x, method="BFGS", options=options
    )

    if not final_result['success']:
        logging.warning(final_result)

    estimates = {key: val for key, val in zip(init_constants.keys(), final_result.x)}
    vol_moments_data2 = vol_moments(vol_data, *final_result.x)
    sliding_it2 = sliding_window_view(vol_moments_data2, window=len(win))
    moment_cov2 = np.mean([(x.T * win).dot(x) for x in sliding_it2], axis=0)
    moment_derivative = vol_moments_grad(vol_data, **estimates)

    cov = pd.DataFrame(np.linalg.pinv(moment_derivative.T @ np.linalg.solve(moment_cov2, moment_derivative)),
                       columns=list(init_constants.keys()), index=list(init_constants.keys()))

    if not final_result.success:
        logging.warning("Convergence results are %s.\n", final_result)

    return estimates, cov / (vol_data.size - 1)


def create_est_table(estimates, truth, cov, num_se=1.96):
    """
    Create a table that prints the estimates, truth, and confidence interval.

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
    upper_ci = [estimates[name] + num_se * np.sqrt(cov.loc[name, name]) for name in names]

    return_df = pd.DataFrame(np.column_stack([true_values, est_values, lower_ci, upper_ci]),
                             columns=['truth', 'estimate', 'lower ci', 'upper ci'], index=names)
    return_df.sort_index(inplace=True)
    return_df['in_ci'] = ((return_df['truth'] >= return_df['lower ci'])
                          & (return_df['truth'] <= return_df['upper ci']))

    return return_df


def cov_to_corr(cov):
    """Convert a covariance matrix to a correlation matrix."""
    corr = pd.DataFrame(np.atleast_2d(np.diag(cov))**(-.5) * cov.values / np.atleast_2d(np.diag(cov)).T**.5,
                        index=cov.index, columns=cov.columns)

    return corr


def estimate_zeta(data, parameter_mapping=None):
    """Estimate the log_scaled covariance paramter."""
    if parameter_mapping is None:
        parameter_mapping = {'vol': 'psi', 'vol.shift(1)': 'beta', 'Intercept': 'gamma'}

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
    Estimate the reduced-form model in one step.

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


def compute_omega(data, vol_estimates=None, vol_cov=None):
    """
    Compute the reduced-form paramters and their covariance matrix.

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

    omega_names = ['beta', 'gamma', 'log_both', 'log_scale', 'psi', 'logit_rho', 'zeta']

    covariance = reduced_form_cov.loc[omega_names, omega_names].sort_index(axis=0).sort_index(axis=1)

    return {name: reduced_form_estimates[name] for name in omega_names}, covariance


def _qlr_in(prices, omega, omega_cov):

    link_in = compute_link(prices=prices, omega=omega)

    cov_pi = covariance_kernel(prices, prices, omega_cov=omega_cov, omega=omega)

    try:
        returnval = np.asscalar(link_in.T @ np.linalg.solve(cov_pi, link_in))
    except np.linalg.LinAlgError:
        returnval = np.inf

    if np.isnan(returnval):
        logging.warning("_qlr_in found a nan-value")
        returnval = np.inf

    return returnval


def _minimize_function_multiple_x0(func_to_minimize, x0, omega, bounds, true_prices):

    minimize_result = minimize(func_to_minimize, x0=x0, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 2500})

    x0 = np.asarray(true_prices)

    result_in = minimize(func_to_minimize, x0=x0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 2500})

    if result_in.fun <= minimize_result.fun:
        minimize_result = result_in

    x0[0] = (np.clip(-(1 - omega['zeta'])**.5, bounds[0][0], bounds[0][1]) if
             omega['zeta'] < 1 else bounds[0][1])

    x0[1] = max(bounds[1])
    x0[2] = min(bounds[2])

    result_in = minimize(func_to_minimize, x0=x0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 2500})

    if result_in.fun <= minimize_result.fun:
        minimize_result = result_in

    if not minimize_result['success']:
        logging.warning(minimize_result)

    return minimize_result


def qlr_stat(true_prices, omega, omega_cov, bounds):
    """
    Compute the qlr_stat given the omega_estimates and covariance matrix.

    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    true_prices : dict
    bounds : dict of dict

    Returns
    ------
    scalar

    """
    bounds_in = [(bounds['phi']['min'], bounds['phi']['max']),
                 (bounds['pi']['min'], bounds['pi']['max']),
                 (bounds['theta']['min'], bounds['theta']['max'])]

    low_bounds, high_bounds = np.array(bounds_in).T
    x0 = np.random.uniform(low_bounds, high_bounds)

    constraint_dict, init = compute_constraint_prices(omega=omega,
                                                      omega_cov=omega_cov,
                                                      bounds=bounds)

    # If we violate the contraint, we want to always reject.
    if constraint_dict['fun'](true_prices, omega=omega) < 0:
        logging.warning("We violated the constraint.")
        return tuple(true_prices) + (np.inf,)

    func_to_minimize = partial(_qlr_in, omega=omega, omega_cov=omega_cov)

    minimize_result = _minimize_function_multiple_x0(func_to_minimize, x0,
                                                     omega, bounds_in,
                                                     true_prices)

    returnval = _qlr_in(true_prices, omega, omega_cov) - minimize_result.fun

    # If the differrence above is not a number, I want the qlr_in(true_prices) to be worse.
    if np.isnan(returnval):
        returnval = np.inf
        logging.warning("""The differene between the true value and the
                        minimized function in qlr_stat is not a number""")
    elif returnval < 0:
        returnval = 0

    return tuple(true_prices) + (returnval,)


def qlr_sim(true_prices, omega, omega_cov, innov_dim, bounds, alpha=None):
    """
    Simulate the qlr_stat given the omega_estimates and covariance matrix, redrawing the error.

    It computes the erorr relevant for computing the moments projected onto (theta, pi) many times.

    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    true_prices : dict
    bounds : iterable of tuples
    alpha : scalar in (0,1), optional
        If alpha is none return all the draws otherwise return the (1-alpha) precentile.

    Returns
    ------
    scalar

    """
    bounds_in = [(bounds['phi']['min'], bounds['phi']['max']),
                 (bounds['pi']['min'], bounds['pi']['max']),
                 (bounds['theta']['min'], bounds['theta']['max'])]

    constraint_dict, init = compute_constraint_prices(omega=omega,
                                                      omega_cov=omega_cov,
                                                      bounds=bounds)
    low_bounds, high_bounds = np.array(bounds_in).T
    x0 = np.random.uniform(low_bounds, high_bounds)

    # If we violate the contraint, we want to always reject.
    if constraint_dict['fun'](true_prices, omega=omega) < 0:
        return tuple(true_prices) + (0,)

    cov_true_true = covariance_kernel(true_prices, true_prices,
                                      omega_cov=omega_cov, omega=omega)
    cov_params_true = partial(covariance_kernel, prices2=true_prices,
                              omega=omega, omega_cov=omega_cov)

    link_true = compute_link(true_prices, omega)
    # Draw the innovation for the moments
    innovations = stats.multivariate_normal.rvs(cov=cov_true_true, size=innov_dim)

    def qlr_in_star(prices, innov):

        if not np.all(np.isfinite(prices)):
            logging.warning(f"prices are {prices}")
            return np.inf

        try:
            link1 = compute_link(prices, omega)
            residual = (link1 - cov_params_true(prices) @
                        np.linalg.solve(cov_true_true, link_true))

            link_in = (residual + cov_params_true(prices) @
                       np.linalg.solve(cov_true_true, innov))

            cov_prices = covariance_kernel(prices, prices, omega_cov=omega_cov, omega=omega)

            if not np.all(np.isfinite(cov_prices)):
                logging.warn("The covariance kernel contains NaN")
                return np.inf

            # We do not need to pre-multiply by $T$ because we are using scaled versions of the covariances.
            return np.asscalar(link_in.T @ np.linalg.solve(cov_prices, link_in))

        except FloatingPointError and np.linalg.LinAlgError:
            # If we get a matrix algebra error in the previous expression we set the value of the link function to
            # infinity.
            logging.warn("There was either a Floating point error or a linear algebra error in qlr_in_star.")
            return np.inf

    with np.errstate(invalid='raise'):

        def minimized(innov):
            try:
                result = _minimize_function_multiple_x0(
                    lambda x: qlr_in_star(x, innov), x0, omega, bounds_in,
                    true_prices)

                return result.fun

            except FloatingPointError:
                logging.warn("There was a floating point error inside minimized.")
                return np.inf

        results_out = (_qlr_in(true_prices, omega, omega_cov) -
                       np.array([minimized(innov) for innov in innovations]))

    results = np.nan_to_num(results_out)

    if np.any(results < 0):
        logging.warning("""Some of the differences between the true and the
                        minimized value are negative.""")
        results[results <= 0] = 0

    if alpha is None:
        return results

    else:
        # We replace all of the error values with zero because if we a lot of them we want to reject. We do not
        # always reject because we are only redrawing part of the variation.
        returnval = np.percentile(results, 100 * (1 - alpha), interpolation='lower')

        if np.isnan(returnval):
            raise FloatingPointError("Returnval is not finite.")

        return tuple(true_prices) + (returnval,)


def compute_qlr_stats(omega, omega_cov, bounds, use_tqdm=True):
    """
    Compute the qlr statistics and organizes them into a dataframe.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estimates' covariance matrix.
    theta_dim : positive scalar, optional
        The number of grid points for the theta price.
    bounds : dict of dict
        The bounds for the structural parameters
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe

    """
    it = product(np.linspace(bounds['phi']['min'], bounds['phi']['max'],
                             bounds['phi']['dim']),
                 np.linspace(bounds['pi']['min'], bounds['pi']['max'],
                             bounds['pi']['dim']),
                 np.linspace(bounds['theta']['min'], bounds['theta']['max'],
                             bounds['theta']['dim']))

    qlr_stat_in = partial(qlr_stat, omega=omega, omega_cov=omega_cov)

    with Pool(8) as pool:
        if use_tqdm:
            draws = list(tqdm(pool.imap_unordered(qlr_stat_in, it),
                              total=bounds['pi']['dim'] *
                              bounds['theta']['dim'] * bounds['theta']['dim'],
                              leave=False))
        else:
            draws = list(pool.imap_unordered(qlr_stat_in, it))

    param_idx = ['phi', 'pi', 'theta']
    draws_df = pd.DataFrame.from_records(draws, columns=param_idx +
                                         ['qlr']).sort_values(by=param_idx)

    return draws_df


def compute_qlr_sim(omega, omega_cov, bounds, innov_dim=10, use_tqdm=True, alpha=.05):
    """
    Compute the qlr statistics and organizes them into a dataframe.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estimates' covariance matrix.
    bounds : dict of dict
        The bounds on the structural paramters
    innov_dim : scalar, optional
        The number of draws to inside the simulation.
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe

    """
    it = product(np.linspace(bounds['phi']['min'], bounds['phi']['max'],
                             bounds['phi']['dim']),
                 np.linspace(bounds['pi']['min'], bounds['pi']['max'],
                             bounds['pi']['dim']),
                 np.linspace(bounds['theta']['min'], bounds['theta']['max'],
                             bounds['theta']['dim']))

    qlr_sim_in = partial(qlr_sim, omega=omega, omega_cov=omega_cov,
                         bounds=bounds, innov_dim=innov_dim, alpha=alpha)

    param_idx = ['phi', 'pi', 'theta']
    with Pool(8) as pool:
        draws = []
        opts = {'total': bounds['pi']['dim'] * bounds['theta']['dim'] *
                bounds['theta']['dim'], 'leave': False}

        if use_tqdm:
            for draw in tqdm(pool.imap_unordered(qlr_sim_in, it), **opts):
                draws.append(draw)
                draws_df = pd.DataFrame.from_records(draws, columns=param_idx +
                                                     ['qlr']).sort_values(by=param_idx)
                filename1 = f"../results/qlr_draws_on_data_{innov_dim}"
                filename2 = "_smaller_region_flattened.tmp.json"""
                draws_df.to_json(filename1 + filename2)
        else:
            draws = list(pool.imap_unordered(qlr_sim_in, it))
            draws_df = pd.DataFrame.from_records(draws, columns=param_idx +
                                                 ['qlr']).sort_values(by=param_idx)

    return draws_df


def merge_draws_and_sims(qlr_stats, qlr_draws):
    """
    Merge the two sets using the parameter values as a multiindex.

    Parameters
    ---------
    qlr_stats : dataframe
    qlr_draws : dataframe

    Returns
    ------
    merged_values : dataframe

    """
    param_idx = ['phi', 'pi', 'theta']
    close_enough = np.allclose(qlr_stats[param_idx], qlr_draws[param_idx])

    if not close_enough:
        raise RuntimeError('The indices are not the same!!!')

    else:
        qlr_stats[param_idx] = qlr_draws[param_idx]

    merged_values = pd.merge(qlr_stats, qlr_draws, left_on=param_idx,
                             right_on=param_idx, suffixes=['_draws', '_stats'])

    return merged_values


def compute_strong_id(omega, omega_cov, bounds):
    """
    Compute the estimates and covariances for the strongly identified solution.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estiamtes' covariance matrix.
    bounds : dict of dict

    Returns
    -------
    rtn_prices : dict
        price estimates
    return_cov : dataframe
        Their covariance matrix

    """
    constraint_dict, init = compute_constraint_prices(omega=omega,
                                                      omega_cov=omega_cov,
                                                      bounds=bounds)

    bounds_in = [(bounds['phi']['min'], bounds['phi']['max']),
                 (bounds['pi']['min'], bounds['pi']['max']),
                 (bounds['theta']['min'], bounds['theta']['max'])]

    low_bounds, high_bounds = np.array(bounds_in).T
    x0 = np.random.uniform(low_bounds, high_bounds)

    func_to_minimize = partial(_qlr_in, omega=omega, omega_cov=omega_cov)

    minimize_result = _minimize_function_multiple_x0(func_to_minimize, x0=x0,
                                                     omega=omega, bounds=bounds_in,
                                                     true_prices=init)

    # minimize_result = minimize(lambda x: _qlr_in(x, omega, omega_cov,
    # ), x0=init, method='SLSQP', constraints=constraint_dict,
    # bounds=bounds_in) rtn_prices = minimize_result.x

    rtn_prices = minimize_result.x

    if not minimize_result['success']:
        logging.warning(minimize_result)

    if np.any(np.isnan(rtn_prices)):
        rtn_prices = init

    bread = compute_link_grad(rtn_prices, omega).T
    inner_cov = bread.T @ omega_cov @ bread

    outer_bread = compute_link_price_grad(rtn_prices, omega)
    outer_bread_inv = np.linalg.pinv(outer_bread.T @ outer_bread)

    # We use this ordering because pi is earlier than theta in the alphabet.
    names = compute_names()

    cov_arr = np.array(outer_bread_inv @ outer_bread.T @ inner_cov
                       @ outer_bread @ outer_bread_inv.T)
    return_cov = pd.DataFrame(cov_arr, columns=names,
                              index=names).sort_index(axis=0).sort_index(axis=1)

    return {name: val for name, val in zip(names, rtn_prices)}, return_cov


def estimate_params_strong_id(data, bounds, vol_estimates=None, vol_cov=None):
    """
    Estimate the model in one step.

    Paramters
    ---------
    data : ndarray
        Must contain rtn and vol columns.
    bounds : dict of dict
        bounds on the structural paramters
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

    price_estimates, price_cov = compute_strong_id(omega=estimates,
                                                   omega_cov=covariance,
                                                   bounds=bounds)
    estimates.update(price_estimates)
    covariance = covariance.merge(price_cov, left_index=True, right_index=True,
                                  how='outer').sort_index(axis=1).sort_index(axis=0)

    return estimates, covariance


def compute_qlr_reject(params, true_prices, innov_dim, bounds, alpha=None,
                       robust_quantile=True):
    """
    Compute the proportion rejected by the model.

    Paramters
    --------
    params: tuple of dict, dataframe
        The paramter estimates and their covariances.
    true_prices : iterable of length 2
        The prices under the null.
    innov_dim : positive scalar, optional
        The number of simulations inside the conditional simulation.
    bounds : dict of dict
        Bounds on the structural paramters.
    alpha : scalar in [0,1], optional
        The signficance level of the test.

    Returns
    ------
    qlr : scalar
        The QLR statistic
    qlr__quantile : scalar
        The weak identificaton robust conditional qlr quantile.

    """
    param_est, param_cov = params
    names = compute_names()
    omega = {name: val for name, val in param_est.items() if name not in names}
    omega_cov = param_cov.query('index not in @names').T.query('index not in @names').T

    qlr = qlr_stat(true_prices=true_prices, omega=omega, omega_cov=omega_cov,
                   bounds=bounds)[-1]

    if robust_quantile:
        qlr_quantile = qlr_sim(true_prices=true_prices, omega=omega,
                               alpha=alpha, omega_cov=omega_cov,
                               innov_dim=innov_dim, bounds=bounds)
        if alpha is None:
            return (qlr,) + tuple(qlr_quantile)
        else:
            return (qlr, qlr_quantile[-1])
    else:
        return qlr


def compute_robust_rejection(est_arr, true_params, bounds, alpha=.05,
                             innov_dim=100, use_tqdm=True,
                             robust_quantile=True):
    """
    Compute the proportion rejected by the model.

    Paramters
    --------
    est_arr: iterable of tuple of dict, dataframe
        The paramter estimates.
    true_params : dict
        The value to consider rejecting.
    bounds : dict of dict
        Bounds on the structural paramters.
    alpha : scalar in [0,1], optional
        The signficance level of the test.
    innov_dim : positive scalar, optional
        The number of simulations inside the conditional simulation.
    use_tqdm : bool, optional
        Whether to wrap the iterable using tqdm.
    robust_quantile : bool, optional
        Whether to compute and return the robust conditional QLR quantiles.

    Returns
    ------
    results : dataframe
        The QLR statistics, quantiles, and rejection proportions.

    """
    true_prices = [true_params[name] for name in compute_names()]

    qlr_reject_in = partial(compute_qlr_reject, true_prices=true_prices,
                            innov_dim=innov_dim, alpha=alpha,
                            robust_quantile=robust_quantile, bounds=bounds)

    with Pool(8) as pool:
        if use_tqdm:
            results = pd.DataFrame(list(tqdm(pool.imap_unordered(qlr_reject_in,
                                                                 est_arr),
                                             total=len(est_arr))))
        else:
            results = pd.DataFrame(list(pool.imap_unordered(qlr_reject_in, est_arr)))

    if robust_quantile:
        results.columns = ['qlr_stat', 'robust_qlr_qauntile']
        results['robust'] = results.loc[:, 'qlr_stat'] >= results.loc[:, 'robust_qlr_qauntile']
    else:
        results.columns = ['qlr_stat']

    results['standard'] = results.loc[:, 'qlr_stat'] >= stats.chi2.ppf(1 - alpha, df=len(true_prices))

    return results
