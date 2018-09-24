"""This module contains the functions that are used to simulate and estimate the model and perform inference."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
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
_x, _y, beta, gamma, rho, scale, delta, psi, zeta = sym.symbols('_x _y beta gamma rho scale delta psi zeta')
theta, pi, phi = sym.symbols('theta pi phi')

# We define the link functions.
_psi_sym = (phi / sym.sqrt(scale * (1 + rho))) + (1 - phi**2) / 2 - (1 - phi**2) * theta
_theta_sym = sym.solveset(psi - _psi_sym, theta).args[0]
_A_func = rho * _x / (1 + scale * _x)
_C_func = psi * _x + ((1 - phi**2) / 2) * _x**2
_B_func_in = 1 + scale * _x
_beta_sym = (_A_func.replace(_x, pi + _C_func.replace(_x, theta - 1)) - _A_func.replace(
    _x, pi + _C_func.replace(_x, theta)))
_gamma_sym = delta * (sym.log(_B_func_in.replace(_x, pi + _C_func.replace(_x, theta - 1))) -
                      sym.log(_B_func_in.replace(_x, pi + _C_func.replace(_x, theta))))

# We create the link functions.
compute_gamma = sym.lambdify((delta, pi, rho, scale, theta, phi), _gamma_sym.replace(psi, _psi_sym),
                             modules='numpy')
compute_beta = sym.lambdify((pi, rho, scale, theta, phi), _beta_sym.replace(psi, _psi_sym), modules='numpy')
compute_psi = sym.lambdify((rho, scale, theta, phi), _psi_sym, modules='numpy')

# We create a function to compute theta
compute_theta = sym.lambdify((psi, rho, scale, zeta), _theta_sym.replace(phi, -sym.sqrt(1 - zeta)),
                             modules='numpy')
_pi_from_gamma = sym.solveset(gamma / delta - (_B_func_in.replace(_x, pi + _C_func.replace(_x, theta - 1)) /
                                               (_B_func_in.replace(_x, pi + _C_func.replace(_x, theta)))),
                              pi).args[0].args[0].simplify()

compute_pi = sym.lambdify((delta, gamma, psi, rho, scale, theta, zeta), _pi_from_gamma.replace(
    phi, -sym.sqrt(1 - zeta)), modules='numpy')

_link_sym = sym.simplify(sym.Matrix([beta - _beta_sym, gamma - _gamma_sym, psi - _psi_sym, 1 - (zeta + phi**2)]))

_link_in0 = sym.lambdify((phi, beta, delta, gamma, psi, rho, scale, zeta), _link_sym[-1],
                         modules='numpy')
_link_in1 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta), _link_sym,
                         modules='numpy')
_link_in2 = sym.lambdify((phi, theta, beta, delta, gamma, psi, rho, scale, zeta), _link_sym[-2:],
                         modules='numpy')
_link_in3 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta), _link_sym[1:],
                         modules='numpy')


def compute_constraint_prices(omega, case):
    """Compute the slackness in the nonlinear constraint."""
    phi_init = -np.sqrt(1 - omega['zeta']) if omega['zeta'] < 1 else 0
    theta_init = compute_theta(psi=omega['psi'], scale=omega['scale'], rho=omega['scale'], zeta=omega['zeta'])
    pi_init = compute_pi(delta=omega['delta'], gamma=omega['gamma'], psi=omega['psi'], scale=omega['scale'],
                         rho=omega['scale'], zeta=omega['zeta'], theta=theta_init)

    if case == 0:
        prices_init = np.nan_to_num([phi_init])
    elif case == 2:
        prices_init = np.nan_to_num([phi_init, theta_init])
    else:
        prices_init = np.nan_to_num([phi_init, pi_init, theta_init])

    constraint_in = partial(constraint, omega=omega, case=case)
    constraint_dict = {'type': 'ineq', 'fun': constraint_in}

    if constraint_in(prices_init) < 0:
        prices_init[1] = -.1

    if constraint_in(prices_init) < 0:
        prices_init[2] = .1

    return constraint_dict, prices_init


def compute_names(case):
    """Compute the names used in each case."""
    if case == 0:
        names = ['phi']
    elif case == 2:
        names = ['phi', 'theta']
    else:
        names = ['phi', 'pi', 'theta']

    return names


def compute_bounds(case):
    """Compute the bounds for the structural parameters."""
    if case == 0:
        bounds = [(-.9, .9)]
    elif case == 2:
        bounds = [(-.9, .9), (-10, None)]
    else:
        bounds = [(-.9, .9), (None, 10), (-10, None)]

    return bounds


def compute_link(prices, omega, case):
    """Compute the link function."""
    if case == 0:
        compute_link_in = _link_in0
    elif case == 1:
        compute_link_in = _link_in1
    elif case == 2:
        compute_link_in = _link_in2
    elif case == 3:
        compute_link_in = _link_in3
    else:
        raise NotImplementedError

    return compute_link_in(*prices, **omega)


_link_grad_sym = sym.simplify(sym.Matrix([_link_sym.jacobian([beta, delta, gamma, psi, rho, scale, zeta])]))
_link_grad_in0 = sym.lambdify((phi, beta, delta, gamma, psi, rho, scale, zeta), _link_grad_sym[-1, :],
                              modules='numpy')
_link_grad_in1 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta), _link_grad_sym,
                              modules='numpy')
_link_grad_in2 = sym.lambdify((phi, theta, beta, delta, gamma, psi, rho, scale, zeta), _link_grad_sym[-2:, :],
                              modules='numpy')
_link_grad_in3 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta),
                              _link_grad_sym[1:, :], modules='numpy')


def compute_link_grad(prices, omega, case):
    """Compute the gradient of the link function with respect to the reduced-form paramters."""
    if case == 0:
        compute_link_grad_in = _link_grad_in0
    elif case == 1:
        compute_link_grad_in = _link_grad_in1
    elif case == 2:
        compute_link_grad_in = _link_grad_in2
    elif case == 3:
        compute_link_grad_in = _link_grad_in3
    else:
        raise NotImplementedError

    return compute_link_grad_in(*prices, **omega)


_link_price_grad_sym = sym.simplify(sym.Matrix([_link_sym.jacobian([phi, pi, theta])]))

_link_price_grad_in0 = sym.lambdify((phi, beta, delta, gamma, psi, rho, scale, zeta), _link_price_grad_sym[-1, 0],
                                    modules='numpy')
_link_price_grad_in1 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta),
                                    _link_price_grad_sym, modules='numpy')
_link_price_grad_in2 = sym.lambdify((phi, theta, beta, delta, gamma, psi, rho, scale, zeta),
                                    _link_price_grad_sym[-2:, [0, 2]], modules='numpy')
_link_price_grad_in3 = sym.lambdify((phi, pi, theta, beta, delta, gamma, psi, rho, scale, zeta),
                                    _link_price_grad_sym[1:, :], modules='numpy')


def compute_link_price_grad(prices, omega, case):
    """Compute the gradient of the link function with respect to the structural parameters."""
    if case == 0:
        compute_link_price_grad_in = _link_price_grad_in0
    elif case == 1:
        compute_link_price_grad_in = _link_price_grad_in1
    elif case == 2:
        compute_link_price_grad_in = _link_price_grad_in2
    elif case == 3:
        compute_link_price_grad_in = _link_price_grad_in3
    else:
        raise NotImplementedError

    return np.atleast_2d(compute_link_price_grad_in(*prices, **omega))


_constraint_sym = sym.simplify(_B_func_in.replace(_x, pi + _C_func.replace(_x, theta - 1)))
_constraint1 = sym.lambdify((phi, pi, theta, psi, rho, scale), _constraint_sym, modules='numpy')


def constraint(prices, omega, case=1):
    """Compute the constraint implied by logarithm's argument in the second link function being postiive."""
    if case == 0 or case == 2:
        return 1

    constraint1 = _constraint1(*prices, psi=omega['psi'], rho=omega['rho'], scale=omega['scale'])

    return constraint1


_mean = (rho * _x + scale * delta)
_var = 2 * scale * rho * _x + scale**2 * delta
row1 = _y - _mean
row3 = _y**2 - (_mean**2 + _var)
_vol_moments = sym.Matrix([row1, row1 * _x, row3, row3 * _x, row3 * _x**2])


compute_vol_moments = sym.lambdify([_x, _y, rho, scale, delta], _vol_moments, modules='numpy')
compute_vol_moments_grad = sym.lambdify([_x, _y, rho, scale, delta], _vol_moments.jacobian([delta, rho, scale])[:],
                                        modules='numpy')

# I now define the covariance kernel.
omega_cov = sym.MatrixSymbol('omega_cov', _link_grad_sym.shape[1], _link_grad_sym.shape[1])
pi1, pi2, theta1, theta2, phi1, phi2 = sym.symbols("""pi1 pi2 theta1 theta2 phi1 phi2""")


_link_grad_left = _link_grad_sym.replace(pi, pi1).replace(theta, theta1).replace(phi, phi1)
_link_grad_right = _link_grad_sym.replace(pi, pi2).replace(theta, theta2).replace(phi, phi2)


_cov_kernel_in0 = sym.lambdify((phi1, phi2, psi, beta, delta, gamma, rho, scale, zeta, omega_cov),
                               _link_grad_left[-1, :] * omega_cov * _link_grad_right[-1, :].T, modules='numpy')

_cov_kernel_in1 = sym.lambdify((phi1, pi1, theta1, phi2, pi2, theta2, psi, beta, delta, gamma, rho, scale, zeta,
                                omega_cov), _link_grad_left * omega_cov * _link_grad_right.T, modules='numpy')

_cov_kernel_in2 = sym.lambdify((phi1, theta1, phi2, theta2, psi, beta, delta, gamma, rho, scale, zeta, omega_cov),
                               _link_grad_left[-2:, :] * omega_cov * _link_grad_right[-2:, :].T, modules='numpy')

_cov_kernel_in3 = sym.lambdify((phi1, pi1, theta1, phi2, pi2, theta2, psi, beta, delta, gamma, rho, scale, zeta,
                                omega_cov), _link_grad_left[1:, :] * omega_cov * _link_grad_right[1:, :].T,
                               modules='numpy')


def covariance_kernel(prices1, prices2, omega, omega_cov, case):
    """Compute the covarinace of the implied gaussian process as a function of the structural paramters."""
    if case == 0:
        covariance_kernel_in = _cov_kernel_in0
    elif case == 1:
        covariance_kernel_in = _cov_kernel_in1
    elif case == 2:
        covariance_kernel_in = _cov_kernel_in2
    elif case == 3:
        covariance_kernel_in = _cov_kernel_in3
    else:
        raise NotImplementedError

    return covariance_kernel_in(*prices1, *prices2, omega_cov=omega_cov, **omega)


def simulate_autoregressive_gamma(delta=1, rho=0, scale=1, initial_point=None, time_dim=100,
                                  start_date='2000-01-01'):
    """
    Provide draws from the ARG(1) process of Gourieroux & Jaisak.

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

    draws = _simulate_autoregressive_gamma(delta=delta, rho=rho, scale=scale, initial_point=initial_point,
                                           time_dim=time_dim)

    draws = pd.DataFrame(draws, pd.date_range(start=start_date, freq='D', periods=time_dim))

    return draws


def simulate_data(theta=1, pi=0, rho=0, scale=1, delta=1, phi=0, initial_point=None, time_dim=100,
                  start_date='2000-01-01', case=1):
    """
    Take the reduced-form paramters and risk prices and returns the data.

    Parameters
    --------
    theta: scalar
    pi : scalar
    rho : scalar
        persistence
    scale : positive scalar
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
    vol_data = simulate_autoregressive_gamma(rho=rho, scale=scale, delta=delta, initial_point=initial_point,
                                             start_date=pd.to_datetime(start_date) - pd.Timedelta('1 day'),
                                             time_dim=time_dim + 1)

    gamma_val = compute_gamma(rho=rho, scale=scale, delta=delta, pi=pi, theta=theta, phi=phi)
    beta_val = compute_beta(rho=rho, scale=scale, pi=pi, theta=theta, phi=phi)
    psi_val = compute_psi(rho=rho, scale=scale, theta=theta, phi=phi)

    if case == 1:
        price_in = (phi, pi, theta)
        if constraint(prices=price_in, omega={'psi': psi_val, 'rho': rho, 'scale': scale}, case=case) < 0:
            raise ValueError(f"""The set of paramters given conflict with each other. No process exists with those
                             paramters. You might want to make the volatility price smaller in magnitude.""")

    mean = gamma_val + beta_val * vol_data.shift(1) + psi_val * vol_data
    var = (1 - phi**2) * vol_data

    draws = mean + np.sqrt(var) * pd.DataFrame(_threadsafe_gaussian_rvs(var.size), index=vol_data.index)
    data = pd.concat([vol_data, draws], axis=1).dropna()
    data.columns = ['vol', 'rtn']

    return data


def vol_moments(vol_data, delta, rho, scale):
    """Compute the moments of the volatility process."""
    x = vol_data.values[:-1]
    y = vol_data.values[1:]

    return pd.DataFrame(np.squeeze(compute_vol_moments(x, y, rho, scale, delta)).T)


def vol_moments_grad(vol_data, delta, rho, scale):
    """Compute the jacobian of the volatility moments."""
    x = vol_data.values[:-1]
    y = vol_data.values[1:]

    delta_mom = [np.mean(val) for val in compute_vol_moments_grad(x, y, rho, scale, delta)]

    return pd.DataFrame(np.reshape(delta_mom, (len(delta_mom) // 3, 3)), columns=['delta', 'rho', 'scale'])


def compute_init_constants(vol_data):
    r"""
    Compute some guesses for the volatlity paramters that we can use to initialize the optimization.

    From the model, we know that intercept = $ scale * \delta$. We also know that the average error variance
    equals $ c^2 \delta * (2 \rho / 1 - \rho) + 1)$. Consequently, $c = error\_var / ( intercept * (2 \rho / 1 -
    \rho) + 1))$, and  $\delta = \text{intercept} / c$.

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
    init_constants['scale'] = error_var / (intercept * (2 * persistence / (1 - persistence) + 1))
    init_constants['delta'] = intercept / init_constants['scale']

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


def compute_vol_gmm(vol_data, init_constants, bounds=None, options=None):
    """
    Use GMM to compute the volatility paramters and their asymptotic covariance matrix.

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
        bounds = [(0, None), (0, 1), (0, None)]

    if options is None:
        options = {'maxiter': 200}

    init_constants = OrderedDict(sorted(init_constants.items(), key=lambda t: t[0]))

    x0 = list(init_constants.values())

    initial_result = minimize(lambda x: compute_mean_square(x, vol_data, vol_moments), x0=x0,
                              options=options, bounds=bounds)

    if not initial_result['success']:
        logging.warning(initial_result)

    weight_matrix = np.linalg.pinv(vol_moments(vol_data, *initial_result.x).cov())

    final_result = minimize(lambda x: compute_mean_square(x, vol_data, vol_moments, weight_matrix),
                            x0=initial_result.x, method="L-BFGS-B", bounds=bounds, options=options)

    if not final_result['success']:
        logging.warning(final_result)

    estimates = {key: val for key, val in zip(init_constants.keys(), final_result.x)}

    weight_matrix = np.linalg.pinv(vol_moments(vol_data, **estimates).cov())
    moment_derivative = vol_moments_grad(vol_data, **estimates)
    cov = pd.DataFrame(np.linalg.pinv(moment_derivative.T @ weight_matrix @ moment_derivative),
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
    """Estimate the scaled covariance paramter."""
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
    Estimate the model in one step.

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

    omega_names = ['beta', 'gamma', 'delta', 'zeta', 'psi', 'rho', 'scale']

    omega_cov = reduced_form_cov.loc[omega_names, omega_names].sort_index(axis=0).sort_index(axis=1)

    return {name: reduced_form_estimates[name] for name in omega_names}, omega_cov


def _qlr_in(prices, omega, omega_cov, case):

    link_in = np.ravel(compute_link(prices=prices, omega=omega, case=case))

    cov_pi = covariance_kernel(prices, prices, omega_cov=omega_cov, omega=omega, case=case)

    returnval = np.asscalar(link_in.T @ np.linalg.solve(cov_pi, link_in))

    if np.isnan(returnval):
        returnval = np.inf

    return returnval


def qlr_stat(true_prices, omega, omega_cov, bounds=None, case=1):
    """
    Compute the qlr_stat given the omega_estimates and covariance matrix.

    Paramters
    --------
    omega : dict
        Paramter estimates
    omega_cov : dataframe
        omega's covariance matrix.
    true_prices : dict
    bounds : iterable of tuples, optional

    Returns
    ------
    scalar

    """
    constraint_in = partial(constraint, omega=omega, case=case)
    constraint_dict = {'type': 'ineq', 'fun': constraint_in}

    # If we violate the contraint, we want to always reject.
    if constraint_in(true_prices) < 0:
        return tuple(true_prices) + (np.inf,)

    bounds = bounds if bounds is not None else compute_bounds(case)

    minimize_result = minimize(lambda x: _qlr_in(x, omega, omega_cov, case=case), x0=true_prices, method='SLSQP',
                               constraints=constraint_dict, bounds=bounds)

    if not minimize_result['success']:
        logging.warning(minimize_result)

    returnval = _qlr_in(true_prices, omega, omega_cov, case=case) - minimize_result.fun

    # If the differrence above is not a number, I want the qlr_in(true_prices) to be worse.
    if np.isnan(returnval):
        returnval = np.inf
    elif returnval < 0:
        returnval = 0

    return tuple(true_prices) + (returnval,)


def qlr_sim(true_prices, omega, omega_cov, innov_dim=10, alpha=.05, bounds=None, case=1):
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
    bounds : iterable of tuples, optional

    Returns
    ------
    scalar

    """
    constraint_in = partial(constraint, omega=omega, case=case)
    constraint_dict = {'type': 'ineq', 'fun': constraint_in}

    # If we violate the contraint, we want to always reject.
    if constraint_in(true_prices) < 0:
        return tuple(true_prices) + (0,)

    cov_true_true = covariance_kernel(true_prices, true_prices, omega_cov=omega_cov, omega=omega, case=case)
    link_true = np.ravel(compute_link(true_prices, omega, case=case))

    def cov_params_true(prices):
        return covariance_kernel(prices, true_prices, omega_cov=omega_cov, omega=omega, case=case)

    def project_resid(prices):
        link_in = np.ravel(compute_link(prices, omega, case=case))

        return link_in - cov_params_true(prices) @ np.linalg.solve(cov_true_true, link_true)

    # Draw the innovation for the moments
    innovations = stats.multivariate_normal.rvs(mean=np.zeros(link_true.size), size=innov_dim)

    def link_star(prices, innov):
        returnval = np.ravel(project_resid(prices) + cov_params_true(prices) @
                             np.linalg.solve(np.linalg.cholesky(cov_true_true), np.ravel(innov)))

        if not np.any(np.isfinite(returnval)):
            np.place(returnval, ~np.isfinite(returnval), np.inf)
            return returnval
        else:
            return returnval

    def qlr_in_star(prices, innov):
        link_in = link_star(prices=prices, innov=innov)
        cov_prices = covariance_kernel(prices, prices, omega_cov=omega_cov, omega=omega, case=case)

        try:
            # We do not need to pre-multiply by $T$ because we are using scaled versions of the covariances.
            returnval = np.asscalar(link_in.T @ np.linalg.lstsq(cov_prices, link_in, rcond=None)[0])
        except ValueError:
            # If we get a matrix algebra error in the previous expression we set the value of the link function to
            # infinity.
            returnval = np.inf

        return returnval

    constraint_dict, prices_init = compute_constraint_prices(omega, case)
    bounds = bounds if bounds is not None else compute_bounds(case)

    results = [qlr_in_star(true_prices, innov=innov) - minimize(lambda x: qlr_in_star(x, innov=innov),
                                                                x0=prices_init, method='SLSQP',
                                                                constraints=constraint_dict, bounds=bounds).fun
               for innov in innovations]

    # We replace all of the error values with zero because if we a lot of them we want to reject.
    # We do not always reject because we are only redrawing part of the variation.
    returnval = np.percentile([val if np.isfinite(val) else 0 for val in results], 100 * (1 - alpha))

    return tuple(true_prices) + (returnval,)


def compute_qlr_stats(omega, omega_cov, theta_dim=20, pi_dim=20, pi_min=-20, pi_max=0, theta_min=0,
                      theta_max=2, use_tqdm=True, case=1):
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
    pi_dim : positive scalar, optional
        The number of grid points for the pi price.
    pi_min : scalar, optional
        The minimum grid point for the volatility price
    pi_max : scalar, optional
        The maximum grid point for the volatility price
    theta_min : scalar, optional
        The minimum grid point for the equity price
    theta_max : scalar, optional
        The maximum grid point for the equity price
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe

    """
    it = product(np.linspace(-1, 1, phi_dim), np.linspace(pi_min, pi_max, pi_dim), np.linspace(theta_min,
                                                                                                  theta_max,
                                                                                                  theta_dim))

    qlr_stat_in = partial(qlr_stat, omega=omega, omega_cov=omega_cov, case=case)

    with Pool(8) as pool:
        if use_tqdm:
            draws = list(tqdm.tqdm_notebook(pool.imap_unordered(qlr_stat_in, it), total=pi_dim * theta_dim,
                                            leave=False))
        else:
            draws = list(pool.imap_unordered(qlr_stat_in, it))

    draws_df = pd.DataFrame.from_records(draws, columns=['theta', 'pi', 'qlr'])

    return draws_df.pivot(index='pi', columns='theta', values='qlr').sort_index(axis=0).sort_index(axis=1)


def compute_qlr_sim(omega, omega_cov, theta_dim=20, pi_dim=20, pi_min=-20, pi_max=0, theta_min=0,
                    theta_max=2, innov_dim=10, use_tqdm=True, alpha=.05, case=1):
    """
    Compute the qlr statistics and organizes them into a dataframe.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estimates' covariance matrix.
    theta_dim : positive scalar, optional
        The number of grid points for the equity price.
    pi_dim : positive scalar, optional
        The number of grid points for the vol price.
    pi_min : scalar, optional
        The minimum grid point for the volatility price
    pi_max : scalar, optional
        The maximum grid point for the volatility price
    theta_min : scalar, optional
        The minimum grid point for the equity price
    theta_max : scalar, optional
        The maximum grid point for the equity price
    innov_dim : scalar, optional
        The number of draws to inside the simulation.
    use_tqdm : bool, optional
        Whehter to use tqdm.

    Returns
    ------
    draws_df : dataframe

    """
    it = product(np.linspace(theta_min, theta_max, theta_dim), np.linspace(pi_min, pi_max, pi_dim))

    qlr_sim_in = partial(qlr_sim, omega=omega, omega_cov=omega_cov, innov_dim=innov_dim, alpha=alpha)

    with Pool(8) as pool:
        if use_tqdm:
            draws = list(tqdm.tqdm_notebook(pool.imap_unordered(qlr_sim_in, it), total=pi_dim * theta_dim,
                                            leave=False))
        else:
            draws = list(pool.imap_unordered(qlr_sim_in, it))

    draws_df = pd.DataFrame.from_records(draws, columns=['theta', 'pi', 'qlr'])

    return draws_df.pivot(index='pi', columns='theta', values='qlr').sort_index(axis=0).sort_index(axis=1)


def compute_strong_id(omega, omega_cov, bounds=None, case=1):
    """
    Compute the estimates and covariances for the strongly identified solution.

    Paramters
    --------
    omega : dict
        estimates
    omega_cov : dataframe
        estiamtes' covariance matrix.
    bounds : iterable of tuples, optional

    Returns
    -------
    rtn_prices : dict
        price estimates
    return_cov : dataframe
        Their covariance matrix

    """
    constraint_dict, prices_init = compute_constraint_prices(omega, case)
    bounds = bounds if bounds is not None else compute_bounds(case)

    minimize_result = minimize(lambda x: _qlr_in(x, omega, omega_cov, case=case), x0=prices_init, method='SLSQP',
                               constraints=constraint_dict, bounds=bounds)
    rtn_prices = minimize_result.x

    if not minimize_result['success']:
        logging.warning(minimize_result)

    if np.any(np.isnan(rtn_prices)):
        rtn_prices = prices_init

    bread = compute_link_grad(rtn_prices, omega, case=case).T
    inner_cov = bread.T @ omega_cov @ bread

    outer_bread = compute_link_price_grad(rtn_prices, omega, case=case)
    outer_bread_inv = np.linalg.pinv(outer_bread.T @ outer_bread)

    # We use this ordering because pi is earlier than theta in the alphabet.
    names = compute_names(case)

    return_cov = pd.DataFrame(outer_bread_inv @ outer_bread.T @ inner_cov @ outer_bread @ outer_bread_inv.T,
                              columns=names, index=names).sort_index(axis=0).sort_index(axis=1)

    return {name: val for name, val in zip(names, rtn_prices)}, return_cov


def estimate_params_strong_id(data, vol_estimates=None, vol_cov=None, case=1):
    """
    Estimate the model in one step.

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

    price_estimates, price_cov = compute_strong_id(omega=estimates, omega_cov=covariance, case=case)
    # Then we compute the reduced form paramters.
    estimates.update(price_estimates)
    covariance = covariance.merge(price_cov, left_index=True, right_index=True,
                                  how='outer').sort_index(axis=1).sort_index(axis=0)

    return estimates, covariance


def compute_qlr_reject(params, true_prices, innov_dim, alpha, robust_quantile=True, case=1):
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
    names = ['theta', 'phi', 'pi']
    omega = {name: val for name, val in param_est.items() if name not in names}
    omega_cov = param_cov.query('index not in @names').T.query('index not in @names').T

    qlr = qlr_stat(true_prices, omega, omega_cov, case=case)[-1]

    if robust_quantile:
        qlr_quantile = qlr_sim(true_prices, omega, omega_cov, innov_dim=innov_dim, alpha=alpha, case=case)[-1]
        return qlr, qlr_quantile
    else:
        return qlr


def compute_robust_rejection(est_arr, true_params, alpha=.05, innov_dim=100, use_tqdm=True, robust_quantile=True,
                             case=1):
    """
    Compute the proportion rejected by the model.

    Paramters
    --------
    est_arr: iterable of tuple of dict, dataframe
        The paramter estimates.
    true_params : dict
        The value to consider rejecting.
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
    true_prices = [true_params[name] for name in compute_names(case)]

    qlr_reject_in = partial(compute_qlr_reject, true_prices=true_prices, innov_dim=innov_dim, alpha=alpha,
                            robust_quantile=robust_quantile)

    with Pool(8) as pool:
        if use_tqdm:
            results = pd.DataFrame(list(tqdm.tqdm_notebook(pool.imap_unordered(qlr_reject_in, est_arr),
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
