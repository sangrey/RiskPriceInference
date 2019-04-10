import pytest
import numpy as np
from hypothesis import given, strategies as st
import sympy as sym


def test_packages():
    """ Ensure I can import the package. """
    import volpriceinference

finite_floats = st.floats(min_value=-10, max_value=10)
positive_floats = st.floats(min_value=.1, max_value=10)
negative_floats = st.floats(min_value=-10, max_value=-.1)
phi_floats = st.floats(min_value=-.9, max_value=-.1)

@given(finite_floats, finite_floats, positive_floats)
def test_A_func(logit_rho, log_scale, x):
    """ Ensure the two A_func implementations return the same value. """
    import volpriceinference as vl
    
    val1 = sym.N(vl.volprice._A_func.replace(vl.logit_rho, logit_rho).replace(
        vl.log_scale, log_scale).replace(vl.volprice._x, x))
    val2 = vl.A_func(logit_rho=logit_rho, log_scale=log_scale, x=x)
    
    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(finite_floats, finite_floats, positive_floats)
def test_B_func(log_both, log_scale, x):
    """ Ensure the two B_func implementations return the same value. """
    import volpriceinference as vl
    
    val1 = sym.N(sym.exp(log_both - log_scale) * (sym.log(vl.volprice._B_func_in.replace(
        vl.volprice._x, x).replace(vl.log_scale, log_scale))))
    val2 = vl.B_func(log_both=log_both, log_scale=log_scale, x=x)

    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(negative_floats, positive_floats, finite_floats, finite_floats, phi_floats, finite_floats)
def test_link2(pi, theta, log_both, log_scale, phi, psi):
    """ Ensure the two implmentation fo the link2 function return the same value. """
    import volpriceinference as vl
    
    val1 = sym.N(vl.volprice._gamma_sym.replace(vl.theta, theta).replace(vl.pi, pi).replace(
        vl.log_both, log_both)).replace(vl.log_scale, log_scale).replace(vl.phi, phi).replace(vl.psi, psi)
    val2 = vl.link2(pi=pi, theta=theta, log_both=log_both, log_scale=log_scale, phi=phi, psi=psi)

    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(phi_floats, negative_floats, positive_floats, finite_floats, finite_floats, finite_floats, 
       finite_floats, finite_floats, finite_floats, finite_floats)
def test_link_total(phi, pi, theta, beta, gamma, log_both, log_scale, logit_rho, psi, zeta):
    
    import volpriceinference as vl
    args = {vl.volprice.phi: phi, vl.volprice.pi: pi, vl.volprice.theta : theta,vl.volprice.beta: beta, 
            vl.volprice.gamma :gamma, vl.volprice.log_both :log_both, 
            vl.volprice.log_scale: log_scale, vl.volprice.logit_rho: logit_rho, vl.volprice.psi : psi,
            vl.volprice.zeta: zeta}
    
    val1 = np.real(np.array(sym.N(vl.volprice._link_sym.xreplace(args))).astype(np.complex))
    val2 = vl.link_total(phi=phi, pi=pi, theta=theta, beta=beta, gamma=gamma, logit_rho=logit_rho, 
                         log_both=log_both, log_scale=log_scale, psi=psi, zeta=zeta)
    
    if np.all(np.isfinite(val2)) and np.all(abs(val2) <= 1e4):
        assert np.allclose(np.ravel(val1), np.ravel(val2), rtol=1e-3, equal_nan=True), \
            f"The two implementations return different values: {val1} and {val2}"

