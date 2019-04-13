"""Ensures that we have the correct implmentation of the model's functions."""
import sympy as sym
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import volpriceinference as vl
pytestmark = pytest.mark.filterwarnings("ignore:.*U.*mode is deprecated:DeprecationWarning")


finite_floats = st.floats(min_value=-10, max_value=10)
positive_floats = st.floats(min_value=.1, max_value=10)
negative_floats = st.floats(min_value=-10, max_value=-.1)
phi_floats = st.floats(min_value=-.9, max_value=-.1)
log_floats = st.floats(min_value=-5, max_value=2)


@given(finite_floats, log_floats, positive_floats)
@settings(max_examples=20)
def test_A_func(logit_rho, log_scale, x):
    """Ensure the two A_func implementations return the same value."""
    val1 = sym.N(vl.volprice._A_func.replace(vl.logit_rho, logit_rho).replace(
        vl.log_scale, log_scale).replace(vl.volprice._x, x))
    val2 = vl.A_func(logit_rho=logit_rho, log_scale=log_scale, x=x)

    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(log_floats, log_floats, positive_floats)
@settings(max_examples=20)
def test_B_func(log_both, log_scale, x):
    """Ensure the two B_func implementations return the same value."""
    val1 = sym.N(sym.exp(log_both - log_scale) * (sym.log(vl.volprice._B_func_in.replace(
        vl.volprice._x, x).replace(vl.log_scale, log_scale))))
    val2 = vl.B_func(log_both=log_both, log_scale=log_scale, x=x)

    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(negative_floats, positive_floats, log_floats, log_floats, phi_floats, finite_floats)
@settings(max_examples=20)
def test_compute_gamma(pi, theta, log_both, log_scale, phi, psi):
    """Ensure the two implmentation fo the compute_gamma function return the same value."""
    val1 = sym.N(vl.volprice._gamma_sym.replace(vl.theta, theta).replace(vl.pi, pi).replace(
        vl.log_both, log_both)).replace(vl.log_scale, log_scale).replace(vl.phi, phi).replace(vl.psi, psi)
    val2 = vl.compute_gamma(pi=pi, theta=theta, log_both=log_both, log_scale=log_scale, phi=phi, psi=psi)

    if np.isfinite(val2):
        assert np.isclose(float(np.real(val1)), float(np.real(val2)), equal_nan=True, rtol=1e-3),\
            f"The two implementations return different values: {val1} and {val2}"


@given(phi_floats, negative_floats, positive_floats, finite_floats, finite_floats, log_floats,
       log_floats, finite_floats, finite_floats, finite_floats)
@settings(max_examples=20)
def test_link_total(phi, pi, theta, beta, gamma, log_both, log_scale, logit_rho, psi, zeta):
    """Ensure the two implementations have the same link functions."""
    args = {vl.volprice.phi: phi, vl.volprice.pi: pi, vl.volprice.theta: theta, vl.volprice.beta: beta,
            vl.volprice.gamma: gamma, vl.volprice.log_both: log_both,
            vl.volprice.log_scale: log_scale, vl.volprice.logit_rho: logit_rho, vl.volprice.psi: psi,
            vl.volprice.zeta: zeta}

    val1 = np.real(np.array(sym.N(vl.volprice._link_sym.xreplace(args))).astype(np.complex))
    val2 = vl.link_total(phi=phi, pi=pi, theta=theta, beta=beta, gamma=gamma, logit_rho=logit_rho,
                         log_both=log_both, log_scale=log_scale, psi=psi, zeta=zeta)

    if np.all(np.isfinite(val2)) and np.all(abs(val2) <= 1e4):
        assert np.allclose(np.ravel(val1), np.ravel(val2), rtol=1e-3, equal_nan=True), \
            f"The two implementations return different values: {val1} and {val2}"


@given(phi_floats, negative_floats, positive_floats, log_floats, log_floats, finite_floats, finite_floats)
@settings(deadline=1000, max_examples=10)
def test_link_gradient(phi, pi, theta, log_both, log_scale, logit_rho, psi):
    """Ensure the two implementations have the same link function gradient gradientsss."""
    args = {vl.volprice.phi: phi, vl.volprice.pi: pi, vl.volprice.theta: theta, vl.volprice.log_both: log_both,
            vl.volprice.log_scale: log_scale, vl.volprice.logit_rho: logit_rho, vl.volprice.psi: psi}

    val1 = np.real(np.array(sym.N(vl.volprice._link_grad_sym.xreplace(args))).astype(np.complex))
    val2 = vl.link_jacobian(phi=phi, pi=pi, theta=theta, logit_rho=logit_rho,
                            log_both=log_both, log_scale=log_scale, psi=psi)

    if np.all(np.isfinite(val2)) and np.all(abs(val2) <= 1e4):
        assert np.allclose(np.ravel(val1), np.ravel(val2), rtol=1e-3, equal_nan=True), \
            f"The two implementations return different values: {val1} and {val2}"
