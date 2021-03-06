#include <stdexcept>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "pybind11/operators.h"
#include <pybind11/iostream.h>
#include <arma_wrapper.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;
using stream_redirect = py::call_guard<py::scoped_ostream_redirect>;                                               
using aw::dmat;
using aw::dvec;

double logistic(double x) {

    return 1.0 / (1 + std::exp(-x));
}

double logit(double x) {

    return std::log(x)  - std::log1p(-x); 
}


/* 
 * Initializes a random generator in a thread_save way to be used globally througout the entire library. 
 */
inline
std::mt19937_64 initialize_mt_generator() { 

    constexpr size_t state_size = std::mt19937_64::state_size * std::mt19937_64::word_size / 32;
    std::vector<uint32_t> random_data(state_size);
    std::random_device source{}; 
    std::generate(random_data.begin(), random_data.end(), std::ref(source));
    std::seed_seq seeds(random_data.begin(), random_data.end());
    std::mt19937_64 engine(seeds);
    
    return engine; 

}

std::vector<double> simulate_autoregressive_gamma(double delta, double rho, double scale, size_t time_dim, double
        initial_point) {
    
    assert (time_dim > 1);

    std::vector<double> draws;
    draws.reserve(time_dim+1);

    /* We start at the initial point. */
    draws.push_back(initial_point);
    auto generator = initialize_mt_generator();

    for(size_t idx=0; idx<time_dim; ++idx) {
        
        /* The conditional distribution of an ARG(1) process is non-centered Gamma, which has a representation as
         * a Poisson mixture of Gamma */
          std::poisson_distribution<int> poi_dist(rho * draws.back() / scale);
          int latent_var = poi_dist(generator);
          std::gamma_distribution<double> gamma_dist(delta + latent_var, scale);
        /* I force the draws to be bounded away from zero. Otherwise we can get stuck there.*/
          draws.push_back(std::max(gamma_dist(generator), 1e-5));
    }

    std::vector<double> return_draws(draws.begin() + 1, draws.end());

    return return_draws;
}

std::vector<double> threadsafe_gaussian_rvs(size_t time_dim) {

    std::normal_distribution<double> dist(0,1);
    auto generator = initialize_mt_generator();

    std::vector<double> return_draws(time_dim);

    for (auto& x : return_draws) {
        x = dist(generator);
    }
    
    return return_draws;
}

double A_func(double x, double logit_rho, double log_scale) {

    double scale =  std::exp(log_scale);
    double rho = logistic(logit_rho); 

    double val = rho * x / (1 + scale * x);

    return val;

}

double A_diff1(double x, double logit_rho, double log_scale) {
    /* Compute the derivatiive of the A function with respecto to x. */
    
    double scale =  std::exp(log_scale);
    double rho = logistic(logit_rho); 
    double numerator = rho * (1 + scale * x) - scale * rho * x;
    double denominator = (1 + scale * x) * (1 + scale * x);

    return numerator / denominator;

}

double A_diff2(double x, double logit_rho, double log_scale) {
    /* Compute the derivatiive of the A function with respecto to logit_rho. */
    
    double scale =  std::exp(log_scale);

    /* The derivative of the logistic function satisfies f'(x) = f(x) ( 1 - f(x))  */
    double numerator = x * std::exp(-1 * logit_rho);
    double denominator_in = 1 + std::exp(-1 * logit_rho); 
    double val = numerator / (denominator_in * denominator_in * (1 + scale * x)); 

    return val;

}

double A_diff3(double x, double logit_rho, double log_scale) {
    /* Compute the derivatiive of the A function with respecto to log_scale. */
    
    double rho = logistic(logit_rho);

    double numerator =  rho * x * std::exp(log_scale) * x;
    double denominator_in =  1 + std::exp(log_scale) * x;

    return -1.0 * numerator / (denominator_in * denominator_in);

}


double B_func(double x, double log_both, double log_scale) { 
    /* Compute the B_func in the accompanying paper. */

    double delta = std::exp(log_both - log_scale);
    double scale =  std::exp(log_scale);

    return delta * std::log(1 + scale * x);
}

double B_diff1(double x, double log_both, double log_scale) {
    /* Compute the derivative of the B_func with respect to the first argument. */

    double delta = std::exp(log_both - log_scale);
    double scale =  std::exp(log_scale);
   
    return (delta / (1 + scale * x)) * scale;

}

double B_diff2(double x, double log_both, double log_scale) { 
    /* Compute the derivative of B w.r.t. log_both */

    return B_func(x, log_both, log_scale);
}

double B_diff3(double x, double log_both, double log_scale) {
    /* Compute the derivative of B w.r.t. log_scale */

    // The std::exp(log_both) term arises because the exp(log_scale) terms cancel. 
    double part1 = x / (1 + std::exp(log_scale) * x) * std::exp(log_both);

    double part2 = -1.0 * B_func(x, log_both, log_scale);

    return part1 + part2;
}

double C_func(double x, double phi, double psi) {

    double val = psi * x - ((1 - phi * phi) / 2.0) * x * x;
    return val;

}

double C_diff3(double x) {

    return x;

}

double compute_beta(double pi, double phi, double theta, double logit_rho, double log_scale, double psi) {

    double val1 = A_func(pi + C_func(theta - 1, phi, psi), logit_rho, log_scale);
    double val2 = A_func(pi + C_func(theta, phi, psi), logit_rho, log_scale);

    return val1 - val2;

}

std::tuple<double, double, double> beta_gradient(double pi, double phi, double theta, double logit_rho, 
        double log_scale, double psi) {

    double d_logit_rho = A_diff2(pi + C_func(theta-1, phi, psi), logit_rho, log_scale) - 
                         A_diff2(pi + C_func(theta, phi, psi), logit_rho, log_scale);

    double d_log_scale = A_diff3(pi + C_func(theta-1, phi, psi), logit_rho, log_scale) - 
                         A_diff3(pi + C_func(theta, phi, psi), logit_rho, log_scale);

    double d_psi = A_diff1(pi + C_func(theta-1, phi, psi), logit_rho, log_scale) * C_diff3(theta-1) - 
                   A_diff1(pi + C_func(theta, phi, psi), logit_rho, log_scale) * C_diff3(theta);  

    return std::make_tuple(d_logit_rho, d_log_scale, d_psi);
}

double compute_gamma(double pi, double phi, double theta, double log_both, double log_scale, double psi) {

    double val1 = B_func(pi + C_func(theta - 1, phi, psi), log_both, log_scale);
    double val2 = B_func(pi + C_func(theta, phi, psi), log_both, log_scale);

    return val1 - val2;
}

std::tuple<double, double, double> gamma_gradient(double pi, double phi, double theta, double log_both, 
        double log_scale, double psi) {

    double d_log_both = B_diff2(pi + C_func(theta-1, phi, psi), log_both, log_scale) - 
                        B_diff2(pi + C_func(theta, phi, psi), log_both, log_scale);
      
    double d_log_scale = B_diff3(pi + C_func(theta-1, phi, psi), log_both, log_scale) - 
                         B_diff3(pi + C_func(theta, phi, psi), log_both, log_scale);

    double d_psi = B_diff1(pi + C_func(theta-1, phi, psi), log_both, log_scale) * C_diff3(theta-1) - 
                   B_diff1(pi + C_func(theta, phi, psi), log_both, log_scale) * C_diff3(theta); 

    return std::make_tuple(d_log_both, d_log_scale, d_psi);

}

double compute_psi(double phi, double theta, double log_scale) { 

    /* We simplify the function using properties of the exponential function. */
    double val = (phi / std::sqrt(2.0)) *  std::exp(-.5 * log_scale) 
        - (1 - phi * phi) / 2.0 + (1 - phi * phi) * theta;

    return val;
}

double psi_gradient(double phi, double log_scale) {

    return (phi / std::sqrt(2.0)) * std::exp(-.5 * log_scale)  * -.5;

}

dvec link_total(double phi, double pi, double theta, double beta, double gamma, double log_both, double log_scale,
        double logit_rho, double psi, double zeta) {

    double beta_diff = beta - compute_beta(pi, phi, theta, logit_rho, log_scale, psi); 
    double gamma_diff = gamma - compute_gamma(pi, phi, theta, log_both, log_scale, psi); 
    double psi_diff = psi - compute_psi(phi, theta, log_scale);
    double zeta_diff = 1 - (zeta + phi * phi);

    dvec returnvec{beta_diff, gamma_diff, psi_diff, zeta_diff};

    return returnvec;

}

dmat link_jacobian(double phi, double pi, double theta, double log_both, double log_scale, double logit_rho,
        double psi) {

    dmat returnmat = arma::zeros<dmat>(4,7);

    // Calculate the first row.
    auto [beta_logit_rho, beta_log_scale, beta_psi] = beta_gradient(pi, phi, theta, logit_rho, log_scale, psi); 
    returnmat(0,0) = 1;
    returnmat(0,3) = -1 * beta_log_scale;
    returnmat(0,4) = -1 * beta_logit_rho; 
    returnmat(0,5) = -1 * beta_psi;


    // Calculate the second row. //
    auto [gamma_log_both, gamma_log_scale, gamma_psi] = gamma_gradient(pi, phi, theta, log_both, log_scale, psi); 

    returnmat(1,1) = 1; 
    returnmat(1,2) = -1 * gamma_log_both;
    returnmat(1,3) = -1 * gamma_log_scale;
    returnmat(1,5) = -1 * gamma_psi;


    // Calculate the third row.
    double psi_log_scale = psi_gradient(phi, log_scale); 
    returnmat(2,3) = - psi_log_scale;
    returnmat(2,5) = 1;
    
    // Calculate the fourth row.
    returnmat(3,6) = -1; 

    return returnmat;
}

dmat covariance_kernel(double phi1, double pi1, double theta1, double phi2,
        double pi2, double theta2,  double log_both, double log_scale, double
        logit_rho, const dmat& omega_cov, double psi) {

    dmat link_grad_left = link_jacobian(phi1,  pi1,  theta1, log_both,
            log_scale, logit_rho,  psi); 

    dmat link_grad_right = link_jacobian(phi2,  pi2,  theta2, log_both,
            log_scale, logit_rho,  psi); 

    return link_grad_left * omega_cov * link_grad_right.t();

}



PYBIND11_MODULE(libvolpriceinference, m) {

    m.def("_simulate_autoregressive_gamma", &simulate_autoregressive_gamma, stream_redirect(), 
          "This function provides draws from the ARG(1) process of Gourieroux & Jaiak",
          "delta"_a=1, "rho"_a=0, "scale"_a=.1, "time_dim"_a=100, "initial_point"_a=.1); 
    
    m.def("_threadsafe_gaussian_rvs", &threadsafe_gaussian_rvs, stream_redirect(), 
          "This function provides a vector of Gaussian random variates that are drawn in a thread safe manner.",
          "time_dim"_a=100); 

    m.def("A_func", &A_func, stream_redirect(), "This function computes function A() in the accompanying paper.",
            "x"_a, "logit_rho"_a, "log_scale"_a);

    m.def("B_func", &B_func, stream_redirect(), "This function computes function B() in the accompanying paper.",
            "x"_a, "log_both"_a, "log_scale"_a);

    m.def("C_func", &C_func, stream_redirect(), "This function computes function C() in the accompanying paper.",
            "x"_a, "phi"_a, "psi"_a);

    m.def("compute_beta", &compute_beta, stream_redirect(), 
            "This function computes function the first link function in the accompanying paper.",
            "pi"_a, "phi"_a, "theta"_a, "logit_rho"_a, "log_scale"_a, "psi"_a);

    m.def("compute_gamma", &compute_gamma, stream_redirect(), 
            "This function computes function the second link function in the accompanying paper.",
            "pi"_a, "phi"_a, "theta"_a, "log_both"_a, "log_scale"_a, "psi"_a);

    m.def("compute_psi", &compute_psi, stream_redirect(), 
            "This function computes function the second link function in the accompanying paper.",
            "phi"_a, "theta"_a, "log_scale"_a);

    m.def("link_total", &link_total, stream_redirect(),
            "This function computes the link function.",
            "phi"_a, "pi"_a, "theta"_a, "beta"_a, "gamma"_a, "log_both"_a,  "log_scale"_a,  "logit_rho"_a, 
            "psi"_a, "zeta"_a); 

    m.def("link_jacobian", &link_jacobian, stream_redirect(),
            "This function computes the jacobian of the link function.", "phi"_a, "pi"_a, "theta"_a, "log_both"_a,
            "log_scale"_a,  "logit_rho"_a, "psi"_a); 

    m.def("A_diff2", &A_diff2, stream_redirect(), 
        "This function computes the derivative of A() with respect to logit_rho", "x"_a, "logit_rho"_a, 
        "log_scale"_a);

    m.def("B_diff3", &B_diff3, stream_redirect(), 
        "This function computes the derivative of B() with respect to log_scale", "x"_a, "log_both"_a, 
        "log_scale"_a);

    m.def("_covariance_kernel", &covariance_kernel, stream_redirect(), 
          "This function computes the covariance kernel defined in the accompanying paper", 
          "phi1"_a, "pi1"_a, "theta1"_a, "phi2"_a, "pi2"_a, "theta2"_a,
          "log_both"_a, "log_scale"_a, "logit_rho"_a, "omega_cov"_a, "psi"_a); 


}
