#include <stdexcept>
#include <cmath>
#include <cassert>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/operators.h"
#include <pybind11/iostream.h>


namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;
using stream_redirect = py::call_guard<py::scoped_ostream_redirect>;                                               

/* 
 * Initializes a random generator in a thread_save way to be used globally througout the entire library. 
 */
inline
std::mt19937_64& initialize_mt_generator() { 
    
    thread_local std::random_device rd; 
    thread_local std::mt19937_64 engine(rd());
    
    return engine; 

}

std::vector<double> simulate_autoregressive_gamma(double delta, double rho, double scale, size_t time_dim, double
        initial_point) {
    
    assert (time_dim > 1);

    std::vector<double> draws;
    draws.reserve(time_dim+1);

    /* We start at the initial point. */
    draws.push_back(initial_point);
    thread_local auto& generator = initialize_mt_generator();

    for(size_t idx=0; idx<time_dim; ++idx) {
        
        /* The conditional distribution of an ARG(1) process is non-centered Gamma, which has a representation as
         * a Poisson mixture of Gamma */
          std::poisson_distribution<int> poi_dist(rho * draws.back() / scale);
          int latent_var = poi_dist(generator);
          std::gamma_distribution<double> gamma_dist(delta + latent_var, scale);
          draws.push_back(gamma_dist(generator));
    }

    std::vector<double> return_draws(draws.begin() + 1, draws.end());

    return return_draws;
}

std::vector<double> threadsafe_gaussian_rvs(size_t time_dim) {

    std::normal_distribution dist(0,1);
    thread_local auto& generator = initialize_mt_generator();

    std::vector<double> return_draws(time_dim);

    for (auto& x : return_draws) {
        x = dist(generator);
    }
    
    return return_draws;
}


PYBIND11_MODULE(libvolpriceinference, m) {

    m.def("_simulate_autoregressive_gamma", &simulate_autoregressive_gamma, stream_redirect(), 
          "This function provides draws from the ARG(1) process of Gourieroux & Jaiak",
          "delta"_a=1, "rho"_a=0, "scale"_a=.1, "time_dim"_a=100, "initial_point"_a=.1); 
    
    m.def("_threadsafe_gaussian_rvs", &threadsafe_gaussian_rvs, stream_redirect(), 
          "This function provides a vector of Gaussian random variates that are drawn in a thread safe manner.",
          "time_dim"_a=100); 

}
