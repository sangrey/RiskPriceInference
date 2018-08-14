#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/operators.h"
#include <pybind11/iostream.h>


namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;
using stream_redirect = py::call_guard<py::scoped_ostream_redirect>;                                               


double abs_gaussian_moments(npulong power, npdouble scale=1) {

    npdouble log_moment =  0.5 * power * std::log(2);
    log_moment +=  std::lgamma(0.5 * (power + 1));
    log_moment += power * std::log(static_cast<double>(scale));

    return std::exp(log_moment);

}

PYBIND11_MODULE(libriskpriceinference, m) {

    m.def("abs_gaussian_moments", &abs_gaussian_moments, stream_redirect(), 
            "Computes the centered abolute gaussian moment of order p.", "power"_a, "scale"_a=1); 

}
