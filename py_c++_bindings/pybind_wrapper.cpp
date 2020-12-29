#include<pybind11/pybind11.h>
#include<cppmult.hpp>

PYBIND_MODULE(pybind11_example,m){
    m.doc()="pybind plugin for c++";
    m.def("cpp_function",&cppmult,"multiply 2 numbers");
}
