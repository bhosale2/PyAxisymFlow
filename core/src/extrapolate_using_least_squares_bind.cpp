// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "extrapolate_using_least_squares.hpp"

namespace py = pybind11;

template <class ShortInt, class Float>
void _extrapolate_using_least_squares_till_first_order(
py::array_t<ShortInt> & current_flag,
py::array_t<ShortInt> & target_flag,
py::array_t<Float> & eta_x,
py::array_t<Float> & eta_y,
py::array_t<Float> & grid_x,
py::array_t<Float> & grid_y
                                                       )
{
    auto py_current_flag = current_flag.mutable_unchecked();
    auto py_target_flag = target_flag.unchecked();
    auto py_eta_x = eta_x.mutable_unchecked();
    auto py_eta_y = eta_y.mutable_unchecked();
    auto py_grid_x = grid_x.unchecked();
    auto py_grid_y = grid_y.unchecked();
    ShortInt *_current_flag = py_current_flag.mutable_data();
    const ShortInt *_target_flag = py_target_flag.data();
    Float *_eta_x = py_eta_x.mutable_data();
    Float *_eta_y = py_eta_y.mutable_data();
    const Float *_grid_x = py_grid_x.data();
    const Float *_grid_y = py_grid_y.data();

    return extrapolate_using_least_squares_till_first_order <ShortInt, Float>(
            _current_flag, current_flag.shape(0), current_flag.shape(1),
             _target_flag, target_flag.shape(0), target_flag.shape(1),
                   _eta_x, eta_x.shape(0), eta_x.shape(1),
                   _eta_y, eta_y.shape(0), eta_y.shape(1),
                  _grid_x, grid_x.shape(0),
                  _grid_y, grid_y.shape(0)
                                                                              );
}

template <class ShortInt, class Float>
void _extrapolate_using_least_squares_till_second_order(
py::array_t<ShortInt> & current_flag,
py::array_t<ShortInt> & target_flag,
py::array_t<Float> & eta_x,
py::array_t<Float> & eta_y,
py::array_t<Float> & grid_x,
py::array_t<Float> & grid_y
                                                        )
{
    auto py_current_flag = current_flag.mutable_unchecked();
    auto py_target_flag = target_flag.unchecked();
    auto py_eta_x = eta_x.mutable_unchecked();
    auto py_eta_y = eta_y.mutable_unchecked();
    auto py_grid_x = grid_x.unchecked();
    auto py_grid_y = grid_y.unchecked();
    ShortInt *_current_flag = py_current_flag.mutable_data();
    const ShortInt *_target_flag = py_target_flag.data();
    Float *_eta_x = py_eta_x.mutable_data();
    Float *_eta_y = py_eta_y.mutable_data();
    const Float *_grid_x = py_grid_x.data();
    const Float *_grid_y = py_grid_y.data();

    return extrapolate_using_least_squares_till_second_order <ShortInt, Float>(
            _current_flag, current_flag.shape(0), current_flag.shape(1),
             _target_flag, target_flag.shape(0), target_flag.shape(1),
                   _eta_x, eta_x.shape(0), eta_x.shape(1),
                   _eta_y, eta_y.shape(0), eta_y.shape(1),
                  _grid_x, grid_x.shape(0),
                  _grid_y, grid_y.shape(0)
                                                                               );
}

PYBIND11_MODULE(extrapolate_using_least_squares, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for extrapolate_using_least_squares.hpp

    Methods
    -------
    extrapolate_using_least_squares_till_first_order
    extrapolate_using_least_squares_till_second_order
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("extrapolate_using_least_squares_till_first_order", &_extrapolate_using_least_squares_till_first_order<short, double>,
        py::arg("current_flag").noconvert(), py::arg("target_flag").noconvert(), py::arg("eta_x").noconvert(), py::arg("eta_y").noconvert(), py::arg("grid_x").noconvert(), py::arg("grid_y").noconvert(),
R"pbdoc(
)pbdoc");

    m.def("extrapolate_using_least_squares_till_second_order", &_extrapolate_using_least_squares_till_second_order<short, double>,
        py::arg("current_flag").noconvert(), py::arg("target_flag").noconvert(), py::arg("eta_x").noconvert(), py::arg("eta_y").noconvert(), py::arg("grid_x").noconvert(), py::arg("grid_y").noconvert(),
R"pbdoc(
)pbdoc");

}

