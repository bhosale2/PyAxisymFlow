// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "weno_2D_kernel.hpp"

namespace py = pybind11;

template <class Float>
void _weno5_FD_2D_novec(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes_x,
py::array_t<Float> & sampled_flux_at_nodes_y,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
      const Float x_alpha,
      const Float y_alpha,
     const Float x_factor,
     const Float y_factor
                        )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes_x = sampled_flux_at_nodes_x.unchecked();
    auto py_sampled_flux_at_nodes_y = sampled_flux_at_nodes_y.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes_x = py_sampled_flux_at_nodes_x.data();
    const Float *_sampled_flux_at_nodes_y = py_sampled_flux_at_nodes_y.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_2D_novec <Float>(
                     _uin, uin.shape(0), uin.shape(1),
 _sampled_flux_at_nodes_x, sampled_flux_at_nodes_x.shape(0), sampled_flux_at_nodes_x.shape(1),
 _sampled_flux_at_nodes_y, sampled_flux_at_nodes_y.shape(0), sampled_flux_at_nodes_y.shape(1),
    _total_flux_at_center, total_flux_at_center.shape(0), total_flux_at_center.shape(1),
           _boundary_flux, boundary_flux.shape(0), boundary_flux.shape(1),
                  x_alpha,
                  y_alpha,
                 x_factor,
                 y_factor
                                     );
}

template <class Float>
void _weno5_FD_2D_all_novec(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes_x,
py::array_t<Float> & sampled_flux_at_nodes_y,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
      const Float x_alpha,
      const Float y_alpha,
     const Float x_factor,
     const Float y_factor
                            )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes_x = sampled_flux_at_nodes_x.unchecked();
    auto py_sampled_flux_at_nodes_y = sampled_flux_at_nodes_y.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes_x = py_sampled_flux_at_nodes_x.data();
    const Float *_sampled_flux_at_nodes_y = py_sampled_flux_at_nodes_y.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_2D_all_novec <Float>(
                     _uin, uin.shape(0), uin.shape(1),
 _sampled_flux_at_nodes_x, sampled_flux_at_nodes_x.shape(0), sampled_flux_at_nodes_x.shape(1),
 _sampled_flux_at_nodes_y, sampled_flux_at_nodes_y.shape(0), sampled_flux_at_nodes_y.shape(1),
    _total_flux_at_center, total_flux_at_center.shape(0), total_flux_at_center.shape(1),
           _boundary_flux, boundary_flux.shape(0), boundary_flux.shape(1),
                  x_alpha,
                  y_alpha,
                 x_factor,
                 y_factor
                                         );
}

template <class Float>
void _weno5_FD_2D_all_novec_reverse_iteration(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes_x,
py::array_t<Float> & sampled_flux_at_nodes_y,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & left_cell_boundary_flux,
py::array_t<Float> & right_cell_boundary_flux,
py::array_t<Float> & bottom_cell_boundary_flux,
py::array_t<Float> & top_cell_boundary_flux,
      const Float x_alpha,
      const Float y_alpha,
     const Float x_factor,
     const Float y_factor
                                              )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes_x = sampled_flux_at_nodes_x.unchecked();
    auto py_sampled_flux_at_nodes_y = sampled_flux_at_nodes_y.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_left_cell_boundary_flux = left_cell_boundary_flux.mutable_unchecked();
    auto py_right_cell_boundary_flux = right_cell_boundary_flux.mutable_unchecked();
    auto py_bottom_cell_boundary_flux = bottom_cell_boundary_flux.mutable_unchecked();
    auto py_top_cell_boundary_flux = top_cell_boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes_x = py_sampled_flux_at_nodes_x.data();
    const Float *_sampled_flux_at_nodes_y = py_sampled_flux_at_nodes_y.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_left_cell_boundary_flux = py_left_cell_boundary_flux.mutable_data();
    Float *_right_cell_boundary_flux = py_right_cell_boundary_flux.mutable_data();
    Float *_bottom_cell_boundary_flux = py_bottom_cell_boundary_flux.mutable_data();
    Float *_top_cell_boundary_flux = py_top_cell_boundary_flux.mutable_data();

    return weno5_FD_2D_all_novec_reverse_iteration <Float>(
                     _uin, uin.shape(0), uin.shape(1),
 _sampled_flux_at_nodes_x, sampled_flux_at_nodes_x.shape(0), sampled_flux_at_nodes_x.shape(1),
 _sampled_flux_at_nodes_y, sampled_flux_at_nodes_y.shape(0), sampled_flux_at_nodes_y.shape(1),
    _total_flux_at_center, total_flux_at_center.shape(0), total_flux_at_center.shape(1),
 _left_cell_boundary_flux, left_cell_boundary_flux.shape(0), left_cell_boundary_flux.shape(1),
_right_cell_boundary_flux, right_cell_boundary_flux.shape(0), right_cell_boundary_flux.shape(1),
_bottom_cell_boundary_flux, bottom_cell_boundary_flux.shape(0), bottom_cell_boundary_flux.shape(1),
  _top_cell_boundary_flux, top_cell_boundary_flux.shape(0), top_cell_boundary_flux.shape(1),
                  x_alpha,
                  y_alpha,
                 x_factor,
                 y_factor
                                                           );
}

PYBIND11_MODULE(weno_2D_kernel, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for weno_2D_kernel.hpp

    Methods
    -------
    weno5_FD_2D_novec
    weno5_FD_2D_all_novec
    weno5_FD_2D_all_novec_reverse_iteration
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("weno5_FD_2D_novec", &_weno5_FD_2D_novec<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes_x").noconvert(), py::arg("sampled_flux_at_nodes_y").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("x_alpha"), py::arg("y_alpha"), py::arg("x_factor"), py::arg("y_factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_2D_all_novec", &_weno5_FD_2D_all_novec<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes_x").noconvert(), py::arg("sampled_flux_at_nodes_y").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("x_alpha"), py::arg("y_alpha"), py::arg("x_factor"), py::arg("y_factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_2D_all_novec_reverse_iteration", &_weno5_FD_2D_all_novec_reverse_iteration<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes_x").noconvert(), py::arg("sampled_flux_at_nodes_y").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("left_cell_boundary_flux").noconvert(), py::arg("right_cell_boundary_flux").noconvert(), py::arg("bottom_cell_boundary_flux").noconvert(), py::arg("top_cell_boundary_flux").noconvert(), py::arg("x_alpha"), py::arg("y_alpha"), py::arg("x_factor"), py::arg("y_factor"),
R"pbdoc(
)pbdoc");

}

