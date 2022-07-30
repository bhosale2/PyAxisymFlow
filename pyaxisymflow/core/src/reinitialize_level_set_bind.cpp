// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "reinitialize_level_set.hpp"

namespace py = pybind11;

template <class Float>
void _reinitialize_level_set_using_hamilton_jacobi(
py::array_t<Float> & phi_old,
py::array_t<Float> & phi_new,
py::array_t<Float> & sign_function,
py::array_t<Float> & dphi_dx,
py::array_t<Float> & dphi_dy,
py::array_t<Float> & result_buffer,
py::array_t<Float> & left_cell_boundary_flux,
py::array_t<Float> & right_cell_boundary_flux,
py::array_t<Float> & bottom_cell_boundary_flux,
py::array_t<Float> & top_cell_boundary_flux,
      const Float delta_x,
      const Float delta_y,
     const Float x_factor,
     const Float y_factor,
  const Float reinit_band,
const Float reinit_tolerance
                                                   )
{
    auto py_phi_old = phi_old.mutable_unchecked();
    auto py_phi_new = phi_new.mutable_unchecked();
    auto py_sign_function = sign_function.mutable_unchecked();
    auto py_dphi_dx = dphi_dx.mutable_unchecked();
    auto py_dphi_dy = dphi_dy.mutable_unchecked();
    auto py_result_buffer = result_buffer.mutable_unchecked();
    auto py_left_cell_boundary_flux = left_cell_boundary_flux.mutable_unchecked();
    auto py_right_cell_boundary_flux = right_cell_boundary_flux.mutable_unchecked();
    auto py_bottom_cell_boundary_flux = bottom_cell_boundary_flux.mutable_unchecked();
    auto py_top_cell_boundary_flux = top_cell_boundary_flux.mutable_unchecked();
    Float *_phi_old = py_phi_old.mutable_data();
    Float *_phi_new = py_phi_new.mutable_data();
    Float *_sign_function = py_sign_function.mutable_data();
    Float *_dphi_dx = py_dphi_dx.mutable_data();
    Float *_dphi_dy = py_dphi_dy.mutable_data();
    Float *_result_buffer = py_result_buffer.mutable_data();
    Float *_left_cell_boundary_flux = py_left_cell_boundary_flux.mutable_data();
    Float *_right_cell_boundary_flux = py_right_cell_boundary_flux.mutable_data();
    Float *_bottom_cell_boundary_flux = py_bottom_cell_boundary_flux.mutable_data();
    Float *_top_cell_boundary_flux = py_top_cell_boundary_flux.mutable_data();

    return reinitialize_level_set_using_hamilton_jacobi <Float>(
                 _phi_old, phi_old.shape(0), phi_old.shape(1),
                 _phi_new, phi_new.shape(0), phi_new.shape(1),
           _sign_function, sign_function.shape(0), sign_function.shape(1),
                 _dphi_dx, dphi_dx.shape(0), dphi_dx.shape(1),
                 _dphi_dy, dphi_dy.shape(0), dphi_dy.shape(1),
           _result_buffer, result_buffer.shape(0), result_buffer.shape(1),
 _left_cell_boundary_flux, left_cell_boundary_flux.shape(0), left_cell_boundary_flux.shape(1),
_right_cell_boundary_flux, right_cell_boundary_flux.shape(0), right_cell_boundary_flux.shape(1),
_bottom_cell_boundary_flux, bottom_cell_boundary_flux.shape(0), bottom_cell_boundary_flux.shape(1),
  _top_cell_boundary_flux, top_cell_boundary_flux.shape(0), top_cell_boundary_flux.shape(1),
                  delta_x,
                  delta_y,
                 x_factor,
                 y_factor,
              reinit_band,
         reinit_tolerance
                                                                );
}

template <class Float>
void _reinitialize_level_set_using_hamilton_jacobi_heap_memory(
py::array_t<Float> & phi_old,
py::array_t<Float> & phi_new,
      const Float delta_x,
      const Float delta_y,
     const Float x_factor,
     const Float y_factor,
  const Float reinit_band,
const Float reinit_tolerance
                                                               )
{
    auto py_phi_old = phi_old.mutable_unchecked();
    auto py_phi_new = phi_new.mutable_unchecked();
    Float *_phi_old = py_phi_old.mutable_data();
    Float *_phi_new = py_phi_new.mutable_data();

    return reinitialize_level_set_using_hamilton_jacobi_heap_memory <Float>(
                 _phi_old, phi_old.shape(0), phi_old.shape(1),
                 _phi_new, phi_new.shape(0), phi_new.shape(1),
                  delta_x,
                  delta_y,
                 x_factor,
                 y_factor,
              reinit_band,
         reinit_tolerance
                                                                            );
}

PYBIND11_MODULE(reinitialize_level_set, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for reinitialize_level_set.hpp

    Methods
    -------
    reinitialize_level_set_using_hamilton_jacobi
    reinitialize_level_set_using_hamilton_jacobi_heap_memory
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("reinitialize_level_set_using_hamilton_jacobi", &_reinitialize_level_set_using_hamilton_jacobi<double>,
        py::arg("phi_old").noconvert(), py::arg("phi_new").noconvert(), py::arg("sign_function").noconvert(), py::arg("dphi_dx").noconvert(), py::arg("dphi_dy").noconvert(), py::arg("result_buffer").noconvert(), py::arg("left_cell_boundary_flux").noconvert(), py::arg("right_cell_boundary_flux").noconvert(), py::arg("bottom_cell_boundary_flux").noconvert(), py::arg("top_cell_boundary_flux").noconvert(), py::arg("delta_x"), py::arg("delta_y"), py::arg("x_factor"), py::arg("y_factor"), py::arg("reinit_band"), py::arg("reinit_tolerance"),
R"pbdoc(
)pbdoc");

    m.def("reinitialize_level_set_using_hamilton_jacobi_heap_memory", &_reinitialize_level_set_using_hamilton_jacobi_heap_memory<double>,
        py::arg("phi_old").noconvert(), py::arg("phi_new").noconvert(), py::arg("delta_x"), py::arg("delta_y"), py::arg("x_factor"), py::arg("y_factor"), py::arg("reinit_band"), py::arg("reinit_tolerance"),
R"pbdoc(
)pbdoc");

}

