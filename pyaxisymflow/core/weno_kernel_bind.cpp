// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "weno_kernel.hpp"

namespace py = pybind11;

template <class Float>
void _weno5_FD_1D(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
        const Float alpha,
       const Float factor,
          const int start,
            const int end
                  )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_1D <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
           _boundary_flux, boundary_flux.shape(0),
                    alpha,
                   factor,
                    start,
                      end
                               );
}

template <class Float>
void _weno5_FD_1D_novec(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
        const Float alpha,
       const Float factor
                        )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_1D_novec <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
           _boundary_flux, boundary_flux.shape(0),
                    alpha,
                   factor
                                     );
}

template <class Float>
void _weno5_FD_1D_novec_optimized(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
        const Float alpha,
       const Float factor
                                  )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_1D_novec_optimized <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
           _boundary_flux, boundary_flux.shape(0),
                    alpha,
                   factor
                                               );
}

template <class Float>
void _weno5_FD_1D_all_novec(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
        const Float alpha,
       const Float factor
                            )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_1D_all_novec <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
           _boundary_flux, boundary_flux.shape(0),
                    alpha,
                   factor
                                         );
}

template <class Float>
void _weno5_FD_1D_all_novec_optimized(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & left_cell_boundary_flux,
py::array_t<Float> & right_cell_boundary_flux,
        const Float alpha,
       const Float factor
                                      )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_left_cell_boundary_flux = left_cell_boundary_flux.mutable_unchecked();
    auto py_right_cell_boundary_flux = right_cell_boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_left_cell_boundary_flux = py_left_cell_boundary_flux.mutable_data();
    Float *_right_cell_boundary_flux = py_right_cell_boundary_flux.mutable_data();

    return weno5_FD_1D_all_novec_optimized <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
 _left_cell_boundary_flux, left_cell_boundary_flux.shape(0),
_right_cell_boundary_flux, right_cell_boundary_flux.shape(0),
                    alpha,
                   factor
                                                   );
}

template <class Float>
void _weno5_FD_1D_all_novec_optimized_two(
 py::array_t<Float> & uin,
py::array_t<Float> & sampled_flux_at_nodes,
py::array_t<Float> & total_flux_at_center,
py::array_t<Float> & boundary_flux,
        const Float alpha,
       const Float factor
                                          )
{
    auto py_uin = uin.unchecked();
    auto py_sampled_flux_at_nodes = sampled_flux_at_nodes.unchecked();
    auto py_total_flux_at_center = total_flux_at_center.mutable_unchecked();
    auto py_boundary_flux = boundary_flux.mutable_unchecked();
    const Float *_uin = py_uin.data();
    const Float *_sampled_flux_at_nodes = py_sampled_flux_at_nodes.data();
    Float *_total_flux_at_center = py_total_flux_at_center.mutable_data();
    Float *_boundary_flux = py_boundary_flux.mutable_data();

    return weno5_FD_1D_all_novec_optimized_two <Float>(
                     _uin, uin.shape(0),
   _sampled_flux_at_nodes, sampled_flux_at_nodes.shape(0),
    _total_flux_at_center, total_flux_at_center.shape(0),
           _boundary_flux, boundary_flux.shape(0),
                    alpha,
                   factor
                                                       );
}

PYBIND11_MODULE(weno_kernel, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for weno_kernel.hpp

    Methods
    -------
    weno5_FD_1D
    weno5_FD_1D_novec
    weno5_FD_1D_novec_optimized
    weno5_FD_1D_all_novec
    weno5_FD_1D_all_novec_optimized
    weno5_FD_1D_all_novec_optimized_two
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("weno5_FD_1D", &_weno5_FD_1D<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"), py::arg("start"), py::arg("end"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_1D_novec", &_weno5_FD_1D_novec<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_1D_novec_optimized", &_weno5_FD_1D_novec_optimized<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_1D_all_novec", &_weno5_FD_1D_all_novec<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_1D_all_novec_optimized", &_weno5_FD_1D_all_novec_optimized<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("left_cell_boundary_flux").noconvert(), py::arg("right_cell_boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"),
R"pbdoc(
)pbdoc");

    m.def("weno5_FD_1D_all_novec_optimized_two", &_weno5_FD_1D_all_novec_optimized_two<double>,
        py::arg("uin").noconvert(), py::arg("sampled_flux_at_nodes").noconvert(), py::arg("total_flux_at_center").noconvert(), py::arg("boundary_flux").noconvert(), py::arg("alpha"), py::arg("factor"),
R"pbdoc(
Saves a modulo call, not much difference)pbdoc");

}

