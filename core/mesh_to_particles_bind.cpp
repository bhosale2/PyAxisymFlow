// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "mesh_to_particles.hpp"

namespace py = pybind11;

template <class Float>
void _wrap_particles_around_1D_domain(
py::array_t<Float> & particle_positions,
 const Float domain_start,
   const Float domain_end
                                      )
{
    auto py_particle_positions = particle_positions.mutable_unchecked();
    Float *_particle_positions = py_particle_positions.mutable_data();

    return wrap_particles_around_1D_domain <Float>(
      _particle_positions, particle_positions.shape(0),
             domain_start,
               domain_end
                                                   );
}

template <class Float>
void _wrap_particles_around_2D_domain(
py::array_t<Float> & particle_positions_x,
py::array_t<Float> & particle_positions_y,
const Float domain_start_x,
 const Float domain_end_x,
const Float domain_start_y,
 const Float domain_end_y
                                      )
{
    auto py_particle_positions_x = particle_positions_x.mutable_unchecked();
    auto py_particle_positions_y = particle_positions_y.mutable_unchecked();
    Float *_particle_positions_x = py_particle_positions_x.mutable_data();
    Float *_particle_positions_y = py_particle_positions_y.mutable_data();

    return wrap_particles_around_2D_domain <Float>(
    _particle_positions_x, particle_positions_x.shape(0), particle_positions_x.shape(1),
    _particle_positions_y, particle_positions_y.shape(0), particle_positions_y.shape(1),
           domain_start_x,
             domain_end_x,
           domain_start_y,
             domain_end_y
                                                   );
}

template <class Float>
void _mesh_to_particles_mp4(
py::array_t<Float> & input_field,
py::array_t<Float> & particle_positions,
py::array_t<Float> & output_field,
      const Float delta_x
                            )
{
    auto py_input_field = input_field.unchecked();
    auto py_particle_positions = particle_positions.unchecked();
    auto py_output_field = output_field.mutable_unchecked();
    const Float *_input_field = py_input_field.data();
    const Float *_particle_positions = py_particle_positions.data();
    Float *_output_field = py_output_field.mutable_data();

    return mesh_to_particles_mp4 <Float>(
             _input_field, input_field.shape(0),
      _particle_positions, particle_positions.shape(0),
            _output_field, output_field.shape(0),
                  delta_x
                                         );
}

template <class Float>
void _mesh_to_particles_2D_mp4(
py::array_t<Float> & input_field_x,
py::array_t<Float> & input_field_y,
py::array_t<Float> & particle_positions_x,
py::array_t<Float> & particle_positions_y,
py::array_t<Float> & output_field_x,
py::array_t<Float> & output_field_y,
      const Float delta_x,
      const Float delta_y
                               )
{
    auto py_input_field_x = input_field_x.unchecked();
    auto py_input_field_y = input_field_y.unchecked();
    auto py_particle_positions_x = particle_positions_x.unchecked();
    auto py_particle_positions_y = particle_positions_y.unchecked();
    auto py_output_field_x = output_field_x.mutable_unchecked();
    auto py_output_field_y = output_field_y.mutable_unchecked();
    const Float *_input_field_x = py_input_field_x.data();
    const Float *_input_field_y = py_input_field_y.data();
    const Float *_particle_positions_x = py_particle_positions_x.data();
    const Float *_particle_positions_y = py_particle_positions_y.data();
    Float *_output_field_x = py_output_field_x.mutable_data();
    Float *_output_field_y = py_output_field_y.mutable_data();

    return mesh_to_particles_2D_mp4 <Float>(
           _input_field_x, input_field_x.shape(0), input_field_x.shape(1),
           _input_field_y, input_field_y.shape(0), input_field_y.shape(1),
    _particle_positions_x, particle_positions_x.shape(0), particle_positions_x.shape(1),
    _particle_positions_y, particle_positions_y.shape(0), particle_positions_y.shape(1),
          _output_field_x, output_field_x.shape(0), output_field_x.shape(1),
          _output_field_y, output_field_y.shape(0), output_field_y.shape(1),
                  delta_x,
                  delta_y
                                            );
}

template <class Float>
void _mesh_to_particles_2D_unbounded_mp4(
py::array_t<Float> & input_field_x,
py::array_t<Float> & input_field_y,
py::array_t<Float> & particle_positions_x,
py::array_t<Float> & particle_positions_y,
py::array_t<Float> & output_field_x,
py::array_t<Float> & output_field_y,
      const Float delta_x,
      const Float delta_y
                                         )
{
    auto py_input_field_x = input_field_x.unchecked();
    auto py_input_field_y = input_field_y.unchecked();
    auto py_particle_positions_x = particle_positions_x.unchecked();
    auto py_particle_positions_y = particle_positions_y.unchecked();
    auto py_output_field_x = output_field_x.mutable_unchecked();
    auto py_output_field_y = output_field_y.mutable_unchecked();
    const Float *_input_field_x = py_input_field_x.data();
    const Float *_input_field_y = py_input_field_y.data();
    const Float *_particle_positions_x = py_particle_positions_x.data();
    const Float *_particle_positions_y = py_particle_positions_y.data();
    Float *_output_field_x = py_output_field_x.mutable_data();
    Float *_output_field_y = py_output_field_y.mutable_data();

    return mesh_to_particles_2D_unbounded_mp4 <Float>(
           _input_field_x, input_field_x.shape(0), input_field_x.shape(1),
           _input_field_y, input_field_y.shape(0), input_field_y.shape(1),
    _particle_positions_x, particle_positions_x.shape(0), particle_positions_x.shape(1),
    _particle_positions_y, particle_positions_y.shape(0), particle_positions_y.shape(1),
          _output_field_x, output_field_x.shape(0), output_field_x.shape(1),
          _output_field_y, output_field_y.shape(0), output_field_y.shape(1),
                  delta_x,
                  delta_y
                                                      );
}

PYBIND11_MODULE(mesh_to_particles, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for mesh_to_particles.hpp

    Methods
    -------
    wrap_particles_around_1D_domain
    wrap_particles_around_2D_domain
    mesh_to_particles_mp4
    mesh_to_particles_2D_mp4
    mesh_to_particles_2D_unbounded_mp4
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("wrap_particles_around_1D_domain", &_wrap_particles_around_1D_domain<double>,
        py::arg("particle_positions").noconvert(), py::arg("domain_start"), py::arg("domain_end"),
R"pbdoc(
Wrap particles around in a domain
Have the option of doing it
1. in a halo region of 10 points from start and end
2. all particles
To not cause undefined behvaior , the limits do not exceed the size,
and so whichever is minimal between the strategy and size is chosen.)pbdoc");

    m.def("wrap_particles_around_2D_domain", &_wrap_particles_around_2D_domain<double>,
        py::arg("particle_positions_x").noconvert(), py::arg("particle_positions_y").noconvert(), py::arg("domain_start_x"), py::arg("domain_end_x"), py::arg("domain_start_y"), py::arg("domain_end_y"),
R"pbdoc(
Wraps particles around a two dimensional domain
Delegates calls to 1D version : please look at its documentation
to see options)pbdoc");

    m.def("mesh_to_particles_mp4", &_mesh_to_particles_mp4<double>,
        py::arg("input_field").noconvert(), py::arg("particle_positions").noconvert(), py::arg("output_field").noconvert(), py::arg("delta_x"),
R"pbdoc(
mesh to particle interpolation using MP4 kernel in a 1D periodic domain)pbdoc");

    m.def("mesh_to_particles_2D_mp4", &_mesh_to_particles_2D_mp4<double>,
        py::arg("input_field_x").noconvert(), py::arg("input_field_y").noconvert(), py::arg("particle_positions_x").noconvert(), py::arg("particle_positions_y").noconvert(), py::arg("output_field_x").noconvert(), py::arg("output_field_y").noconvert(), py::arg("delta_x"), py::arg("delta_y"),
R"pbdoc(
mesh to particle interpolation using MP4 kernel in a 2D periodic domain)pbdoc");

    m.def("mesh_to_particles_2D_unbounded_mp4", &_mesh_to_particles_2D_unbounded_mp4<double>,
        py::arg("input_field_x").noconvert(), py::arg("input_field_y").noconvert(), py::arg("particle_positions_x").noconvert(), py::arg("particle_positions_y").noconvert(), py::arg("output_field_x").noconvert(), py::arg("output_field_y").noconvert(), py::arg("delta_x"), py::arg("delta_y"),
R"pbdoc(
mesh to particle interpolation using MP4 kernel in a 2D unbounded domain)pbdoc");

}

