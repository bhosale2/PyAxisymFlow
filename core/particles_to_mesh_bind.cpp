// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "particles_to_mesh.hpp"

namespace py = pybind11;

template <class Float>
void _particles_to_mesh_mp4(
py::array_t<Float> & particle_positions,
py::array_t<Float> & input_field_at_particle_positions,
py::array_t<Float> & output_field,
      const Float delta_x
                            )
{
    auto py_particle_positions = particle_positions.unchecked();
    auto py_input_field_at_particle_positions = input_field_at_particle_positions.unchecked();
    auto py_output_field = output_field.mutable_unchecked();
    const Float *_particle_positions = py_particle_positions.data();
    const Float *_input_field_at_particle_positions = py_input_field_at_particle_positions.data();
    Float *_output_field = py_output_field.mutable_data();

    return particles_to_mesh_mp4 <Float>(
      _particle_positions, particle_positions.shape(0),
_input_field_at_particle_positions, input_field_at_particle_positions.shape(0),
            _output_field, output_field.shape(0),
                  delta_x
                                         );
}

template <class Float>
void _particles_to_mesh_2D_mp4(
py::array_t<Float> & particle_positions_x,
py::array_t<Float> & particle_positions_y,
py::array_t<Float> & input_field_at_particle_positions,
py::array_t<Float> & output_field_at_mesh,
      const Float delta_x,
      const Float delta_y
                               )
{
    auto py_particle_positions_x = particle_positions_x.unchecked();
    auto py_particle_positions_y = particle_positions_y.unchecked();
    auto py_input_field_at_particle_positions = input_field_at_particle_positions.unchecked();
    auto py_output_field_at_mesh = output_field_at_mesh.mutable_unchecked();
    const Float *_particle_positions_x = py_particle_positions_x.data();
    const Float *_particle_positions_y = py_particle_positions_y.data();
    const Float *_input_field_at_particle_positions = py_input_field_at_particle_positions.data();
    Float *_output_field_at_mesh = py_output_field_at_mesh.mutable_data();

    return particles_to_mesh_2D_mp4 <Float>(
    _particle_positions_x, particle_positions_x.shape(0), particle_positions_x.shape(1),
    _particle_positions_y, particle_positions_y.shape(0), particle_positions_y.shape(1),
_input_field_at_particle_positions, input_field_at_particle_positions.shape(0), input_field_at_particle_positions.shape(1),
    _output_field_at_mesh, output_field_at_mesh.shape(0), output_field_at_mesh.shape(1),
                  delta_x,
                  delta_y
                                            );
}

template <class Float>
void _particles_to_mesh_2D_unbounded_mp4(
py::array_t<Float> & particle_positions_x,
py::array_t<Float> & particle_positions_y,
py::array_t<Float> & input_field_at_particle_positions,
py::array_t<Float> & output_field_at_mesh,
      const Float delta_x,
      const Float delta_y
                                         )
{
    auto py_particle_positions_x = particle_positions_x.unchecked();
    auto py_particle_positions_y = particle_positions_y.unchecked();
    auto py_input_field_at_particle_positions = input_field_at_particle_positions.unchecked();
    auto py_output_field_at_mesh = output_field_at_mesh.mutable_unchecked();
    const Float *_particle_positions_x = py_particle_positions_x.data();
    const Float *_particle_positions_y = py_particle_positions_y.data();
    const Float *_input_field_at_particle_positions = py_input_field_at_particle_positions.data();
    Float *_output_field_at_mesh = py_output_field_at_mesh.mutable_data();

    return particles_to_mesh_2D_unbounded_mp4 <Float>(
    _particle_positions_x, particle_positions_x.shape(0), particle_positions_x.shape(1),
    _particle_positions_y, particle_positions_y.shape(0), particle_positions_y.shape(1),
_input_field_at_particle_positions, input_field_at_particle_positions.shape(0), input_field_at_particle_positions.shape(1),
    _output_field_at_mesh, output_field_at_mesh.shape(0), output_field_at_mesh.shape(1),
                  delta_x,
                  delta_y
                                                      );
}

PYBIND11_MODULE(particles_to_mesh, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for particles_to_mesh.hpp

    Methods
    -------
    particles_to_mesh_mp4
    particles_to_mesh_2D_mp4
    particles_to_mesh_2D_unbounded_mp4
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("particles_to_mesh_mp4", &_particles_to_mesh_mp4<double>,
        py::arg("particle_positions").noconvert(), py::arg("input_field_at_particle_positions").noconvert(), py::arg("output_field").noconvert(), py::arg("delta_x"),
R"pbdoc(
particle to mesh interpolation using MP4 kernel in a 1D periodic domain)pbdoc");

    m.def("particles_to_mesh_2D_mp4", &_particles_to_mesh_2D_mp4<double>,
        py::arg("particle_positions_x").noconvert(), py::arg("particle_positions_y").noconvert(), py::arg("input_field_at_particle_positions").noconvert(), py::arg("output_field_at_mesh").noconvert(), py::arg("delta_x"), py::arg("delta_y"),
R"pbdoc(
particleto mesh interpolation using MP4 kernel in a 2D periodic domain)pbdoc");

    m.def("particles_to_mesh_2D_unbounded_mp4", &_particles_to_mesh_2D_unbounded_mp4<double>,
        py::arg("particle_positions_x").noconvert(), py::arg("particle_positions_y").noconvert(), py::arg("input_field_at_particle_positions").noconvert(), py::arg("output_field_at_mesh").noconvert(), py::arg("delta_x"), py::arg("delta_y"),
R"pbdoc(
particleto mesh interpolation using MP4 kernel in a 2D unbounded domain)pbdoc");

}

