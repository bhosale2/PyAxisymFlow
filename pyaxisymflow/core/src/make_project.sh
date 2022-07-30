#!/usr/bin/env bash
# not posix compliant

if command -v python3 >/dev/null 2>&1; then
	# TODO Make it uniform with python and don't have definitions in both languages
	declare -a cpp_header_files=("weno_kernel.hpp" "weno_2D_kernel.hpp" "mesh_to_particles.hpp" "particles_to_mesh.hpp" "extrapolate_using_least_squares.hpp" "reinitialize_level_set.hpp")
	for headerfile in "${cpp_header_files[@]}"; do
		./bindthem.py "${headerfile}"
	done
	python3 setup.py build_ext --inplace
else
	echo "Python3 is a required dependency, please install it"
fi
