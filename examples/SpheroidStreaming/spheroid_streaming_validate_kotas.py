import os
import numpy as np
from spheroid_streaming import simulate_oscillating_spheroid

if __name__ == "__main__":

    cases = np.loadtxt(
        "experimental_data_parameter_range.txt", delimiter=",", skiprows=1
    )
    for (spheroid_AR, womersley_square) in cases:
        dirname = f"spheroidAR{spheroid_AR}_womersley_square{womersley_square}"
        if os.path.exists(dirname):
            os.system(f"rm -rf {dirname}")
        os.makedirs(dirname, exist_ok=True)

        os.chdir(dirname)
        print(f"Simulating AR={spheroid_AR}, womersley_square={womersley_square}")
        simulate_oscillating_spheroid(
            womersley_square=womersley_square,
            radius=0.075,
            spheroid_AR=spheroid_AR,
            grid_size_z=512,
            plot_figure=True,
            save_vtk=True,
        )
        os.chdir("../")
