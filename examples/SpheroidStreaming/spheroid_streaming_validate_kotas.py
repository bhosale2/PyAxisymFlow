import os
from spheroid_streaming import simulate_oscillating_spheroid

# (spheroid_AR, womersley_sq) pairs to validate from Kotas 2007
cases = [
    (0.75, 28.195),
    (0.75, 42.4298),
    (1.0, 31.3415),
    (1.0, 41.5948),
    (1.0, 53.5910),
    (1.0, 59.7499),
    (1.0, 66.8879),
    (1.0, 80.1726),
    (1.0, 93.8293),
    (1.0, 100.2640),
    (1.0, 149.4063),
    (1.3, 37.1179),
    (1.3, 42.7611),
    (1.3, 74.2809),
    (1.3, 73.6915),
    (1.3, 110.5705),
    (1.3, 111.0121),
    (2.0, 48.7771),
]

if __name__ == "__main__":

    for (spheroid_AR, womersley_sq) in cases:
        dirname = f"spheroidAR{spheroid_AR}_womersley_sq{womersley_sq}"
        if os.path.exists(dirname):
            os.system(f"rm -rf {dirname}")
        os.makedirs(dirname, exist_ok=True)

        os.chdir(dirname)
        print(f"Simulating AR={spheroid_AR}, womersley_sq={womersley_sq}")
        simulate_oscillating_spheroid(
            womersley_sq=womersley_sq,
            radius=0.075,
            spheroid_AR=spheroid_AR,
            grid_size_z=512,
            plot_figure=True,
            save_vtk=True,
        )
        os.chdir("../")
