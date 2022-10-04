import numpy as np
import matplotlib.pyplot as plt

from pyaxisymflow.utils.custom_cmap import lab_cmp


def plot_velocity_profile_with_theory(sim_r, sim_v, theory_r, theory_v, time, v_wall):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    ax.plot(sim_v, sim_r)
    ax.scatter(theory_v, theory_r)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$V_z$")
    ax.set_xlim([-v_wall, v_wall])
    fig.savefig("snap_" + str("%0.4d" % (time * 100)) + ".png")
    plt.close()


def plot_vorticity_contours(
    z_grid, r_grid, vorticity, solid_eta, solid_char_func, wall_char_func, time
):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)

    # Plot vorticity contours
    vort = ax.contourf(
        z_grid,
        r_grid,
        vorticity,
        levels=np.linspace(-25, 25, 25),
        extend="both",
        cmap=lab_cmp,
    )

    # plot reference map contours
    ax.contour(z_grid, r_grid, solid_eta, levels=20, cmap="Greens", linewidths=2)

    # plot elastic body - fluid interface
    ax.contour(
        z_grid,
        r_grid,
        solid_char_func,
        linewidths=3,
        levels=[0.5],
        colors="red",
    )

    # plot oscillating wall
    ax.contour(
        z_grid,
        r_grid,
        wall_char_func,
        linewidths=3,
        levels=[0.5],
        colors="k",
    )

    fig.colorbar(vort, ax=ax, location="right")

    fig.savefig("vort_" + str("%0.4d" % (time * 100)) + ".png")
    plt.close()
