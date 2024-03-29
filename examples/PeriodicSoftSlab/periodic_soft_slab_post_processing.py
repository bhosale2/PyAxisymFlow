import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pyaxisymflow.utils.plotset import plotset
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


def plot_time_dependent_theory_comparison(
    times,
    sim_r,
    sim_v_list,
    theory_r,
    theory_v_list,
):
    assert len(times) == len(sim_v_list), "Simulation list length mismatch"
    assert len(times) == len(theory_v_list), "Theory list length mismatch"

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    for i in range(len(times)):
        color = lab_cmp(times[i])
        ax.plot(
            sim_v_list[i, :],
            sim_r,
            lw=2,
            color=color,
            label=r"$t/T$={:.2f}".format(times[i]),
        )
        ax.scatter(
            theory_v_list[i, :],
            theory_r,
            facecolor="none",
            edgecolor=color,
            linewidths=2,
        )
    ax.legend(fontsize=14)
    ax.set_xlabel(r"$v_z / \hat{V}_{wall}$", fontsize=18)
    ax.set_ylabel(r"$r/(L_s + L_f)$", fontsize=18)
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 1])

    fig.savefig("theory_comparison.png")
    plt.show()


def plot_convergence(dx, l2_error, linf_error):
    plotset()
    fig, ax = plt.subplots()
    num_time = l2_error.shape[1]

    markers = ["o", "v", "s", "d", "^"]
    for i in range(num_time):
        ax.scatter(
            dx,
            l2_error[:, i],
            s=100,
            facecolor="none",
            edgecolor=lab_cmp(0.1),
            linewidths=2,
            marker=markers[i],
        )
        ax.scatter(
            dx,
            linf_error[:, i],
            s=100,
            facecolor="none",
            edgecolor=lab_cmp(0.9),
            linewidths=2,
            marker=markers[i],
        )

    h_space = np.linspace(1e-6, 1.0, 100)
    ax.plot(h_space, 30 * h_space, "--k")
    ax.plot(h_space, 100 * h_space**2, "--k")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\|\mathbf{e}\|$")
    ax.set_xlim([7e-4, 1.5e-2])
    ax.set_ylim([1e-6, 1.0])

    fig.savefig("spatial_convergence.png")
    plt.show()


def soft_slab_plotset():
    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 15.0
    plt.rcParams["xtick.major.width"] = 2.0
    plt.rcParams["xtick.minor.size"] = 7.0
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.major.size"] = 15.0
    plt.rcParams["ytick.major.width"] = 2.0
    plt.rcParams["ytick.minor.size"] = 7.0
    plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.pad"] = 10.0
    plt.rcParams["xtick.minor.pad"] = 10.0
    plt.rcParams["ytick.major.pad"] = 10.0
    plt.rcParams["ytick.minor.pad"] = 10.0

    plt.rcParams["xtick.labelsize"] = 36.0
    plt.rcParams["ytick.labelsize"] = 36.0

    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["lines.linewidth"] = 4.0
    plt.rcParams["axes.grid"] = False

    bwr = plt.get_cmap("bwr")

    new_colors = np.vstack(
        (
            bwr(np.linspace(0.0, 1.0, 256)),
            bwr(np.linspace(1.0, 0.0, 256)),
        )
    )
    return ListedColormap(new_colors, "BimodalRedBlue")
