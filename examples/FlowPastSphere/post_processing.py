import numpy as np
import matplotlib.pyplot as plt

from pyaxisymflow.utils.custom_cmap import lab_cmp
from pyaxisymflow.utils.plotset import plotset


def plot_contours(t, Z, R, vorticity, u_z, char_func, Re):
    plotset()
    plt.figure(figsize=(10, 5))
    plt.contourf(
        Z,
        R,
        vorticity,
        levels=100,
        extend="both",
        cmap=lab_cmp,
    )

    plt.contour(
        Z,
        R,
        u_z,
        levels=[
            0.0,
        ],
        colors="red",
    )

    plt.contour(
        Z,
        R,
        char_func,
        levels=[
            0.5,
        ],
        colors="grey",
    )

    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    plt.savefig("snap_" + str("%0.4d" % (t * 100)) + f"_Re{Re}" + ".png")
    plt.clf()
    plt.close("all")


def plot_Cd_vs_Re(sim_Re, sim_results):
    fig, ax = plt.subplots()
    fig.set_size_inches([10, 8])
    ax.plot(sim_Re, sim_results, lw=2, color="b", label="Simulation result")
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 3])
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"$C_d$")
    return fig, ax


def compare_with_exp_data(ax):
    filename = "roo.txt"
    data = {"Re": [], "Cd": []}

    with open(filename, "r") as f:
        array_name = "Re"
        curr = f.readline()

        while curr:
            if curr == "\n":
                array_name = "Cd"
                curr = f.readline()
                continue
            data[array_name].append(float(curr[:-1]))
            curr = f.readline()
        f.close()

    ax.plot(data["Re"], data["Cd"], "*k", label="Roos & William")
    ax.legend()


def compare_with_emp_eqn(ax):
    Re = np.linspace(2, 200, 100)
    Cd = 24 / Re * (1 + 0.15 * Re**0.687)
    ax.plot(Re, Cd, lw=2, color="r", label=r"$C_d = \frac{24}{Re}(1+0.15*Re^{0.687})$")
    ax.legend()
