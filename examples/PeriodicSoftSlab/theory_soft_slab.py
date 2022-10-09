import numpy as np
from sympy import (
    exp,
    I,
    conjugate,
    diff,
    lambdify,
    symbols,
    besselj,
    bessely,
)
from sympy.solvers.solveset import linsolve


def theory_axisymmetric_soft_slab_spatial(
    L_f,
    L_s,
    shear_rate,
    omega,
    G,
    V_wall,
    rho_f,
    nu_f,
    rho_ratio=1.0,
    nu_ratio=1.0,
    **kwargs,
):

    # Theoretical Solution
    y, t1 = symbols("y, t1", real=True)

    # define params
    rho_s = rho_ratio * rho_f
    nu_s = nu_ratio * nu_f

    mu_f = nu_f * rho_f
    mu_s = nu_s * rho_s

    lam1 = np.sqrt(1j * omega / nu_f)
    lam2 = omega / np.sqrt(-1j * omega * nu_s + G / rho_s)

    # solution form
    A, B, C = symbols("A, B, C")
    vel_f = (A * besselj(0, lam1 * y) + B * bessely(0, lam1 * y)) * exp(-I * omega * t1)
    u_s = (C * besselj(0, lam2 * y)) * exp(-I * omega * t1)
    vel_s = diff(u_s, t1)

    # solve for coeffs
    k1, k2, k3, k4, k5, k6, k7, k8 = symbols("k1, k2, k3, k4, k5, k6, k7, k8")
    eq1 = A * k1 + B * k2 - V_wall / 2
    eq2 = A * k3 + B * k4 - C * k5
    eq3 = A * k6 + B * k7 - C * k8
    (ans,) = linsolve([eq1, eq2, eq3], (A, B, C))

    from scipy.special import jv, yv

    ans = ans.subs(k1, jv(0, lam1 * (L_s + L_f)))
    ans = ans.subs(k2, yv(0, lam1 * (L_s + L_f)))
    ans = ans.subs(k3, jv(0, lam1 * L_s))
    ans = ans.subs(k4, yv(0, lam1 * L_s))
    ans = ans.subs(k5, -1j * omega * jv(0, lam2 * L_s))
    ans = ans.subs(k6, jv(1, lam1 * L_s))
    ans = ans.subs(k7, yv(1, lam1 * L_s))
    ans = ans.subs(
        k8, lam2 * jv(1, lam2 * L_s) / (mu_f * lam1) * (G - mu_s * 1j * omega)
    )
    vel_f = vel_f.subs(A, ans[0].simplify())
    vel_f = vel_f.subs(B, ans[1].simplify())
    vel_s = vel_s.subs(C, ans[2].simplify())

    vel_f += conjugate(vel_f)
    vel_s += conjugate(vel_s)

    vel_fl = lambdify([y, t1], vel_f)
    vel_sl = lambdify([y, t1], vel_s)

    eps = 1e-20

    Y = kwargs.pop("resolution", np.linspace(eps, (L_s + L_f), 30)).copy()

    def theory_axisymmetric_soft_slab_temporal(time):
        vel_comb = (Y < L_s) * np.real(vel_sl(Y, time * np.ones_like(Y))) + (
            Y >= L_s
        ) * np.real(vel_fl(Y, time * np.ones_like(Y)))
        return vel_comb

    return theory_axisymmetric_soft_slab_temporal, Y


if __name__ == "__main__":
    # Input variables
    Re = 10.0
    Er = 0.25
    rho_f = 1.0
    L = 0.4
    zeta = 1.0  # L_f / L_s
    V_wall = 1.0
    omega = 2 * np.pi

    # Compute relevant variables
    L_f = L * 0.5 * zeta / (1.0 + zeta)
    L_s = L * 0.5 - L_f
    shear_rate = V_wall / omega / (L_s + L_f)
    nu_f = shear_rate * omega * L_f**2 / Re
    G = rho_f * nu_f * shear_rate * omega / Er

    # Plotting domain
    y = np.linspace(1e-20, L_s + L_f, 100)
    time = np.linspace(0.1, 1, 10)

    theory_axisymmetric_soft_slab_temporal, Y = theory_axisymmetric_soft_slab_spatial(
        L_f, L_s, shear_rate, omega, G, V_wall, rho_f, nu_f, resolution=y, nu_ratio=0.0
    )

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from periodic_soft_slab_post_processing import soft_slab_plotset

    cmap = soft_slab_plotset()
    fig, ax = plt.subplots(figsize=(10, 8))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize())
    sm.set_array([])
    for t in time:
        result = theory_axisymmetric_soft_slab_temporal(t)
        mappable = ax.plot(
            result / V_wall, y / (L_s + L_f), lw=2, color=cmap((t - 0.1) / 0.9)
        )

    cbar = fig.colorbar(sm, ax=ax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1.0"])
    cbar.ax.tick_params(size=0)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0], [-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.savefig("theory_plot.eps", format="eps")
    fig.show()
