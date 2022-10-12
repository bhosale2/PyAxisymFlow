import pickle
import numpy as np
from numpy import linalg as la

from periodic_soft_slab import simualte_periodic_soft_slab
from periodic_soft_slab_post_processing import plot_convergence


if __name__ == "__main__":
    grid_sizes = [64, 128, 256, 512]
    filename = "converge_data.pkl"

    save_result = False

    if save_result:
        results = []

        for grid_size_r in grid_sizes:
            result = simualte_periodic_soft_slab(
                grid_size_r=grid_size_r,
                Re=10,
                Er=0.25,
                domain_AR=8,
                match_resolution=True,
                compare_with_theory=False,
                plot_contour=False,
            )
            results.append(result)

        with open(filename, "wb") as f:
            pickle.dump(results, f)

    else:
        with open(filename, "rb") as f:
            results = pickle.load(f)

    nondim_t_list = np.arange(0.2, 1.2, 0.2)
    dx = 0.5 / np.array(grid_sizes)  # 0.5 is the domain r-length
    offset = 10.0
    l2_error = []
    linf_error = []

    for i, result in enumerate(results):
        time_history = result["time_history"]
        sim_pos = result["sim_positions"]
        sim_vel = result["sim_velocities"]
        theory_pos = result["theory_positions"]
        theory_vel = result["theory_velocities"]

        # Find the indices that corresponds to the desired nondimensional time
        idx_time = []
        for nondim_t in nondim_t_list:
            idx_time.append(np.argmin(np.abs(time_history - offset - nondim_t)))
        idx_time = np.array(idx_time, dtype=int)

        l2_error.append(
            la.norm(
                sim_vel[idx_time, :] - theory_vel[idx_time, :],
                ord=2,
                axis=1,
            )
            * dx[i]
        )

        linf_error.append(
            la.norm(
                sim_vel[idx_time, :] - theory_vel[idx_time, :],
                ord=np.inf,
                axis=1,
            )
        )

    l2_error = np.array(l2_error)
    linf_error = np.array(linf_error)

    plot_convergence(dx=dx, l2_error=l2_error, linf_error=linf_error)
