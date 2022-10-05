import pickle
import numpy as np
from numpy import linalg as la

from periodic_soft_slab import simualte_periodic_soft_slab
from periodic_soft_slab_post_processing import plot_convergence


if __name__ == "__main__":
    grid_sizes = [128, 256, 512, 1024]
    filename = "converge_data.pkl"

    save_result = False

    if save_result:
        results = []

        for grid_size_r in grid_sizes:
            result = simualte_periodic_soft_slab(
                grid_size_r=grid_size_r,
                Re=10,
                Er=0.25,
                domain_AR=16,
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
    offset = 10.0
    l2_error = []
    linf_error = []
    for result in results:
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

        # Find the indices in simulation results that matches theory positions
        idx_pos = []
        for theory_p in theory_pos:
            idx_pos.append(np.argmin(np.abs(sim_pos - theory_p)))
        idx_pos = np.array(idx_pos, dtype=int)

        l2_error.append(
            la.norm(
                sim_vel[idx_time, :][:, idx_pos] - theory_vel[idx_time, :],
                ord=2,
                axis=1,
            )
            / sim_pos.shape[0]
        )

        linf_error.append(
            la.norm(
                sim_vel[idx_time, :][:, idx_pos] - theory_vel[idx_time, :],
                ord=np.inf,
                axis=1,
            )
            / sim_pos.shape[0]
        )

    l2_error = np.array(l2_error)
    linf_error = np.array(linf_error)
    dx = 0.5 / np.array(grid_sizes)
    plot_convergence(dx=dx, l2_error=l2_error, linf_error=linf_error)
