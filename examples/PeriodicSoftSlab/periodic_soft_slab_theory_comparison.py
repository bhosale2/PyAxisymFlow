import numpy as np
import pickle

from periodic_soft_slab import simualte_periodic_soft_slab
from periodic_soft_slab_post_processing import plot_time_dependent_theory_comparison

if __name__ == "__main__":
    filename = "sim_data.pkl"
    save_result = False

    if save_result:
        result = simualte_periodic_soft_slab(
            grid_size_r=256,
            Re=10,
            Er=0.25,
            compare_with_theory=False,
            plot_contour=False,
        )

        with open(filename, "wb") as f:
            pickle.dump(result, f)

    else:
        with open(filename, "rb") as f:
            result = pickle.load(f)

    time_history = result["time_history"]
    sim_positions = result["sim_positions"]
    sim_velocities = result["sim_velocities"]
    theory_positions = result["theory_positions"]
    theory_velocities = result["theory_velocities"]

    nondim_t_list = np.array([0.75, 1.0])
    offset = 10.0  # Plot after 10 cycles so the system is dynamically stable

    idx = []
    for nondim_t in nondim_t_list:
        idx.append(np.argmin(np.abs(time_history - offset - nondim_t)))

    idx = np.array(idx, dtype=int)
    plot_time_dependent_theory_comparison(
        times=np.abs(time_history[idx] - offset),
        sim_r=sim_positions,
        sim_v_list=sim_velocities[idx],
        theory_r=theory_positions,
        theory_v_list=theory_velocities[idx],
    )
