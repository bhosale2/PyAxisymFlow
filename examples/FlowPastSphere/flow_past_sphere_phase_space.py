import pickle
import os
import matplotlib.pyplot as plt

from flow_past_sphere import simulate_flow_past_sphere
from post_processing import compare_with_emp_eqn, compare_with_exp_data, plot_Cd_vs_Re

if __name__ == "__main__":
    sim_Re = range(25, 160, 10)
    filename = "drag_data.txt"

    load_data = False

    if load_data:
        with open(filename, "rb") as f:
            sim_results = pickle.load(f)

    else:
        sim_results = []
        for Re in sim_Re:
            print(f"Simulating Re = {Re}")
            sim_results.append(simulate_flow_past_sphere(Re, converge_drag=True))

        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, "wb") as f:
            pickle.dump(sim_results, f)

    fig, ax = plot_Cd_vs_Re(sim_Re, sim_results)
    compare_with_exp_data(ax)
    compare_with_emp_eqn(ax)

    plt.savefig("flow_past_sphere_phase_space.png")
    plt.show()
