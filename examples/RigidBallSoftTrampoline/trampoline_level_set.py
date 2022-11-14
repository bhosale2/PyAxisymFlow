import numpy as np
from matplotlib import pyplot as plt


def trampoline_level_set(
    inp_X,
    inp_Y,
    center_basal_x,
    distance_between_centers_x,
    center_y,
    r_basal,
    r_tip,
):
    """Gives the level set of a Trampoline"""
    # 1. Basal circle
    # 2. Straight line above and below symmetry plane
    # 3. Tip circle
    # center_y = 0.5
    # r_basal = 0.1
    # r_tip = 0.05
    # distance_between_centers_x = 0.3
    # center_basal_x = 0.2
    center_tip_x = center_basal_x + distance_between_centers_x
    phi = 0.0 * inp_X

    basal_theta = np.arccos((r_basal - r_tip) / distance_between_centers_x)
    THETA = np.arctan2((inp_Y - center_y), (inp_X - center_basal_x))
    idx_one = np.abs(THETA) > basal_theta

    # In this region, the level set should be circle distance desu
    phi[idx_one] = (
        np.sqrt(
            (inp_X[idx_one] - center_basal_x) ** 2 + (inp_Y[idx_one] - center_y) ** 2
        )
        - r_basal
    )

    # Do for the smaller circle next
    THETA = np.arctan2((inp_Y - center_y), (inp_X - center_tip_x))
    idx_two = np.abs(THETA) < basal_theta
    phi[idx_two] = (
        np.sqrt((inp_X[idx_two] - center_tip_x) ** 2 + (inp_Y[idx_two] - center_y) ** 2)
        - r_tip
    )

    # Now shift FOR to the point of attachment and calculate distance to line
    attach_basal_x = center_basal_x + r_basal * np.cos(basal_theta)
    attach_basal_y = center_y + r_basal * np.sin(basal_theta)

    # in this shifted coordinates, the line is y = -cot(basal_theta)*x
    shifted_X = inp_X - attach_basal_x
    shifted_Y = inp_Y - attach_basal_y

    cot_basal_theta = 1.0 / np.tan(basal_theta)
    dist = (shifted_Y + cot_basal_theta * shifted_X) / np.sqrt(1 + cot_basal_theta**2)
    mask_idx = (inp_Y > center_y) & np.logical_not(np.logical_or(idx_one, idx_two))
    phi[mask_idx] = dist[mask_idx]

    # Do the same for down under
    attach_basal_x = center_basal_x + r_basal * np.cos(basal_theta)
    attach_basal_y = center_y - r_basal * np.sin(basal_theta)

    # in this shifted coordinates, the line is y = cot(basal_theta)*x
    shifted_X = inp_X - attach_basal_x
    shifted_Y = inp_Y - attach_basal_y

    dist = -(shifted_Y - cot_basal_theta * shifted_X) / np.sqrt(
        1 + cot_basal_theta**2
    )
    mask_idx = (inp_Y < center_y) & np.logical_not(np.logical_or(idx_one, idx_two))
    phi[mask_idx] = dist[mask_idx]

    return phi


def __internal_level_set_test(inp_X, inp_Y, phi_input):
    import scipy.sparse as spp

    del_x = inp_X[0, 1] - inp_X[0, 0]
    del_y = inp_Y[1, 0] - inp_Y[0, 0]
    print(del_x, del_y)
    grid_size = inp_X.shape
    mat_size = grid_size[0] * grid_size[1]
    grad_x = spp.diags([-1, 1], [-1, 1], shape=(mat_size, mat_size)) / 2.0 / del_x
    grad_y = (
        spp.diags([-1, 1], [-grid_size[1], grid_size[1]], shape=(mat_size, mat_size))
        / 2.0
        / del_y
    )
    phi_gradient = (
        (
            grad_x
            @ phi.reshape(
                -1,
            )
        )
        ** 2
        + (
            grad_y
            @ phi.reshape(
                -1,
            )
        )
        ** 2
    ).reshape(grid_size[0], grid_size[1])
    return np.sqrt(phi_gradient)


if __name__ == "__main__":
    n_points = 128
    del_x = 1.0 / float(n_points)
    x = np.linspace(0.5 * del_x, 1.0 - del_x, n_points)
    X, Y = np.meshgrid(x, x)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    phi = octopus_arm_level_set(
        X,
        Y,
        center_basal_x=0.6,
        distance_between_centers_x=0.25,
        center_y=0.4,
        r_basal=0.04,
        r_tip=0.04,
    )
    activation_phi = octopus_arm_level_set(
        X,
        Y,
        center_basal_x=0.6,
        distance_between_centers_x=0.25,
        center_y=0.4,
        r_basal=0.02,
        r_tip=0.02,
    )

    # Level set of an ellipse
    phi_two = np.fliplr(phi)
    phi_ellipse = np.sqrt((X - 0.5) ** 2 / 0.1**2 + (Y - 0.6) ** 2 / 0.18**2) - 1

    idx = (phi > 0.0) * (phi_two > 0.0) * (phi_ellipse > 0.0)

    phi_new = 0.0 * X
    phi_new[idx] = phi[idx] + phi_two[idx] + phi_ellipse[idx]
    # ax.contour(X, Y, phi_new, levels=[0], colors="white", linewidths=3.0)
    # cont = ax.contourf(X, Y, phi_new, levels=30, cmap=plt.get_cmap("Reds"))

    # phi = Y + X
    # ax.contour(X, Y, phi, levels=[0], colors="white", linewidths=3.0)
    # cont = ax.contourf(X, Y, phi, levels=30, cmap=plt.get_cmap("Reds"))

    ax.contour(X, Y, phi, levels=[0], colors="white", linewidths=3.0)
    cont = ax.contourf(X, Y, phi, levels=30, cmap=plt.get_cmap("Reds"))
    ax.contour(X, Y, activation_phi, levels=[0], colors="black", linewidths=3.0)

    ax.set_aspect("equal")
    plt.show()
