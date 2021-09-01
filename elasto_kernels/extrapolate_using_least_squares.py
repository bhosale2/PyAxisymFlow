import numpy as np

from core.extrapolate_using_least_squares import (
    extrapolate_using_least_squares_till_first_order as extrapolate_using_least_squares,
    # extrapolate_using_least_squares_till_second_order as extrapolate_using_least_squares,
)


def extrapolate_eta_using_least_squares(
    inp_phi,
    phi_thresh_lower_bound: float,
    phi_thresh_upper_bound: float,
    inp_eta_X,
    inp_eta_Y,
    inp_x,
    inp_y,
):
    """
    Extrapolates the variables inp_eta_X, inp_eta_Y outside its defined zone

    inp_phi : level set field

    phi_thresh_lower_bound : demarcates the zone in which the variables are defined
    usually is 0.0 (i.e. inp_phi < 0.0 defines the area of definition)

    phi_thresh_upper_bound : demarcates the zone in which the variables need to be
    extrapolated. phi_thresh_upper_bound > phi_thresh_lower_bound logically, but
    an assert check is skipped for speed.

    inp_eta_X, inp_eta_Y : variables to be extrapolated

    inp_x, inp_y : 1D fields defining the grid points at (X,Y).
    """
    current_flag = np.asarray(inp_phi < phi_thresh_lower_bound, dtype=np.int16)
    # Tag zones where eta is needed in the current iteration
    target_flag = np.asarray(inp_phi < phi_thresh_upper_bound, dtype=np.int16)
    extrapolate_using_least_squares(
        current_flag, target_flag, inp_eta_X, inp_eta_Y, inp_x, inp_y
    )
