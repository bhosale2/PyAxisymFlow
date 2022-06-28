import numpy as np

def compute_force_on_body(R, part_char_func, rho_f, brink_lam, u_z, U_z_cm_part, part_vol, dt, diff):
    """
    computes penalization and unsteady forces
    """
    F_pen = rho_f * brink_lam * np.sum(R * part_char_func * (u_z - U_z_cm_part))
    F_un =  (diff*part_vol) /dt

    return F_pen,F_un 
