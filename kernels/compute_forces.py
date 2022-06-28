import numpy as np

def compute_force_on_body(R, part_char_func, rho_f, brink_lam, u_z, U_z_cm_part, part_vol, dt, diff):
    """
    computes brinkamn penalization force [1] and unsteady forces [2]
    [1]Bhosale, Yashraj, Tejaswin Parthasarathy, and Mattia Gazzola. "A remeshed vortex method for mixed rigid/soft body fluid–structure interaction." 
    Journal of Computational Physics 444 (2021): 110577.
    [2]Uhlmann, Markus. "An immersed boundary method with direct forcing for the simulation of particulate flows."
    Journal of computational physics 209.2 (2005): 448-476.
    """
    F_pen = rho_f * brink_lam * np.sum(R * part_char_func * (u_z - U_z_cm_part))
    F_un =  (diff*part_vol) /dt

    return F_pen,F_un 
