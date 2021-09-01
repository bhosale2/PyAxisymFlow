import numpy as np


def zalesak_disc_LS(X, Y, x_cm, y_cm , R, s, offset):
    """
    returns the indicator for a Zalesak disc with:
    x_cm, y_cm: centre of the circle
    R: radius of the circle
    s: thickness of the gap
    offset: offset of the gap from the other end
    """
    phi = 0 * X
    H1 = np.sqrt((X - x_cm) ** 2 + (Y - y_cm) ** 2) - R < 0
    gap_left = x_cm - 0.5 * s
    gap_right = x_cm + 0.5 * s
    gap_top = y_cm + R - offset
    H2 = (X > gap_left) * (X < gap_right) * (Y < gap_top) * H1
    phi += 1
    phi = phi - 2 * H1 + 2 * H2
    return phi
