import numpy as np


def superellipse_LS(X, Y, x_cm, y_cm, a_left, a_right, n_left, n_right, b):
    """
    returns the indicator for a asymmetric superellipse: (x / a)^n + (y / b)^n  = 1 with
    x_cm, y_cm: centre of the superellipse
    a_left: left side semi major axis
    a_right: right side semi major axis
    n_left: left side 'superness' power
    n_right: right side 'superness' power
    b: common semi minor axis
    rotations needs to be added
    """
    phi = np.ones((X.shape[0], X.shape[1]))
    Xs = X - x_cm
    Ys = Y - y_cm
    H1 = (Xs < 0) * ((Xs / a_left) ** n_left + (Ys / b) ** n_left < 1)
    H2 = (Xs > 0) * ((Xs / a_right) ** n_right + (Ys / b) ** n_right < 1)
    H = np.logical_or(H1, H2)
    phi -= 2 * H
    return phi
