import numpy as np


def cross_LS(X, Y, x_cm, y_cm, L, B):
    """
    returns the indicator for a cross with:
    x_cm, y_cm: centre of the cross
    L: length
    B: breadth
    """
    phi = np.ones((X.shape[0], X.shape[1]))
    H1 = (
        (X > (x_cm - 0.5 * B))
        * (X < (x_cm + 0.5 * B))
        * (Y > (y_cm - 0.5 * L))
        * (Y < (y_cm + 0.5 * L))
    )
    H2 = (
        (X > (x_cm - 0.5 * L))
        * (X < (x_cm + 0.5 * L))
        * (Y > (y_cm - 0.5 * B))
        * (Y < (y_cm + 0.5 * B))
    )
    H = np.logical_or(H1, H2)
    phi -= 2 * H
    return phi
