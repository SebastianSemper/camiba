# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import numpy.linalg as npl


def one_d(mat_cov, num_s):
    """
        One-dimensional ESPRIT

    Parameters
    ----------

    mat_cov : ndarray
        covariance matrix of the signal
    num_s : int
        number of frequencies in the signal

    Returns
    -------
    ndarray
        extracted frequencies
    """
    U, S, V = npl.svd(mat_cov)
    Uhat = npl.lstsq(U[: -1, :num_s], U[1:, :num_s])

    phi = np.mod(np.angle(npl.eigvals(Uhat[0])), 2 * np.pi)

    return phi


def two_d(mat_cov, num_s):
    """
        Two-dimensional ESPRIT

    Parameters
    ----------

    mat_cov : ndarray
        covariance matrix of the signal
    num_s : int
        number of frequencies in the signal

    Returns
    -------
    ndarray
        extracted frequencies
    """

    return 0
