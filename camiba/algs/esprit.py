# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as npl


def one_d(matC, numS):
    """
        One-dimensional ESPRIT
    """
    U, S, V = npl.svd(matC)
    Uhat = npl.lstsq(U[: -1, :numS], U[1:, :numS])

    phi = np.mod(np.angle(npl.eigvals(Uhat[0])), 2 * np.pi)

    return phi


def two_d(matC, numS):
    """
        Two-dimensional ESPRIT
    """

    return 0
