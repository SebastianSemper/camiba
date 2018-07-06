# This file is part of Camiba.
#
# Camiba is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Camiba is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Camiba. If not, see <http://www.gnu.org/licenses/>.
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
    U, _, _ = npl.svd(mat_cov)
    Uhat = npl.lstsq(U[: -1, :num_s], U[1:, :num_s], rcond=None)

    phi = np.mod(np.angle(npl.eigvals(Uhat[0])) / (2 * np.pi), 1)

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


def r_d(mat_cov, arr_d, num_s):
    """
        R-dimensional ESPRIT

    Parameters
    ----------

    mat_cov : ndarray
        covariance matrix of the signal
    arr_d : ndarray
        array containing the size of each dimension
    num_s : int
        number of frequencies in the signal

    Returns
    -------
    ndarray
        extracted frequencies
    """
    # get the signal subspace
    mat_U, _, _ = npl.svd(mat_cov)

    # list of estimated U
    lst_Phi_hat = []
    lst_U_hat = []

    # extract the number of dimensions
    num_r = len(arr_d)

    # initialize the solution array
    arr_res = np.zeros((num_s, num_r))

    # get an estimate of the signal subspace
    # mat_U, arr_S, mat_V = npl.svd(mat_cov)
    mat_subsel_1, mat_subsel_2 = _build_subsel(arr_d)

    for ii, dd in enumerate(arr_d):

        # iterative solve the least squares problems
        lst_Phi_hat.append(
            npl.lstsq(
                mat_subsel_1[ii].dot(mat_U[:, :num_s]),
                mat_subsel_2[ii].dot(mat_U[:, :num_s]),
                rcond=None
            )[0]
        )

        arr_res[:, ii] = np.mod(
            np.angle(
                npl.eigvals(
                    lst_Phi_hat[ii]
                )
            ) / (2 * np.pi),
            1
        )

    return arr_res


def _build_subsel(arr_d):

    J1 = []
    J2 = []

    for ii, dd in enumerate(arr_d):
        K1 = np.eye(dd - 1, dd)
        K2 = np.flipud(np.fliplr(K1))
        num_low = np.prod(arr_d[:ii])
        num_up = np.prod(arr_d[ii+1:])

        J1.append(
            np.kron(np.kron(np.eye(num_low), K1), np.eye(num_up))
        )
        J2.append(
            np.kron(np.kron(np.eye(num_low), K2), np.eye(num_up))
        )
    return (J1, J2)
