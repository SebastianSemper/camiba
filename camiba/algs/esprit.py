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
from ..cs.soe import _largest_div, _find_block_length


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


def smoothing(arr_y, arr_d, arr_p):
    """
    R-dimensional smoothing for single snapshot ESPRIT

    Parameters
    ----------

    arr_y : ndarray
        array containing the measurements
    arr_d : ndarray
        array containing the size of each dimension
    arr_k : ndarray
        array containing the block advance in each dimension

    Returns
    -------
    ndarray
    generated measurements
    """

    arr_l = np.empty_like(arr_d)
    arr_k = np.empty_like(arr_d)

    for ii in range(arr_d.shape[0]):
        arr_l[ii] = _find_block_length(arr_d[ii], arr_p[ii])
        arr_k[ii] = int((arr_d[ii] - arr_l[ii]) / arr_p[ii] + 1)

    lst_y = []

    smoothing_rec(
        arr_y.reshape((*arr_d,)),
        arr_p,
        arr_k,
        arr_l,
        lst_y,
        0
        )

    return (np.array(lst_y), arr_l)


def smoothing_rec(arr_y, arr_p, arr_k, arr_l, lst_y, axis):
    if arr_p.shape[0] == 0:
        lst_y.append(np.copy(arr_y).reshape((-1)))
    else:
        for ii in range(arr_k[0]):
            arr_z = np.swapaxes(arr_y, 0, axis)
            smoothing_rec(
                arr_z[ii*arr_p[0]:(ii*arr_p[0] + arr_l[0]), :],
                arr_p[1:],
                arr_k[1:],
                arr_l[1:],
                lst_y,
                axis+1
            )



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
        arr_res[:, ii] = np.angle(
                npl.eigvals(
                    lst_Phi_hat[ii]
                )
            )

    return arr_res


def _build_smooth_subsel(arr_p, arr_k, arr_l, num_N):

    ten_J = np.empty((np.prod(arr_k), np.prod(arr_l), num_N))

    for dd in range(arr_p.shape[0]):
        mat_eye = np.eye(arr_l[dd])
        mat_zero = np.zeros((arr_l[dd], arr_l[dd] * arr_k[dd]))
        ten_J_tmp = np.ones(1)
        for ii in range(arr_k[dd]):
            mat_tmp = np.copy(mat_zero)
            mat_tmp[:, arr_p[dd]*ii: arr_p[dd]*ii + arr_l[dd]] = mat_eye[:]
            ten_J_tmp = np.kron(ten_J_tmp, mat_tmp)
        print(ten_J_tmp.shape)


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
