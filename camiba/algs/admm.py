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
This module provides different specialiced implementations of the well known
method of alternating direction multipliers [ADMM]_. Since the method
is so general, many variants exist and many optimizations problems can be
recast into a problem, which can be solved by a specific ADMM.
"""

import numpy as np
import numpy.linalg as npl
from ..linalg.basic import soft_thrshld
from ..linalg.multilevel import *


def bpdn_1d(A, b, x_init, rho, alpha, num_steps):
    """
    Basis Pursuit Denoising

    For given matrix A, vector b and z > 0, ADDM approximates a solution to

    min ||x||_1 s.t. ||A * x - b||_2 < z.

    The algorithm needs a thresholding parameter, an initial guess for a
    solution and a parameter for the augmented lagrangian.

    Parameters
    ----------

    A : ndarray
        system matrix
    b : ndarray
        measurement vector
    x_init : ndarray
        initial solution guess
    rho : float
        parameter for augmented lagrangian
    alpha : float
        thresholding parameter
    num_steps : int
        number of steps

    Returns
    -------
    ndarray
        approximated solution to BPDN
    """
    num_M = A.shape

    x = np.zeros(num_M)
    z = np.copy(x_init)
    u = np.zeros(num_M)

    AAt = A.dot(A.conj().T)

    P = np.eye(num_M) - A.conj().T.dot(npl.solve(AAt, A))

    q = A.conj().T.dot(npl.solve(AAt, b))

    x_hat = alpha*x + (1-alpha)*z
    u = u + (x_hat - z)

    for ii in range(num_steps):
        x = P.dot(z-u) + q

        x_hat = alpha*x + (1-alpha)*z
        z = soft_thrshld(x_hat+u, 1.0/rho)

        u = u + (x_hat - z)

    return z


def anm_lse_1d(A, AH, AHA, y, rho, tau, num_steps):
    """
    Atomic Norm Denoising for 1D Line Spectral Estimation

    This ADMM approximates a solution to

    min_[x,u,t] 1/(2n) trace T(u) + 1/2 t
    s.t.
    [[T(u), x], [x^H, t]] >= 0, ||b - Ax||_2 < z

    for given matrix A, vector b and z > 0. Moreover, T(u) maps the vector
    u to a Hermitian Toeplitz matrix defined by u.

    Parameters
    ----------

    A : ndarray
        system matrix
    AH : ndarray
        Hermitian transpose of system matrix
    AHA : ndarray
        Gram matrix of system matrix
    b : ndarray
        measurement vector
    rho : float
        parameter for augmented lagrangian
    alpha : float
        thresholding parameter
    num_steps : int
        number of steps

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, T(u), t
    """
    dtype = np.promote_types(A.dtype, y.dtype)

    K, L = A.shape

    mat_eye = np.eye(L)
    e1 = -L * .5 * (tau / rho) * mat_eye[0]
    rhoInv = 1. / rho
    tauHalf = -.5 * tau

    Inv = npl.inv(AHA + 2 * rho * mat_eye)

    x = np.zeros((L), dtype)
    t = 0
    u = np.zeros((L), dtype)
    T = np.zeros((L + 1, L + 1), dtype)
    Lb = np.zeros((L + 1, L + 1), dtype)
    Z = np.zeros((L + 1, L + 1), dtype)

    Winv = 1. / (np.linspace(L, 1, L))
    Winv[0] = 2. / L

    for ii in range(num_steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
            AH.dot(b) + 2 * (Lb[:L, L] + rho * Z[:L, L])
        )

        u = Winv * (
            ToepAdj(Z[:L, :L] + rhoInv * Lb[:L, :L]) + e1
        )

        T[:L, :L] = spToep(u.conj())
        T[:L, L] = x
        T[L, :L] = np.conj(x)
        T[L,  L] = t

        Zspec, Zbase = npl.eigh(T - rhoInv * Lb)
        arrPos = (Zspec > 0)

        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)

    return (x, T[:L, :L], t)


def _calc_w(arr_d):
    """
        Calculate Weighting Matrix in Derivative

    This is needed in anm_lse_nd to generate the diagonal of a weighting
    matrix during the gradient decent step.
    """

    ten_w = np.ones((*arr_d,))
    return _calc_wrec(arr_d, ten_w)


def _calc_wrec(arr_d, ten_w):
    """
        Recursion Function in Weight Matrix Calculation

    This is a recursive function used in _calc_w.
    """
    # number of dimensions in current level
    num_d = arr_d.shape[0]

    # get size of resulting block toeplitz matrix
    prdD0 = np.prod(arr_d)

    # get an array of all partial sequential products
    # starting at the front
    prdD1 = np.prod(arr_d[1:])

    arrF = 2 * (arr_d[0] - (np.arange(arr_d[0]) + 1) + 1)
    arrF[0] -= arr_d[0]

    if num_d > 1:
        for nn in range(arr_d[0]):
            _calc_wrec(arr_d[1:], ten_w[nn])

    for ii in range(arr_d[0]):
        ten_w[ii] *= arrF[ii]


def anm_lse_nd(A, AH, AHA, y, rho, tau, num_steps, arr_d):
    """
    Atomic Norm Denoising for 1D Line Spectral Estimation

    This ADMM approximates a solution to

    min_[x,u,t] 1/(2n) trace T(u) + 1/2 t
    s.t.
    [[T(u), x], [x^H, t]] >= 0, ||b - Ax||_2 < z

    for given matrix A, vector b and z > 0. Moreover, T(u) maps the tensor of
    order l u to a l-level Hermitian Toeplitz matrix defined by u.

    Parameters
    ----------

    A : ndarray
        system matrix
    AH : ndarray
        Hermitian transpose of system matrix
    AHA : ndarray
        Gram matrix of system matrix
    b : ndarray
        measurement vector
    rho : float
        parameter for augmented lagrangian
    alpha : float
        thresholding parameter
    num_steps : int
        number of steps
    arr_d : ndarray
        dimension sizes

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, T(u), t
    """
    # detect datatype
    dtype = np.promote_types(A.dtype, y.dtype)

    # extract the shaped
    K, L = A.shape

    mat_eye = np.eye(L)
    e1 = -L * .5 * (tau / rho) * mat_eye[0]
    rhoInv = 1. / rho
    tauHalf = -.5 * tau

    AHy = AH.dot(y)
    Inv = npl.inv(AHA + 2 * rho * mat_eye)

    x = np.zeros((L), dtype)

    t = 0

    u = np.zeros((*arr_d,), dtype)

    T = np.zeros((L + 1, L + 1), dtype)

    Lb = np.zeros((L + 1, L + 1), dtype)
    Z = np.zeros((L + 1, L + 1), dtype)

    Winv = 1. / _calc_w(arr_d)

    for ii in range(steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
            AHy + 2 * (Lb[:L, L] + rho * Z[:L, L])
        )

        u = Winv * (
            ToepAdj(arr_d, Z[:L, :L] + rhoInv * Lb[:L, :L]) + e1
        )

        T[:L, :L] = Toep(arr_d, u)
        T[:L, L] = x
        T[L, :L] = np.conj(x)
        T[L, L] = t

        Zspec, Zbase = npl.eigh(T - rhoInv * Lb)

        arrPos = (Zspec > 0)

        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)

    return (x, Toep(u), t)
