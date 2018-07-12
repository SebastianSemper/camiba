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

import time as tm
import pdb
import numpy as np
import numpy.linalg as npl
from ..linalg.basic import soft_thrshld
from ..linalg.multilevel import *
cimport numpy as np

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def printdType(objlst, ns):
    for oo in objlst:
        print(namestr(oo, ns))
        try:
            print(oo.dtype)
        except:
            print(type(oo))

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
    y : ndarray
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

    arr_d = np.array([L])

    for ii in range(num_steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
            AH.dot(y) + 2 * (Lb[:L, L] + rho * Z[:L, L])
        )

        u = Winv * (
            hermToepAdj(arr_d, arr_d, Z[:L, :L] + rhoInv * Lb[:L, :L]) + e1
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


cdef np.ndarray dervA_Tu(
    np.ndarray A,
    np.ndarray ten_u
):
    return hermToepAdj(A - 0.5*A.flat[0], ten_u)

cpdef np.ndarray anm_lse_r_d(
    np.ndarray A,
    np.ndarray AH,
    np.ndarray AHA,
    np.ndarray y,
    np.float64_t rho,
    np.float64_t tau,
    np.int64_t num_steps,
    np.ndarray arr_d,
    bint verbose=False
):

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
    y : ndarray
        measurement vector
    rho : float
        parameter for augmented lagrangian
    tau : float
        thresholding parameter
    num_steps : int
        number of steps
    arr_d : ndarray
        dimension sizes
    verbose : bool
        verbosity flag

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, T(u), t
    """
    # detect datatype
    prbDtype = np.promote_types(A.dtype, y.dtype)

    # extract the shapes
    cdef int K = A[:].shape[0]
    cdef int L = A[:].shape[1]

    # account for the MMW case
    cdef int M = 1
    if len(y[:].shape) > 1:
        M = y[:].shape[1]

    # size of the array of defining elements of the ML-Toeplitz matrix
    cdef np.ndarray arr_s = 2 * arr_d - 1
    arr_s[0] = arr_d[0]

    cdef np.ndarray mat_eye = np.eye((L), dtype=prbDtype)

    cdef np.complex128_t tauHalf = -.5 * tau

    cdef np.ndarray AHy = AH.dot(y)

    cdef np.ndarray x = np.zeros((L, M), dtype=prbDtype)

    cdef np.ndarray t = np.zeros((M, M), dtype=prbDtype)

    cdef np.ndarray u = np.zeros((*arr_s,), dtype=prbDtype)

    cdef np.ndarray T = np.zeros((L + M, L + M), dtype=prbDtype)

    cdef np.ndarray Lb = np.zeros((L + M, L + M), dtype=prbDtype)
    cdef np.ndarray Z = np.zeros((L + M, L + M), dtype=prbDtype)
    cdef np.ndarray Zold = np.zeros((L + M, L + M), dtype=prbDtype)

    cdef np.ndarray Winv = hermToepAdj(
        1.0 * np.ones(
            (L, L),
            dtype=prbDtype
        # ) - 0.5*np.eye(
        #     (L),
        #     dtype=prbDtype
        ),
        u
    )
    Winv[Winv < 0.01] = 1.0
    Winv = 1.0 / Winv

    e1 = np.zeros((L, L), dtype=prbDtype)
    e1.flat[0] = - .25 * M * tau / rho

    e1 = hermToepAdj(e1, u)

    u[:] = e1[:]


    cdef int ii

    cdef np.ndarray Zspec = np.empty(L + M, dtype=prbDtype)
    cdef np.ndarray Zbase = np.empty((L + M, L + M), dtype=prbDtype)


    cdef np.float64_t mu = 10.0
    cdef np.float64_t nu = 2.0
    cdef np.float64_t rk
    cdef np.float64_t sk

    for ii in range(num_steps):

        num_tm_s = tm.time()
        t[:] = Z[L:, L:] + (1. / rho) * (Lb[L:, L:] + tauHalf)

        # print(AHy[:].shape)
        # print((AHA + 2.0 * rho * mat_eye).shape)
        # print((AHy + 2.0 * (Lb[:L, L:] + rho * Z[:L, L:]))[:].shape)
        x[:] = npl.solve(
            AHA + 2.0 * rho * mat_eye,
            AHy + 2.0 * (Lb[:L, L:] + rho * Z[:L, L:])
        )

        u[:] = Winv * (
            hermToepAdj((Z[:L, :L] + (1. / rho) * Lb[:L, :L]), u) + e1
        )

        hT = hermToep(u)

        # print(T[:L, L:].shape, x[:].shape)
        T[:L, :L] = 0.5 * (hT + hT.conj().T)
        T[:L, L:] = x
        T[L:, :L] = x.conj().T
        T[L:, L:] = t

        Zspec[:], Zbase[:] = npl.eigh(T - (2. / rho) * Lb)

        arrPos = (Zspec > 0)

        Zold[:] = Z[:]
        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)
        Lb = 0.5 * (Lb + Lb.T.conj())

        rk = np.linalg.norm(Zspec[Zspec<=0])
        sk = np.linalg.norm(Z - Zold)

        if rk > (mu * sk):
            rho *= nu
        elif sk > (mu * rk):
            rho /= nu

        if verbose:
            print(
                "Iteration step %d took %fs" % (ii + 1, tm.time() - num_tm_s)
            )
            print(
                "Current error is %f" % npl.norm(A.dot(x) - y)
            )
            print(
                "Current primal error is %f" % rk
            )
            print(
                "Current dual error is %f" % sk
            )


    return hermToep(u)
