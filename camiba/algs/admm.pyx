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
    num_M = A.shape[1]

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


cpdef anm_lse_r_d(
    np.ndarray A,
    np.ndarray AH,
    np.ndarray AHA,
    np.ndarray y,
    np.float64_t rho,
    np.float64_t tau,
    np.int64_t num_steps,
    np.ndarray arr_d,
    np.float64_t mu,
    np.float64_t nu,
    np.float64_t alpha,
    bint adaptive=False,
    bint verbose=False,
    bint debug=False,
    callback=None,
    init=False,
    u_init=None
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
    mu : float
    nu : float
    alpha : float
    adaptive : bool
        adaptivity flag
    verbose : bool
        verbosity flag
    debug : bool
        verbosity flag
    callback : callable
        callable to evaluate after each iteration

    Returns
    -------
    (ndarray, ndarray, ndarray)
        x, T(u), t
    """

    # debugging
    if debug:
        callback_vals = {
            'est': [],
            'gradx': [],
            'gradu': [],
            'gradt': [],
            'rho': [],
            'rk': [],
            'sk': []
        }

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

    cdef np.ndarray AHy = AH.dot(y)

    cdef np.ndarray x = np.random.randn(L, M) + 1j * np.random.randn(L, M)

    cdef np.ndarray t = np.random.randn(M, M) + 1j * np.random.randn(M, M)

    cdef np.ndarray u = np.random.randn(*arr_s) + 1j * np.random.randn(*arr_s)
    if init:
        u = HermToepInv(u_init, arr_d)
    # cdef np.ndarray u = HermToepInv(
    #     np.cov(AH.dot(y)),
    #     arr_d
    # )

    cdef np.ndarray T = np.empty((L + M, L + M), dtype=prbDtype)

    hT = hermToep(u)

    # print(T[:L, L:].shape, x[:].shape)
    T[:L, :L] = 0.5 * (hT + hT.conj().T)
    T[:L, L:] = x
    T[L:, :L] = x.conj().T
    T[L:, L:] = t

    # cdef np.ndarray Lb = np.eye((L + M)).astype(prbDtype)
    # cdef np.ndarray Z = np.eye((L + M)).astype(prbDtype)
    # cdef np.ndarray Zold = np.eye((L + M)).astype(prbDtype)
    cdef np.ndarray Lb = np.zeros((L + M, L + M)).astype(prbDtype)
    cdef np.ndarray Z = np.zeros((L + M, L + M)).astype(prbDtype)
    cdef np.ndarray Zold = np.zeros((L + M, L + M)).astype(prbDtype)

    cdef np.ndarray Winv = hermToepAdj(
        1.0 * np.ones(
            (L, L),
            dtype=prbDtype
        ),
        u
    )
    Winv[Winv < 0.01] = 1.0
    Winv = 1.0 / Winv

    e1 = np.zeros((L, L), dtype=prbDtype)
    e1.flat[0] = 1

    e1 = hermToepAdj(e1, u)

    cdef int ii

    cdef np.ndarray Zspec = np.empty(L + M, dtype=prbDtype)
    cdef np.ndarray Zbase = np.empty((L + M, L + M), dtype=prbDtype)

    cdef np.float64_t rk
    cdef np.float64_t sk

    for ii in range(num_steps):

        num_tm_s = tm.time()
        t[:] = Z[L:, L:] + (1. / rho) * (
            Lb[L:, L:] - 0.5 * tau * np.eye(M)
        )

        x[:] = npl.solve(
            AHA + 2.0 * rho * mat_eye,
            AHy + 2.0 * (Lb[:L, L:] + rho * Z[:L, L:])
        )

        u[:] = Winv * (
            hermToepAdj((rho * Z[:L, :L] + Lb[:L, :L]), u)
            - 0.5 * (tau / rho) * e1
        )

        hT = hermToep(u)

        T[:L, :L] = 0.5 * (hT + hT.conj().T)
        T[:L, L:] = x
        T[L:, :L] = x.conj().T
        T[L:, L:] = t

        Zspec[:], Zbase[:] = npl.eigh(T - (1. / rho) * Lb)

        ZspecNorm = alpha * npl.norm(Zspec)
        arrPos = (Zspec > ZspecNorm)

        Zold[:] = Z[:]
        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)
        Lb = 0.5 * (Lb + Lb.T.conj())

        rk = npl.norm(Zspec[Zspec<=ZspecNorm]) / np.sqrt(Zspec[:].shape[0])
        sk = npl.norm(Z - Zold) / np.sqrt(
            Z[:].shape[0] * Z[:].shape[1]
        )

        if adaptive:
            if rk > (mu * sk):
                rho *= nu
                print("Increasing rho to %f" % (rho))
            elif sk > (mu * rk):
                rho /= nu
                print("Decreasing rho to %f" % (rho))

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

        if debug:
            callback_vals['est'].append(callback(hT, ii))
            # callback_vals['err'].append(callback(hT)[0])
            # callback_vals['freq'].append(callback(hT)[1])
            callback_vals['gradx'].append(
                npl.norm(0.5 * (AHA.dot(x)
                - AH.dot(y)
                - Lb[:L, L:]
                - rho * (Z[:L, L:] - x)))
            )
            callback_vals['gradu'].append(npl.norm(
                0.5 * tau * e1
                - hermToepAdj((rho * Z[:L, :L] + Lb[:L, :L]), u)
                + rho * u * (1.0 / Winv)
            ))
            callback_vals['gradt'].append(npl.norm(
                0.5 * tau * np.eye(M)
                - Lb[L:, L:]
                - rho * (Z[L:, L:] - t)
            ))
            callback_vals['rk'].append(rk)
            callback_vals['sk'].append(sk)
            callback_vals['rho'].append(rho)


    if debug:
        return (hermToep(u), callback_vals)
    else:
        return (hermToep(u), (0,0))
