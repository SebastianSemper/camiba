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


def bpdn_1d(
    A,              # system matrix
    b,              # measurements
    x_init,         # inital iterate
    rho,            # parameter of augmented lagrangian
    alpha,          # thresholding parameter
    num_maxsteps    # maximum number of steps
):
    """
    Basis Pursuit Denoising
    """
    numN, numM = A.shape

    x = np.zeros(numM)
    z = np.copy(x_init)
    u = np.zeros(numM)

    AAt = A.dot(A.conj().T)

    P = np.eye(numM) - A.conj().T.dot(npl.solve(AAt, A))

    q = A.conj().T.dot(npl.solve(AAt, b))

    x_hat = alpha*x + (1-alpha)*z
    u = u + (x_hat - z)

    for ii in range(num_maxsteps):
        x = P.dot(z-u) + q

        x_hat = alpha*x + (1-alpha)*z
        z = soft_thrshld(x_hat+u, 1.0/rho)

        u = u + (x_hat - z)

    return z


def anm_lse_1d(
    A,          # compression matrix
    AH,         # A^H
    AHA,        # A^H * A
    y,          # measurements
    rho,        # augmented lagrangian parameter
    tau,        # regularization parameter
    steps       # maximum number of steps
):
    """
    Atomic Norm Denoising for 1D Line Spectral Estimation


    """
    dtype = np.promote_types(A.dtype, y.dtype)

    K, L = A.shape

    I = np.eye(L)
    e1 = -L * .5 * (tau / rho) * I[0]
    rhoInv = 1. / rho
    tauHalf = -.5 * tau

    Inv = npl.inv(AHA + 2 * rho * I)

    x = np.zeros((L), dtype)
    t = 0
    u = np.zeros((L), dtype)
    T = np.zeros((L + 1, L + 1), dtype)
    Lb = np.zeros((L + 1, L + 1), dtype)
    Z = np.zeros((L + 1, L + 1), dtype)

    Winv = 1. / (np.linspace(L, 1, L))
    Winv[0] = 2. / L

    for ii in range(steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
            AH.dot(y) + 2 * (Lb[:L, L] + rho * Z[:L, L])
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


def calcW(arrD):
    """
        Calculate Weighting Matrix in Derivative
    """

    tenW = np.ones((*arrD,))
    return _calcWrec(arrD, tenW)


def _calcWrec(arrD, tenW):
    """
        Recursion Function in Weight Matrix Calculation
    """
    # number of dimensions in current level
    numD = arrD.shape[0]

    # get size of resulting block toeplitz matrix
    prdD0 = np.prod(arrD)

    # get an array of all partial sequential products
    # starting at the front
    prdD1 = np.prod(arrD[1:])

    arrF = 2 * (arrD[0] - (np.arange(arrD[0]) + 1) + 1)
    arrF[0] -= arrD[0]

    if numD > 1:
        for nn in range(arrD[0]):
            _calcWrec(arrD[1:], tenW[nn])

    for ii in range(arrD[0]):
        tenW[ii] *= arrF[ii]


def anm_lse_nd(A, AH, AHA, y, rho, tau, steps, arrD):

    dtype = np.promote_types(A.dtype, y.dtype)

    K, L = A.shape

    # number of dimensions
    D = len(arrD.shape)

    I = np.eye(L)
    e1 = -L * .5 * (tau / rho) * I[0]
    rhoInv = 1. / rho
    tauHalf = -.5 * tau

    AHy = AH.dot(y)
    Inv = npl.inv(AHA + 2 * rho * I)

    x = np.zeros((L), dtype)

    t = 0

    u = np.zeros((*arrD,), dtype)

    T = np.zeros((L + 1, L + 1), dtype)

    Lb = np.zeros((L + 1, L + 1), dtype)
    Z = np.zeros((L + 1, L + 1), dtype)

    Winv = 1. / calcW(arrD)

    for ii in range(steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
                AHy + 2 * (Lb[:L, L] + rho * Z[:L, L])
            )

        u = Winv * (
                ToepAdj(arrD, Z[:L, :L] + rhoInv * Lb[:L, :L]) + e1
            )

        T[:L, :L] = Toep(arrD, u)
        T[:L, L] = x
        T[L, :L] = np.conj(x)
        T[L, L] = t

        Zspec, Zbase = npl.eigh(T - rhoInv * Lb)

        arrPos = (Zspec > 0)

        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)

    return (x, Toep(u), t)
