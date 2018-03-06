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
This module aims at making work with some more specific and but still generally
applicable stuff way easier.
"""

import numpy as np
import numpy.linalg as npl
import numpy.random as npr


def hard_thrshld(arr_x, num_k):
    """
        Hard Thresholding

    Parameters
    ----------

    arr_x : ndarray
        vector to threshold
    num_k : int
        thresholding parameter

    Returns
    -------
    ndarray
        thresholded vector
    """

    sort = np.argsort(np.abs(arr_x))
    arr_x[sort[:-num_k]] = 0
    return arr_x


def soft_thrshld(arr_x, num_alpha):
    """
        Soft Thresholding

    Parameters
    ----------

    arr_x : ndarray
        vector to threshold
    num_k : int
        thresholding parameter

    Returns
    -------
    ndarray
        thresholded vector
    """

    arr_sig = np.sign(arr_x)
    arr_y = np.abs(arr_x) - num_alpha
    arr_y[arr_y < 0] = 0
    return arr_y*arr_sig


def H(X):
    """
        Return the Hermitian transpose of X

    """
    return np.conj(X).T


def aB(a):
    """
        Convert a Zero-One-Vector t a Boolean Vector

    """
    return(np.array(a, dtype=bool))


def proj_sphere(X):
    """
        Project columns of X onto the unit sphere

    This method takes a two-dimensional (real or complex) array X
    and projects the columns onto the unit sphere.

    It is very useful, if multiple vectors of the same size have to be
    normalized.
    """

    # allocate memory for the result
    mat_Z = np.empty((X.shape[0], X.shape[1]), X.dtype)

    # iterate through the columns
    for ii in range(0, X.shape[1]):
        mat_Z[:, ii] = X[:, ii]/np.sqrt(np.conj(X[:, ii]).dot(X[:, ii]))

    return mat_Z


def khatri_rao(X, Y):
    """
    Calculate columnwise Kronecker Product

    This is also known as the Khatri-Rao product.
    """
    # check if both matrices have same smount of columns
    if X.shape[1] != Y.shape[1]:
        raise TypeError('Matrices can not be multiplied')

    # the number of rows of the output
    N = X.shape[0]*Y.shape[0]

    # infer data-type form the factors
    Z = np.zeros(
        (N, X.shape[1]),
        dtype=np.promote_types(X.dtype, Y.dtype)
    )

    # now push the reshaped outer products into the columns of the result
    for ii in range(0, X.shape[1]):
        Z[:, ii] = np.squeeze(
            np.asarray((np.outer(X[:, ii], Y[:, ii]).reshape(N, 1)))
        )

    return(Z)


def sampleRIP(mat_X):
    """
        Lower RIP-Constant bound

    For a given matrix, we randomly sample vectors on the unit sphere
    and try to approximate the RIP-constant based on the distortion
    X applies to the sampled vector.
    """

    # extract the dimensions
    num_m = mat_X.shape[0]
    N = mat_X.shape[1]
    R = range(0, N)
    delta = np.zeros(num_m-1)

    # dunno what we are doing here
    for ii in range(1, num_m):
        print(ii)
        d = 0
        if ii > 1:
            d = delta[ii-2]

        for jj in range(0, (N-ii)**2):
            S = npr.choice(R, size=ii, replace=0)
            mat_X_S = mat_X[:, S]
            num_X_S_norm = np.max(
                npl.svd(H(mat_X_S).dot(mat_X_S) - np.eye(ii), compute_uv=0))
            d = max(d, num_X_S_norm)
            delta[ii-1] = d
    return delta


def coh(X):
    """Self-Coherence of a matrix X"""
    Y = proj_sphere(X)
    G = np.abs(np.dot(H(Y), Y))
    G = G - np.eye(G.shape[1])
    return np.max(G)


def mut_coh(X, Y):
    """Mutual Coherence of Two Matrices"""
    X1 = proj_sphere(X)
    Y1 = proj_sphere(Y)
    G = np.abs(np.dot(H(X1), Y1))
    return np.max(G)


def normGram(
    X
):
    """
        Calculate the Normalized Gram Matrix
    """
    arr_n = np.zeros(X.shape[1])
    for ii in range(0, X.shape[1]):
        arr_n[ii] = 1.0/np.sqrt(np.sum(X[:, ii]**2))

    return(((arr_n*X).T).dot(arr_n*X))


def same_supp(
    x1,
    x2
):
    """
        Check if two vectors have the same support
    """

    d = (1*(x1 != 0)) - (1*(x2 != 0))
    return bool(1 - ((np.sum(np.abs(d))) > 0))


def has_supp_size(
    x,
    s
):
    """
        checks if a vector has a certain support size
    """

    return 1*(np.sum(x != 0) == s)


def arrayToString(x):
    """
        Convert a numpy array to a string
    """
    x_txt = list(map(lambda tt: str(tt), (x.tolist())))
    x_txt = ",".join(x_txt)
    return x_txt


def T1_dist_l2(phi, phihat):
    phi1 = phi.squeeze().reshape((-1, +1))
    phi2 = phihat.squeeze().reshape((+1, -1))

    D1 = np.mod(phi1 - phi2, 2 * np.pi)
    D2 = np.mod(phi2 - phi1, 2 * np.pi)

    D = np.minimum(D1, D2)

    err = np.empty(D.shape[0])

    for ii in range(D.shape[0]):
        minInd = np.unravel_index(np.argmin(D), D.shape)
        err[ii] = D[minInd[0], minInd[1]]
        D = np.delete(np.delete(D, minInd[1], 1), minInd[0], 0)

    return npl.norm(err) ** 2
