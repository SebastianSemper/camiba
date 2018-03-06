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
Provides the Orthogonal Matching Pursuit Algorithm to approximate a solution
to the l_0-minimization problem, which reads as

min ||x||_0 s.t. Ax = b

for given matrix A and vector b and it is described and analyzed in [OMP]_.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from ..linalg.basic import aB
from ..linalg.basic import H


def recover(
    mat_A,
    arr_b,
    num_steps,
):
    """
        Orthogonal Matching Pursuit Algorithm

    Parameters
    ----------

    mat_A : ndarray
        system matrix
    arr_b : ndarray
        measurement vector
    num_steps : int
        iteration steps

    Returns
    -------
    ndarray
        estimated sparse solution
    """

    dt_type = np.promote_types(arr_b.dtype, mat_A.dtype)

    # generate array for reweighting the matrix A
    arr_d = np.empty(mat_A.shape[1], dtype=dt_type)
    for ii in range(0, mat_A.shape[1]):
        arr_d[ii] = np.sqrt(np.conj(mat_A[:, ii]).dot(mat_A[:, ii]))

    num_m = mat_A.shape[1]
    x = np.zeros(num_m, dtype=dt_type)
    S = np.zeros(num_m, dtype=dt_type)

    for ii in range(0, num_steps):

        # calc correlation
        r = np.abs(mat_A.conj().T.dot((arr_b - mat_A.dot(x))))/arr_d

        # add maximal correlation to supprt
        S[np.argmax(r*(1-S))] += 1

        # generate matrices with columns restricted to support
        mat_A_S = mat_A[:, aB(S)]
        mat_B_S = H(mat_A_S)

        # solve approximation problem with
        # pseudo-inverse on current support
        try:
            x_S = npl.solve(mat_B_S.dot(mat_A_S), mat_B_S.dot(arr_b))
        except npl.linalg.LinAlgError:
            return x

        x[aB(S)] = x_S

    return x
