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
Provides the Iterative Hard Thresholding Algorithm to approximate a solution
to the l_0-minimization problem, which reads as

min ||x||_0 s.t. Ax = b

for given matrix A and vector b and it is described and analyzed in [IHT]_.
"""

import numpy as np
import numpy.linalg as npl

from ..linalg.basic import hard_thrshld


def recover(
    mat_A,
    arr_b,
    num_maxsteps,
    num_k
):
    """
        IHT Algorithm

    Parameters
    ----------

    mat_A : ndarray
        system matrix
    arr_b : ndarray
        measurement vector
    num_steps : int
        iteration steps
    num_k : int
        thresholding parameter

    Returns
    -------
    ndarray
        estimated sparse solution
    """

    # initialize a first solution
    arr_x = np.zeros(mat_A.shape[1])

    # calculate step size
    num_max_ev = npl.svd((mat_A.conj().T).dot(mat_A), compute_uv=0)[0] * 2
    num_t = 1.0/num_max_ev

    for ii in range(0, num_steps):

        arr_residual = mat_A.dot(arr_x) - arr_b

        arr_x = hard_thrshld(
            arr_x - 2*num_t * (mat_A.conj().T).dot(arr_residual),
            num_k
        )
    return arr_x
