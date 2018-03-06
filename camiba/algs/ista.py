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
Implements the algorithm of iterative soft thresholding
for approximating a solution to a l_1-minimization problem, which reads as

min ||x||_a + lambda * ||Ax - b||_2^2

for given matrix A and vector b and it is described and analyzed in [ISTA]_.
"""

from ..linalg.basic import soft_thrshld
import numpy as np
import numpy.linalg as npl


def recover(mat_A, arr_b, x_init, num_lambda, num_steps):
    """
        Iterative Soft Thresholding Algorithm

    Parameters
    ----------

    mat_A : ndarray
        system matrix
    arr_b : ndarray
        measurement vector
    x_init : ndarray
        initial solution guess
    num_lambda : int
        thresholding parameter
    num_steps : int
        iteration steps

    Returns
    -------
    ndarray
        estimated sparse solution
    """

    # initialize a first solution
    arr_x = np.copy(x_init)

    # estimate optimal step size
    num_max_ev = npl.svd((mat_A.conj().T).dot(mat_A), compute_uv=0)[0] * 2

    # do the iteration
    num_t = 1.0/num_max_ev

    for ii in range(0, num_steps):

        arr_residual = mat_A.dot(arr_x) - arr_b
        residual = npl.norm(arr_residual)

        arr_x = soft_thrshld(
            arr_x - 2*num_t * (mat_A.conj().T).dot(arr_residual),
            num_t * num_lambda)
    return arr_x
