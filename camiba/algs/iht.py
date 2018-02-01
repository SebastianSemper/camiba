# -*- coding: utf-8 -*-
"""provides the iterative hard thresholding algorithm to approximate a solution
    to the l_0-minimization problem."""

import numpy as np
import numpy.linalg as npl

from ..linalg.basic import hard_thrshld


def recover(
    mat_A,
    arr_b,
    num_maxsteps,
    num_k
):
    """IHT Algorithm for matrix mat_A and vector arr_b"""

    # initialize a first solution
    arr_x = np.zeros(mat_A.shape[1])

    num_max_ev = npl.svd((mat_A.conj().T).dot(mat_A), compute_uv=0)[0] * 2
    num_t = 1.0/num_max_ev

    for ii in range(0, num_maxsteps):

        arr_residual = mat_A.dot(arr_x) - arr_b

        arr_x = hard_thrshld(arr_x - 2*num_t \
            * (mat_A.conj().T).dot(arr_residual), num_k)
    return arr_x
