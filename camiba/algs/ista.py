# -*- coding: utf-8 -*-
"""provides the iterative shrinkage thresholding algorithm to approximate a solution
    to the l_0-minimization problem."""


from ..linalg.basic import soft_thrshld
import numpy as np
import numpy.linalg as npl

def recover(mat_A, arr_b, x_init, num_lambda, num_maxsteps):
    """ISTA Algorithm for matrix mat_A and vector arr_b"""

    # initialize a first solution
    arr_x = np.copy(x_init)

    num_max_ev = npl.svd((mat_A.conj().T).dot(mat_A), compute_uv=0)[0] * 2

    # do the iteration
    num_t = 1.0/num_max_ev

    for ii in range(0, num_maxsteps):

        arr_residual = mat_A.dot(arr_x) - arr_b
        residual = npl.norm(arr_residual)

        residual_last = residual

        arr_x = soft_thrshld(arr_x - 2*num_t *
                        (mat_A.conj().T).dot(arr_residual), num_t*num_lambda)
    return arr_x
