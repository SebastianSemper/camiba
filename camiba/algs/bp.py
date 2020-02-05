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
Provides the real and direct implementation of basis pursuit

min ||x||_1 s.t. Ax = b

for given matrix A and vector b.
"""

import numpy as np
from scipy.optimize import linprog
import numpy.linalg as npl


def recover(
    mat_A,
    arr_b
):
    """
        Basis Pursuit Algorithm

    Parameters
    ----------

    mat_A : ndarray
        system matrix
    arr_b : ndarray
        measurement vector

    Returns
    -------
    ndarray
        estimated ell1 solution
    """

    c = np.ones(2 * mat_A.shape[1])
    A_eq = np.block([mat_A, -mat_A])
    b_eq = np.copy(arr_b)

    try:
        sol = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            method='interior-point',
            options={
                'lstsq': True,
                'sym_pos': False
            }
        )

        if sol.success:
            x = sol.x[:mat_A.shape[1]] - sol.x[mat_A.shape[1]:]
            return x
        else:
            raise RuntimeError("BP did not converge!")
    except:
        raise RuntimeError("BP did not converge!")
