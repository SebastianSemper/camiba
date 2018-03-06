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

"""


import numpy as np
import numpy.linalg as npl


def one_d(mat_cov, num_s):
    """
        One-dimensional ESPRIT

    Parameters
    ----------

    mat_cov : ndarray
        covariance matrix of the signal
    num_s : int
        number of frequencies in the signal

    Returns
    -------
    ndarray
        extracted frequencies
    """
    U, S, V = npl.svd(mat_cov)
    Uhat = npl.lstsq(U[: -1, :num_s], U[1:, :num_s])

    phi = np.mod(np.angle(npl.eigvals(Uhat[0])), 2 * np.pi)

    return phi


def two_d(mat_cov, num_s):
    """
        Two-dimensional ESPRIT

    Parameters
    ----------

    mat_cov : ndarray
        covariance matrix of the signal
    num_s : int
        number of frequencies in the signal

    Returns
    -------
    ndarray
        extracted frequencies
    """

    return 0
