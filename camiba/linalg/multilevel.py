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
import numpy as np
from scipy.linalg import toeplitz as spToep


def Toep(tenU, verbose=False):
    """
        Construct a Hermitian Multilevel Toeplitz Matrix

    Parameters
    ----------

    tenU : ndarray
        The tensor, which describes the multilevel Toeplitz matrix
    verbose :
        Verbosity Flag

    Returns
    -------
    ndarray
        The constructed matrix as 2D ndarray
    """

    arrD = np.array([*tenU.shape])

    T = _toepRecursion(arrD, tenU, verbose)

    return T - 1j*np.eye(np.prod(arrD))*np.imag(T[0, 0])


def _toepRecursion(arrN, tenU, verbose=False):
    """
        Recursion during Multilevel Toeplitz Construction

    Parameters
    ----------

    arrN : ndarray
        The sizes of the levels in the current iteration
    tenU : ndarray
        The tensor, in the current iteration
    verbose :
        Verbosity Flag

    Returns
    -------
    ndarray
        The constructed (sub)matrix as 2D ndarray
    """
    # number of dimensions in current level
    numD = arrN.shape[0]

    # get size of resulting block toeplitz matrix
    num_N = np.prod(arrN)

    # get an array of all partial sequential products
    # starting at the front
    arrNprod = np.prod(arrN[1:])

    # allocate memory for the result
    T = np.zeros((num_N, num_N), dtype='complex')

    # check if we can go a least a level deeper
    if numD > 1:

        # iterate over size of the first dimension
        for nn in range(arrN[0]):

            # number of instances a single block occurs.
            # the further down / right it starts,
            # the lower its count
            countBlocks = arrN[0] - nn

            # now calculate the block recursively
            subT = _toepRecursion(arrN[1:], tenU[nn])

            if nn == 0:
                # in the case nn = 0, we are on the diagonal an we can
                # just put it there no questions asked
                for mm in range(countBlocks):
                    T[
                        mm * arrNprod: (mm + 1) * arrNprod,
                        mm * arrNprod: (mm + 1) * arrNprod
                    ] = subT
            else:
                # we are above and below the diagonal we put a conjugated
                # block below and to enforce the Hermitian structure, we put
                # the unconjugated version above
                for mm in range(countBlocks):
                    # this are the blocks above the diagonal, since we start
                    # in the first row, when mm = 0
                    T[
                        mm * arrNprod: (mm + 1) * arrNprod,
                        (mm + nn) * arrNprod:
                        (mm + 1 + nn) * arrNprod,
                    ] = subT

                    # this are the blocks below the diagonal, since we start
                    # in the first column for mm = 0, so we have to conjugate
                    # everything
                    T[
                        (mm + nn) * arrNprod: (mm + 1 + nn) * arrNprod,
                        mm * arrNprod: (mm + 1) * arrNprod
                    ] = subT.conj()

        return T
    else:
        # if we are in a lowest level, we just construct the right
        # single level toeplitz block
        return spToep(tenU[:num_N].conj())


def ToepAdj(arrD, mat_A, verbose=False):
    """
        Calculate the Adjoint Operator of <T,A>

    Parameters
    ----------

    arrD : ndarray
        The sizes of the levels in the current iteration
    tenU : ndarray
        The tensor, in the current iteration
    verbose :
        Verbosity Flag

    Returns
    -------
    ndarray
        The constructed (sub)matrix as 2D ndarray
    """
    tenR = np.zeros((*arrD,), dtype=mat_A.dtype)

    _toepAdjRec(arrD, mat_A, tenR, verbose)

    return tenR


def _toepAdjRec(arrN, mat_A, tenR, verbose=False):
    """
        Recursion for the Adjoint of <T,A>
    """
    # number of dimensions in current level
    numD = arrN.shape[0]

    # get an array of all partial sequential products
    # starting at the front
    arrNprod = np.prod(arrN[1:])

    if numD > 1:
        for nn in range(arrN[0]):
            countBlocks = arrN[0] - nn

            for mm in range(countBlocks):
                _toepAdjRec(
                    arrN[1:],
                    mat_A[
                        (mm + nn) * arrNprod: (mm + 1 + nn) * arrNprod,
                        mm * arrNprod: (mm + 1) * arrNprod
                    ],
                    tenR[nn],
                    verbose
                )
    else:
        for ii in range(arrN[0]):
            tenR[ii] += np.trace(mat_A[:(arrN[0]-ii), ii:])
