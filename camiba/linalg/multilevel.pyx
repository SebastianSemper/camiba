#cython: language_level=3, boundscheck=False, wraparound=False

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
cimport numpy as np


cpdef np.ndarray Toep(
    np.ndarray ten_U,
    bint verbose=False
):

    cdef np.ndarray arr_s = np.array([*(ten_U[:].shape)]).astype('int')
    cdef np.ndarray arr_d = ((arr_s + 1) / 2).astype('int')

    return _Toep_rec(
        ten_U,
        arr_s,
        arr_d,
        verbose
    )


cdef np.ndarray _Toep_rec(
    np.ndarray ten_U,
    np.ndarray arr_s,
    np.ndarray arr_d,
    bint verbose
):
    cdef int arr_s0 = arr_s[0]
    cdef int arr_d0 = arr_d[0]

    cdef int num_n = np.prod(arr_d)
    cdef int num_blk_s = np.prod(arr_d[1:])
    cdef int ss, dd, ss_

    cdef np.ndarray mat_T
    cdef np.ndarray mat_blk

    # check if we are already in the deepest level
    if num_n != arr_d0:
        mat_T = np.zeros((num_n, num_n), dtype=ten_U.dtype)
        # iterate over the number of blocks
        for ss in range(arr_s0):
            mat_blk = _Toep_rec(ten_U[ss], arr_s[1:], arr_d[1:], verbose)

            # check if we are still walking up the rows
            if ss < arr_d0:
                for dd in range(ss + 1):
                    mat_T[
                        num_n - (ss + 1) * num_blk_s + dd * num_blk_s:
                        num_n - ss * num_blk_s + dd * num_blk_s,
                        dd * num_blk_s:(dd+1) * num_blk_s
                    ] = mat_blk[:]

            # or if we are already walking along the columns
            else:
                ss_ = ss - arr_d0 + 1
                for dd in range(0, arr_s0 - ss):
                    mat_T[
                        dd * num_blk_s: (dd + 1) * num_blk_s,
                        ss_ * num_blk_s + dd * num_blk_s:
                        (ss_ + 1) * num_blk_s + dd * num_blk_s
                    ] = mat_blk[:]

        return mat_T
    else:
        return spToep(
            ten_U[:arr_d[0]][::-1],
            ten_U[arr_d[0] - 1:]
        )


cpdef np.ndarray hermToep(
    np.ndarray ten_U,
    bint verbose=False
):
    cdef np.ndarray arr_s
    cdef np.ndarray arr_d

    cdef int num_blk_s
    cdef int num_n

    cdef np.ndarray mat_hT

    cdef int ii, dd
    arr_s = np.array([*(ten_U[0].shape)]).astype('int')
    arr_d = ((arr_s + 1) / 2).astype('int')

    num_blk_s = np.prod(arr_d)
    num_n = ten_U[:].shape[0] * num_blk_s
    mat_hT = np.atleast_2d(np.zeros((num_n, num_n), dtype=ten_U.dtype))

    if len(ten_U[:].shape) > 1:
        for ii in range(ten_U[:].shape[0]):
            for dd in range(ii + 1):
                mat_hT[
                    num_n - (ii + 1) * num_blk_s + dd * num_blk_s:
                    num_n - ii * num_blk_s + dd * num_blk_s,
                    dd * num_blk_s:(dd + 1) * num_blk_s
                ] = Toep(ten_U[ii])
    else:
        mat_hT = spToep(ten_U)
        mat_hT -= 1j * np.diag(np.imag(np.diag(mat_hT)))
        return mat_hT

    cdef void* base_ptr
    cdef void* row_ptr
    cdef void* end_ptr

    base_ptr = mat_hT.data

    if mat_hT.dtype != np.complex128:
        raise TypeError("Wrong datatype of mat_hT")

    for ii in range(0, mat_hT.shape[0]):

        row_ptr = base_ptr + ii * mat_hT.strides[0] + ii * mat_hT.strides[1]
        (<np.complex128_t*>row_ptr)[0].real *= 0.5

        row_ptr += mat_hT.strides[1]

        end_ptr = row_ptr + (mat_hT.shape[1] - ii - 1) * mat_hT.strides[1]
        while row_ptr < end_ptr:
            (<np.complex128_t*>row_ptr)[0] = 0
            row_ptr += mat_hT.strides[1]

    (<np.complex128_t*>end_ptr)[0].real *= 0.5

    return mat_hT + mat_hT.T.conj()


cpdef np.ndarray ToepAdj(
    np.ndarray mat_A,
    np.ndarray ten_u
):
    arr_s = np.array([*ten_u[:].shape]).astype('int')
    arr_d = ((arr_s + 1) / 2).astype('int')
    num_n = np.prod(arr_d)

    return ToepAdj_rec(mat_A, arr_d, arr_s)

cdef np.ndarray ToepAdj_rec(
    np.ndarray mat_A,
    np.ndarray arr_d,
    np.ndarray arr_s
):
    cdef int num_n = np.prod(arr_d)
    cdef int num_blk_s
    cdef int ss, dd

    cdef np.ndarray ten_S

    cdef int arr_s0 = arr_s[0]
    cdef int arr_d0 = arr_d[0]

    if num_n != arr_d0:
        ten_S = np.zeros((*arr_s, ), dtype=mat_A.dtype)
        # iterate over the number of blocks
        for ss in range(arr_s0):
            num_blk_s = np.prod(arr_d[1:])
            # check if we are still walking up the rows
            if ss < arr_d0:
                for dd in range(ss + 1):
                    # print(ss, dd, num_n, num_blk_s)
                    # print("rb",num_n - (ss + 1) * num_blk_s + dd * num_blk_s)
                    # print("re",num_n - ss * num_blk_s + dd * num_blk_s)
                    # print("cb",dd * num_blk_s)
                    # print("ce",(dd+1) * num_blk_s)
                    # print(np.sum(mat_A[
                    #     num_n - (ss + 1) * num_blk_s + dd * num_blk_s:
                    #     num_n - ss * num_blk_s + dd * num_blk_s,
                    #     dd * num_blk_s:(dd+1) * num_blk_s
                    # ]))
                    # print("###########")
                    ten_S[ss] += ToepAdj_rec(
                        mat_A[
                            num_n - (ss + 1) * num_blk_s + dd * num_blk_s:
                            num_n - ss * num_blk_s + dd * num_blk_s,
                            dd * num_blk_s:(dd+1) * num_blk_s
                        ],
                        arr_d[1:],
                        arr_s[1:]
                    )
                # print("-----------------")
            # or if we are already walking along the columns
            else:
                ss_ = ss - arr_d0
                for dd in range(1, arr_d0 - ss_):
                    # print(ss, ss_, dd, num_n, num_blk_s)
                    # print("rb",(dd - 1) * num_blk_s)
                    # print("re",dd * num_blk_s)
                    # print("cb",ss_ * num_blk_s + dd * num_blk_s)
                    # print("ce",(ss_ + 1) * num_blk_s + dd * num_blk_s)
                    # print(mat_A[
                    #     dd * num_blk_s: (dd + 1) * num_blk_s,
                    #     ss_ * num_blk_s + dd * num_blk_s:
                    #     (ss_ + 1) * num_blk_s + dd * num_blk_s
                    # ])
                    # print(ten_S[:].shape)
                    # print("###########")
                    ten_S[ss] += ToepAdj_rec(
                        mat_A[
                            (dd - 1) * num_blk_s: dd * num_blk_s,
                            ss_ * num_blk_s + dd * num_blk_s:
                            (ss_ + 1) * num_blk_s + dd * num_blk_s
                        ],
                        arr_d[1:],
                        arr_s[1:]
                    )
    else:
        num_blk_s = mat_A[:].shape[1]
        ten_S = np.zeros(arr_s[0], dtype=mat_A.dtype)
        for ss in range(arr_s0):
            # check if we are walking up the rows
            if ss < arr_d0:
                # print("++++++++++")
                # print(mat_A[num_blk_s - ss - 1:, :ss+1])
                # print("++++++++++")
                ten_S[ss] = np.sum(np.diag(
                    mat_A[num_blk_s - ss - 1:, :ss+1]
                ))
            # or if we are already walking along the columns
            else:
                ss_ = ss - arr_d0 + 1
                # print("++++++++++")
                # print(mat_A[:num_blk_s - ss_, ss_:])
                # print("++++++++++")
                ten_S[ss] = np.sum(np.diag(
                    mat_A[:num_blk_s - ss_, ss_:]
                ))
    return ten_S

cpdef np.ndarray hermToepAdj(
    np.ndarray mat_A,
    np.ndarray ten_u
):
    cdef np.ndarray arr_s = np.array([*ten_u[0].shape]).astype('int')
    cdef np.ndarray arr_d = ((arr_s + 1) / 2).astype('int')
    cdef int num_n = mat_A[:].shape[0]

    cdef np.ndarray ten_s = np.zeros_like(ten_u[:], dtype=mat_A.dtype)

    cdef int ss, dd, num_blk_s

    if len(ten_u[:].shape) > 1:
        num_blk_s = np.prod(arr_d)
        for ss in range(ten_u[:].shape[0]):
            for dd in range(ss + 1):

                ten_s[ss] += ToepAdj(
                    mat_A[
                        num_n - (ss + 1) * num_blk_s + dd * num_blk_s:
                        num_n - ss * num_blk_s + dd * num_blk_s,
                        dd * num_blk_s:(dd + 1) * num_blk_s
                    ],
                    ten_u[ss]
                )
    else:
        num_blk_s = ten_u[:].shape[0]
        for ss in range(ten_u[:].shape[0]):
            ten_s[ss] = np.sum(np.diag(
                mat_A[ss:, :num_blk_s - ss]
            ))

    return ten_s


cpdef np.ndarray ToepInv(
    np.ndarray mat_T,
    np.ndarray arr_d
):
    cdef np.ndarray arr_s = 2 * arr_d - 1
    cdef np.ndarray ten_u = np.zeros((*arr_s,), dtype=mat_T.dtype)
    ToepInv_rec(mat_T, ten_u, arr_d, arr_s)

    return ten_u[:]


cpdef np.ndarray ToepInv_rec(
    np.ndarray mat_T,
    np.ndarray ten_u,
    np.ndarray arr_d,
    np.ndarray arr_s
):
    cdef int num_n = np.prod(arr_d)

    cdef int ss, dd, ss_

    cdef int arr_s0 = arr_s[0]
    cdef int arr_d0 = arr_d[0]

    if len(ten_u[:].shape) > 1:
        for ss in range(arr_s0):
            num_blk_s = np.prod(arr_d[1:])
            # print("arr_d", arr_d[1:])
            # print("num_blk", num_blk_s)
            # print("num_n - (ss + 1) * num_blk_s", num_n - (ss + 1) * num_blk_s)
            # print("num_n - ss * num_blk_s", num_n - ss * num_blk_s)
            # print("num_blk_s", num_blk_s)
            # print("shape", mat_T[:].shape)
            # check if we are still walking up the rows
            if ss < arr_d0:
                ten_u[ss] = ToepInv_rec(
                    mat_T[
                        num_n - (ss + 1) * num_blk_s:
                        num_n - ss * num_blk_s,
                        :num_blk_s
                    ],
                    ten_u[ss],
                    arr_d[1:],
                    arr_s[1:]
                )
            else:
                ss_ = ss - arr_d0 + 1
                ten_u[ss] = ToepInv_rec(
                    mat_T[
                        :num_blk_s,
                        ss_ * num_blk_s:
                        (ss_ + 1) * num_blk_s
                    ],
                    ten_u[ss],
                    arr_d[1:],
                    arr_s[1:]
                )
    else:
        ten_u[:] = np.block([
            mat_T[:,0],
            mat_T[0,1:]
        ])[:]


cpdef np.ndarray HermToepInv(
    np.ndarray mat_T,
    np.ndarray arr_d
):
    cdef np.ndarray arr_s = 2 * arr_d - 1
    arr_s[0] = arr_d[0]
    cdef int num_n = mat_T[:].shape[0]

    cdef np.ndarray ten_u = np.zeros((*arr_s, ), dtype=mat_T.dtype)

    cdef int ss, dd, num_blk_s

    num_blk_s = np.prod(arr_d[1:])
    if len(ten_u[:].shape) > 1:
        for ss in range(arr_d[0]):
            # print("num_n - (ss + 1) * num_blk_s:", num_n - (ss + 1) * num_blk_s)
            # print("num_n - ss * num_blk_s,", num_n - ss * num_blk_s)
            # print(":num_blk_s", num_blk_s)
            # print("matT:", mat_T[
            #     num_n - (ss + 1) * num_blk_s:
            #     num_n - ss * num_blk_s,
            #     :num_blk_s
            # ].shape)
            # print(mat_T[
            #     num_n - (ss + 1) * num_blk_s:
            #     num_n - ss * num_blk_s,
            #     :num_blk_s
            # ].shape)
            # print(ten_u[ss].shape)
            ten_u[ss] = ToepInv(
                mat_T[
                    num_n - (ss + 1) * num_blk_s:
                    num_n - ss * num_blk_s,
                    :num_blk_s
                ],
                arr_d[1:]
            )
    else:
        # print(mat_T[:,0][:].shape, ten_u[:].shape)
        ten_u[:] = mat_T[:,0][:]


    return ten_u
