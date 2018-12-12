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

import numpy as np


"""
this module provides some variants of algorithms to solve the LASSO problem.
These implementations merely serve the purpose of prototypes in order to
analyze the algorithms
"""


def LARS(
    mat_X,
    vec_y,
    num_steps
):

    vec_c_hat = mat_X.T.conj().dot(vec_y)

    # eq. (2.9)
    num_C_hat = np.max(np.abs(vec_c_hat))

    # eq. (2.9)
    set_A = np.isclose(np.abs(vec_c_hat), num_C_hat)
    set_A_bar = np.logical_not(set_A)

    vec_beta_hat = np.zeros(mat_X.shape[1])
    vec_beta_hat[set_A] = np.linalg.lstsq(mat_X[:, set_A], vec_y)[0]
    vec_mu_hat = mat_X.dot(vec_beta_hat)

    print(np.linalg.norm(vec_y))
    print(np.sum(set_A))
    print(np.linalg.norm(vec_y - vec_mu_hat))
    print(np.linalg.norm(vec_y - mat_X.dot(vec_beta_hat)))

    for kk in range(num_steps):
        vec_c_hat = mat_X.T.conj().dot(vec_y - vec_mu_hat)

        # eq. (2.9)
        num_C_hat = np.max(np.abs(vec_c_hat))

        # eq. (2.10)
        vec_s = np.sign(vec_c_hat[set_A])

        # eq. (2.4)
        mat_X_A = (mat_X[:, set_A]).dot(np.diag(vec_s))

        # eq. (2.5)
        mat_G_A = mat_X_A.conj().T.dot(mat_X_A)

        # helper vector
        vec_tmp = np.linalg.solve(
            mat_G_A, np.ones(mat_G_A.shape[0])
        )

        # eq. (2.5)
        num_A_A = np.sum(vec_tmp) ** -.5

        # eq. (2.6)
        vec_w_A = num_A_A * vec_tmp
        vec_d_hat = np.zeros(mat_X.shape[1])
        vec_d_hat[set_A] = vec_w_A * vec_s

        # eq. (2.6)
        vec_u_A = mat_X_A.dot(vec_w_A)

        # eq. (2.11)
        vec_a = mat_X.conj().T.dot(vec_u_A)

        # eq. (2.13)
        vec_quot_1 = np.zeros_like(vec_c_hat)
        vec_quot_1[set_A_bar] = (
            (num_C_hat - vec_c_hat)[set_A_bar] / (num_A_A - vec_a)[set_A_bar]
        )
        vec_quot_2 = np.zeros_like(vec_c_hat)
        vec_quot_2[set_A_bar] = (
            (num_C_hat + vec_c_hat)[set_A_bar] / (num_A_A + vec_a)[set_A_bar]
        )

        # eq (3.4)
        vec_max_gamma = np.zeros(mat_X.shape[1])
        vec_max_gamma[set_A] = - vec_beta_hat[set_A] / vec_d_hat[set_A]

        # eq (3.5)
        if np.any(vec_max_gamma > 0):
            num_gamma_tilde = np.min(vec_max_gamma[vec_max_gamma > 0])
            num_gamma_tilde_j = np.min(np.arange(mat_X.shape[1])[
                np.isclose(vec_max_gamma, num_gamma_tilde)
            ])
        else:
            num_gamma_tilde = np.inf


        set_min_1 = np.logical_and(
            vec_quot_1 > 0,
            set_A_bar
        )

        set_min_2 = np.logical_and(
            vec_quot_2 > 0,
            set_A_bar
        )

        if np.any(set_min_1):
            num_argmin_1 = np.min(vec_quot_1[set_min_1])
        else:
            num_argmin_1 = np.inf

        if np.any(set_min_2):
            num_argmin_2 = np.min(vec_quot_2[set_min_2])
        else:
            num_argmin_2 = np.inf


        if num_argmin_1 < num_argmin_2:
            num_j_hat = np.min(np.arange(mat_X.shape[1])[
                np.isclose(vec_quot_1, num_argmin_1)
            ])
            num_gamma_hat = num_argmin_1
        else:
            num_j_hat = np.min(np.arange(mat_X.shape[1])[
                np.isclose(vec_quot_2, num_argmin_2)
            ])
            num_gamma_hat = num_argmin_2

        if num_gamma_tilde < num_gamma_hat:
            num_gamma_hat = num_gamma_tilde
            set_A[num_gamma_tilde_j] = False
        else:
            set_A[num_j_hat] = True

        vec_beta_hat += num_gamma_hat * vec_d_hat
        vec_mu_hat += num_gamma_hat * vec_u_A
        set_A_bar = np.logical_not(set_A)
        print(np.sum(set_A))
        print(np.linalg.norm(vec_y - vec_mu_hat))
        print(np.linalg.norm(vec_y - mat_X.dot(vec_beta_hat)))
