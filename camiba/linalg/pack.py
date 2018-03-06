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
implements an iterative algorithm for finding unstructured
measurement matrices. works for complex and real valued matrices.
Here we are minimizing a certain potential function induced by the
points on the sphere, thus they repell each other
"""

import math
import time
import os.path
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
from .basic import *


def sq_sum_c(x):
    """speeds up norm calculation -- complex case"""
    return np.sum(np.conj(x)*x)


def sq_sum(x):
    """speeds up norm calculation -- real case"""
    return np.sum(x*x)


def drawTwo(
        num_m,
        num_N,
        num_eps,
        num_h,
        num_max_steps=0
):
    """

    """

    mat_X1 = npr.randn(2, num_m, num_N)
    mat_X2 = np.zeros((2, num_m, num_N))

    mat_F = npr.randn(num_m, num_N)

    tr = list(map(
        lambda ii: npl.norm(mat_X1[ii, :, :] - mat_X2[ii, :, :]), range(2)
    ))

    num_steps = 0

    sw = 0

    print('$MINPOT$ Starting minimizing a potential..')

    print('$MINPOT$ Before optimization')
    print('$MINPOT$ Mutual coherence : %f - coherence of first factor: %f - coherence of second factor: %f' %
          (csg.mut_coh(mat_X1[1, :, :], mat_X1[0, :, :]), csg.coh(mat_X1[1, :, :]), csg.coh(mat_X1[0, :, :])))
    print('$MINPOT$ coherence of KRP: %f' %
          (csg.coh(krp.prod(mat_X1[0, :, :], mat_X1[1, :, :]))))
    while (sum(tr) > num_eps) and (num_steps < num_max_steps):

        mat_F.fill(0)

        sw = 1 - sw

        for ii in range(num_N):
            d1 = mat_X1[sw, :, ii] - mat_X1[sw, :, :].T
            d2 = mat_X1[sw, :, ii] + mat_X1[sw, :, :].T
            d3 = mat_X1[sw, :, ii] - mat_X1[1 - sw, :, :].T
            d4 = mat_X1[sw, :, ii] + mat_X1[1 - sw, :, :].T

            dd1 = np.sum(d1**2, 1)**2
            dd2 = np.sum(d2**2, 1)**2
            dd3 = np.sum(d3**2, 1)**2
            dd4 = np.sum(d4**2, 1)**2

            # set itself to non-zero
            dd1[ii] = 1
            dd2[ii] = 1

            dd1 = 1.0/dd1
            dd2 = 1.0/dd2
            dd3 = 1.0/dd3
            dd4 = 1.0/dd4

            # weight the distances with their norms
            d1 = d1.T*dd1
            d2 = d2.T*dd2
            d3 = d3.T*dd3
            d4 = d4.T*dd4

            # set the wrong one to 0
            d1[:, ii] = 0
            d2[:, ii] = 0

            # sum everything together and add it to force
            mat_F[:, ii] -= np.sum(d1+d2+d3+d4, 1)

        # calc next iterations
        mat_X2[sw, :, :] = csg.projSphere(mat_X1[sw, :, :] - num_h * mat_F)

        # calc difference
        tr = list(map(
            lambda ii: npl.norm(mat_X1[ii, :, :] - mat_X2[ii, :, :]), range(2)
        ))

        mat_X1[sw, :, :] = np.copy(mat_X2[sw, :, :])
        num_steps += 1

    print('$MINPOT$ After optimization')
    print('$MINPOT$ Mutual coherence : %f - coherence of first factor: %f - coherence of second factor: %f' %
          (csg.mut_coh(mat_X1[1, :, :], mat_X1[0, :, :]), csg.coh(mat_X1[1, :, :]), csg.coh(mat_X1[0, :, :])))
    print('$MINPOT$ coherence of KRP: %f' %
          (csg.coh(krp.prod(mat_X1[0, :, :], mat_X1[1, :, :]))))
    return mat_X1


def drawSeed(mat_X, eps, h, steps=0):
    """applies the potential minimizing algorithm
    for a given initial configuration mat_X a given amount of steps.
    another stopping criterion has to be supplied together with a
    step length"""

    # extract dimensions
    num_n, num_m = mat_X.shape

    # init first iteration from parameter
    X0 = mat_X

    # init next iteration
    X1 = np.zeros((num_n, num_m))

    # init forces
    F0 = npr.randn(num_n, num_m)
    F1 = np.zeros((num_n, num_m))

    # adapt step width to problem size
    num_h /= (math.pi*num_m)**(1.0/(num_n-1))

    # init variable step width
    arr_h = num_h*np.ones(num_m)
    num_h_min = 2**(-4)*num_h
    num_h_max = 2**(4)*num_h

    # calc first norm
    tr = npl.norm(X0 - X1, 'fro')

    # count the steps
    num_steps = 0

    s = time.time()
    while (tr > eps and steps == 0) or (steps > 0 and num_steps < steps):

        # calc force matrix
        F1.fill(0)

        for ii in range(0, num_m):
            # calc distances
            d1 = X0[:, ii] - X0.T
            d2 = X0[:, ii] + X0.T

            # calc reziprocal norms of distances
            dd1 = np.sum(d1**2, 1)**2
            dd2 = np.sum(d2**2, 1)**2

            # set itself to non-zero
            dd1[ii] = 1
            dd2[ii] = 1
            dd1 = 1.0/dd1
            dd2 = 1.0/dd2

            # weight the distances with their norms
            d1 = d1.T*dd1
            d2 = d2.T*dd2

            # set the wrong one to 0
            d1[:, ii] = 0
            d2[:, ii] = 0

            # sum everything together and add it to force
            F1[:, ii] -= np.sum(d1+d2, 1)
            arr_h[ii] = arr_h[ii]*2**(np.inner(F0[:, ii], F1[:, ii]) /
                                      (npl.norm(F0[:, ii])*npl.norm(F1[:, ii])))
            arr_h[ii] = min(num_h_max, max(arr_h[ii], num_h_min))

        # buffer force to calc next step width
        F0 = F1

        # calc next iterations
        X1 = csg.projSphere(X0 - arr_h*F1)
        if num_steps % 100 == 0:
            tr = npl.norm(X0 - X1, 'fro')
            print(num_steps, tr, (time.time()-s)/(num_steps+1))

        X0 = X1
        num_steps += 1

    return X0


def draw(num_n, num_m, num_eps, num_h):
    """applies the potential minimizing algorithm
    for given configuration size num_n x num_m.
    another stopping criterion has to be supplied together with a
    step length"""
    print('$MINPOT$ Starting minimizing a potential..')
    # init first draw
    X0 = csg.projSphere(npr.randn(num_n, num_m))
    X1 = np.zeros((num_n, num_m))

    # init force matrcies
    F0 = npr.randn(num_n, num_m)
    F1 = np.zeros((num_n, num_m))

    # adapt step width to number of elements to pack
    num_h /= (math.pi*num_m)**(1.0/(num_n-1))

    arr_h = num_h*np.ones(num_m)
    num_h_min = 2**(-4)*num_h
    num_h_max = 2**(4)*num_h
    tr = npl.norm(X0 - X1, 'fro')

    num_steps = 0

    s = time.time()
    while tr > num_eps:

        # calc force matrix
        F1.fill(0)
        for ii in range(0, num_m):

            # calc distances
            d1 = X0[:, ii] - X0.T
            d2 = X0[:, ii] + X0.T

            # calc reziprocal norms of distances
            dd1 = np.sum(d1**2, 1)**2
            dd2 = np.sum(d2**2, 1)**2

            # set itself to non-zero
            dd1[ii] = 1
            dd2[ii] = 1
            dd1 = 1.0/dd1
            dd2 = 1.0/dd2

            # weight the distances with their norms
            d1 = d1.T*dd1
            d2 = d2.T*dd2

            # set the wrong one to 0
            d1[:, ii] = 0
            d2[:, ii] = 0

            # sum everything together and add it to force
            F1[:, ii] -= np.sum(d1+d2, 1)

            arr_h[ii] = min(num_h_max, max(arr_h[ii]*2**(np.inner(F0[:, ii],
                                                                  F1[:, ii])/(npl.norm(F0[:, ii])*npl.norm(F1[:, ii]))), num_h_min))

        # buffer force to calc next step width
        F0 = F1

        # calc next iterations
        X1 = csg.projSphere(X0 - arr_h*F1)

        # some output
        if num_steps % 100 == 0:
            tr = npl.norm(X0 - X1, 'fro')
            print("$MINPOT$ Step %d - Correction - %f - %f" %
                  (num_steps, tr, (time.time() - s) / (num_steps + 1)))

        # buffer last iteration
        X0 = X1
        num_steps += 1

    print('$MINPOT$ Done minimizing a potential..')
    return X0


def drawComplexSeed(mat_X, num_eps, num_h, steps=0):
    """applies the potential minimizing algorithm
    for a given initial complex configuration mat_X a given amount of steps.
    another stopping criterion has to be supplied together with a
    step length"""

    # init with passed configuration
    X0 = mat_X

    # extract dimensions
    num_n, num_m = X0.shape

    # set up points in equivalent class
    num_res_phi = round(1.5*num_m)
    arr_phi = np.linspace(0, 2*math.pi*(1.0 - 1.0/num_res_phi), num_res_phi)
    num_d_phi = arr_phi[1]-arr_phi[0]

    # init new iteration
    X1 = np.zeros((num_n, num_m), dtype='complex')

    # init force matrices
    F0 = npr.randn(num_n, num_m) + 1j*npr.randn(num_n, num_m)
    F1 = np.empty((num_n, num_m), dtype='complex')
    mat_D = np.empty((num_n, num_m, num_res_phi), dtype='complex')

    # adapt step width to problem size
    num_h /= (math.pi*num_m)**(1.0/(num_n-1))

    # array for step width and their bounds
    arr_h = num_h*np.ones(num_m)
    num_h_min = 2**(-6)*num_h
    num_h_max = 2**(6)*num_h

    # get first trace
    tr = npl.norm(X0 - X1, 'fro')

    # init step count
    num_steps = 0

    s = time.time()
    while (tr > eps and steps == 0) or (steps > 0 and num_steps < steps):

        # calc force matrix
        F1.fill(0)
        for ii in range(0, num_m):

         # calc distances vectors
            mat_D = (X0[:, ii] - np.tensordot(arr_phi_exp, X0.T, axes=0)).T

            # calc distance lengths
            mat_N = np.apply_along_axis(sq_sum_c, 0, mat_D)

            # weight the distance vectors to get a force
            mat_N[ii, :] = 1
            mat_D /= mat_N**2
            mat_D[:, ii, :] = 0

            F1[:, ii] -= np.sum(mat_D[:, :, :], (1, 2))

            arr_h[ii] = arr_h[ii]*2**(abs(np.inner(F0[:, ii], F1[:, ii])) /
                                      (npl.norm(F0[:, ii])*npl.norm(F1[:, ii])))
            arr_h[ii] = min(num_h_max, max(arr_h[ii], num_h_min))

        # buffer force to calc next step width
        F0 = np.copy(F1)

        # calc next iterations
        X1 = csg.projSphere(X0 - arr_h*F1)

        # info output
        if num_steps % 50 == 0:
            tr = npl.norm(X0 - X1, 'fro')
            print(num_steps, tr, (time.time()-s)/(num_steps+1))

        # save next iteration
        X0 = X1

        num_steps += 1

    print('Done minimizing a potential..')
    return X0


def drawComplex(
    num_n,
    num_m,
    num_eps,
    num_h
):
    """applies the potential minimizing algorithm
    for given complex configuration size num_n x num_m.
    another stopping criterion has to be supplied together with a
    step length"""
    print('Starting minimizing a potential..')

    # init equivalent elements
    num_res_phi = int(round(1.5*num_m))
    arr_phi = np.linspace(0, 2*math.pi*(1.0 - 1.0/num_res_phi), num_res_phi)
    num_d_phi = arr_phi[1]-arr_phi[0]
    arr_phi_exp = np.exp(1j*arr_phi)

    # init first iteration
    X0 = csg.projSphere(npr.randn(num_n, num_m) + 1j*npr.randn(num_n, num_m))
    X1 = np.zeros((num_n, num_m), dtype='complex')

    # init forces
    F0 = npr.randn(num_n, num_m) + 1j*npr.randn(num_n, num_m)
    F1 = np.empty((num_n, num_m), dtype='complex')
    mat_D = np.empty((num_n, num_m, num_res_phi), dtype='complex')

    # setup step width according to problem
    num_h /= (math.pi*num_m)**(1.0/(num_n-1))

    # init array for variable step widths
    arr_h = num_h*np.ones(num_m)
    num_h_min = 2**(-7)*num_h
    num_h_max = 2**(7)*num_h

    # init first trace
    tr = npl.norm(X0 - X1, 'fro')

    num_steps = 0

    s = time.time()

    while tr > num_eps:

        # calc force matrix
        F1.fill(0)
        for ii in range(0, num_m):

            # calc distances vectors
            mat_D = (X0[:, ii] - np.tensordot(arr_phi_exp, X0.T, axes=0)).T

            # calc distance lengths
            mat_N = np.apply_along_axis(sq_sum_c, 0, mat_D)

            # weight the distance vectors to get a force
            mat_N[ii, :] = 1
            mat_D /= mat_N**2
            mat_D[:, ii, :] = 0

            # sum up every force
            F1[:, ii] += np.sum(mat_D[:, :, :], (1, 2))

            # update the step width depending on angle between current and last force
            arr_h[ii] = min(num_h_max, max(arr_h[ii]*2**(abs(np.inner(F0[:, ii],
                                                                      F1[:, ii]))/(npl.norm(F0[:, ii])*npl.norm(F1[:, ii]))), num_h_min))

        # save force for next iteration
        F0 = np.copy(F1)

        # calc next iterations
        X1 = csg.projSphere(X0 + arr_h*F1)

        # temporary progress
        if num_steps % 2 == 0:
            tr = npl.norm(X0 - X1, 'fro')
            print("%d,%f,%f,%f" % (num_steps, tr,
                                   (time.time() - s) / (num_steps + 1), csg.coh(X1)))

        # save last iteration
        X0 = np.copy(X1)
        num_steps += 1

    print('Done minimizing a potential..')
    return X0


def drawBuffer(num_n, num_m, num_eps, num_h, str_p):
    """generate a matrix by minimizing a potential with dimension num_n x num_m
        and store it to str_path. it gets regenerated if specified dimensions changed"""
    if os.path.isfile(str_p+'.npy'):
        M = np.load(str_p+'.npy')
        if M.shape[0] == num_n and M.shape[1] == num_m:
            print("Found buffered matrix at " + str_p + ".npy")
            return M
        else:
            print("Dimensions changed, regenerating")
            M = draw(num_n, num_m, num_eps, num_h)
            np.save(str_p, M)
            print("Saving to " + str_p + ".npy")
            return M
    else:
        print("No file " + str_p + ".npy present, regenerating")
        M = draw(num_n, num_m, num_eps, num_h)
        np.save(str_p, M)
        return M


def drawBufferComplex(num_n, num_m, num_eps, num_h, str_p):
    """generate a complex valued matrix by minimizing a potential with dimension num_n x num_m
        and store it to str_path. it gets regenerated if specified dimensions changed"""
    if os.path.isfile(str_p+'_c.npy'):
        M = np.load(str_p+'_c.npy')
        if M.shape[0] == num_n and M.shape[1] == num_m:
            print("Found buffered matrix at " + str_p + "_c.npy")
            return M
        else:
            print("Dimensions changed, regenerating")
            M = drawComplex(num_n, num_m, num_eps, num_h)
            np.save(str_p, M)
            print("Saving to " + str_p + "_c.npy")
            return M
    else:
        print("No file " + str_p + "_c.npy present, regenerating")
        M = drawComplex(num_n, num_m, num_eps, num_h)
        np.save(str_p, M)
        return M
