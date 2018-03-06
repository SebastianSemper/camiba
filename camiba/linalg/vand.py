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
import os.path
import math
import numpy as np
import numpy.linalg as npl
from ..linalg.basic import coh


def opt_c(num_n, num_m, num_tries, num_samples, str_path):
    """
    generate an approximately optimal parameter c for the pack routine
    for a vandermonde matrix of size num_n x num_m and store it into a table
    at location str_path. num_Tries and num_Samples specify how many samples we
    take for directional random search.
    """
    print('VANDER:', num_n, num_m)
    not_found = False
    no_file = False

    # check if file is present
    if os.path.isfile(str_path+'.npy'):

        # load table of parameters c
        mat_c_opt = np.load(str_path+'.npy')

        # search given dimensions and precision in the table
        arr_search = np.apply_along_axis(
            np.all,
            1,
            (mat_c_opt[:, 0:4] == [num_n, num_m, num_tries, num_samples])
        )

        # if we found it, return it
        if np.any(arr_search):
            return mat_c_opt[arr_search, 4][0]
        else:
            not_found = True
    else:
        no_file = True

    if not_found or no_file:
        # if we had no luck in finding a buffered value
        # we generate one

        if no_file:
            mat_c_opt = np.zeros((1, 5))

        # start with initial guess as .5 and search width as .5
        res_c = 1.0
        num_l = 0.9**(0.1*num_n)

        # iteratively shrinken the intervall where we search
        for ii in range(0, num_tries):
            arr_c = np.linspace(res_c - num_l, res_c + num_l, num_samples)
            arr_coh = np.array(
                list(map(lambda c: coh(pack(num_n, num_m, c)), arr_c)))
            res_c = arr_c[np.argmin(arr_coh)]
            print(res_c)
            num_l *= 0.8

        # append it to the current solution
        mat_c_opt = np.vstack(
            [mat_c_opt, [num_n, num_m, num_tries, num_samples, res_c]])

        # save the file
        np.save(str_path+'.npy', mat_c_opt)

        # also return the parameter
        return res_c


def build(arr_z, num_n):
    """build a vandermonde matrix with specified
    first row arr_z and height num_n"""

    mat_V = np.empty((num_n, arr_z.shape[0]), dtype='complex_')
    for ii in range(0, num_n):
        mat_V[ii, :] = arr_z**(ii)

    return mat_V


def pack(
    num_n,
    num_m,
    num_c1
):
    """apply the special packing pattern for the first row
    to generate a vandermonde matrix with (pretty) low coherence"""

    num_n = int(num_n)
    num_m = int(num_m)
    num_M = int(math.ceil(num_m*0.5)*2)

    arr_z = np.zeros(num_m, dtype='complex128')
    arr_phi = np.zeros(num_m)
    arr_c = np.empty(num_m)

    arr_phi = np.linspace(0, 2*math.pi - 2*math.pi/num_M, num_M)
    arr_phi = arr_phi[0:num_m]
    arr_b = np.empty((int(num_M/2), 2))
    arr_b[:, 0:2] = [num_c1, 1/num_c1]
    arr_c = arr_b.view()
    arr_c.shape = (num_M)
    arr_c = arr_c[0:num_m]

    arr_z = arr_c*(np.cos(arr_phi) + 1j*np.sin(arr_phi))
    arr_n = np.arange(num_n)

    mat_V = np.empty((num_n, num_m), dtype='complex128')
    for ii in range(0, num_n):
        mat_V[ii, :] = arr_z**(ii+1)

    return mat_V


def draw(
    num_n,
    num_m,
    str_path
):
    num_c = opt_c(num_n, num_m, 25, 200, str_path)
    return pack(num_n, num_m, num_c)
