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
import numpy.random as npr


class Scenario:
    """
        General Compressed Sensing Scenario

    This class offers a very convenient but still abstract interface to a
    standard compressed sensing scenario, which generally consists of a
    dictionary matrix, a compression matrix and a reconstruction scheme
    represented by an algorithm.
    """

    def __init__(self, mat_D, mat_C, algo):
        """
            CS Scenario Constructor

        The user has to specify the dictionary and measurement matrices
        and an reconstruction algorithm.

        Parameters
        ----------

        mat_D : ndarray
            dictionary matrix
        mat_C : ndarray
            compression matrix
        algo : method
            reconstruction algorithm
        """
        self._num_n, self._num_m = mat_D.shape
        self._num_c = mat_C.shape[0]

        self._mat_D = mat_D
        self._mat_C = mat_C

        self._mat_A = mat_C.dot(mat_D)

        self._algo = algo

    def gen_sparse(self, num_s, entries=[], do_complex=False):
        """
            Generate a Sparse Signal

        This function generates a sparse ground truth signal according
        to the scenarios dimensions of a given sparsity order and makes
        use of a certain set of specified elements if provided.

        Parameters
        ----------

        num_s : int
            desired sparsity order
        entries : list
            list of possible values
        do_complex : bool
            whether the elements are complex or not

        Returns
        -------
        ndarray
            the sparse vector
        """
        dataType = ["float", "complex"][1 * do_complex]

        arr_x = np.zeros(self._num_m, dtype=dataType)
        if entries == []:
            if do_complex:
                arr_s = npr.randn(num_s) + 1j*npr.randn(num_s)
            else:
                arr_s = npr.randn(num_s)
        else:
            arr_s = npr.choice(entries, num_s, replace=True)

        arr_x[npr.choice(range(self._num_m), num_s, replace=False)] = arr_s
        return arr_x

    def gen_sparse_sep(self, num_s, dist, entries=[], do_complex=False):
        """
            Generate a Sparse Signal with Separation Condition

        This function generates a sparse ground truth signal according
        to the scenarios dimensions of a given sparsity order and makes
        use of a certain set of specified elements if provided. Moreover,
        a separation condition is used to not put support elements closer
        than this distance. Ultimatively this generates more "well behaved"
        signals in terms of reconstruction.

        Parameters
        ----------

        num_s : int
            desired sparsity order
        num_dist : int
            desired separation distance
        entries : list
            list of possible values
        do_complex : bool
            whether the elements are complex or not

        Returns
        -------
        ndarray
            the sparse signal
        """
        dataType = ["float", "complex"][1 * do_complex]

        arr_x = np.zeros(self._num_m, dtype=dataType)

        arr_possible = np.array(range(self._num_m), dtype='int')

        for ii in range(num_s):
            prob = npr.choice(arr_possible, 1, replace=False)

            if entries == []:
                if compl:
                    arr_s = npr.randn(num_s) + 1j*npr.randn(num_s)
                else:
                    arr_s = npr.randn(num_s)
            else:
                arr_x[prob] = npr.choice(entries, 1, replace=False)

            arr_possible = arr_possible[np.abs(arr_possible - prob) > dist]

            if len(arr_possible) == 0:
                print("Could not fit all in! Returning a less sparse x!")
                return arr_x

        return arr_x

    def compress(self, arr_y):
        """
            Apply Compression

        Parameters
        ----------
        arr_y : ndarray
            signal to compress

        Returns
        -------
        ndarray
            the compressed measurement
        """
        return self._mat_C.dot(arr_y)

    def to_signal(self, arr_x):
        """
            Transform to signal

        This method takes a sparse vector and multiplies it with the dictionary
        to get the acutal signal from its sparse representation. Often this
        is called forward model.

        Parameters
        ----------
        arr_x : ndarray
            sparse vector

        Returns
        -------
        ndarray
            the signal
        """
        return self._mat_D.dot(arr_x)

    def recover(self, arr_b, args):
        """
            Recover a sparse vector

        Given a measurement and additional arguments for the recovery method
        we call this very method and return its result.

        Parameters
        ----------
        arr_b : ndarray
            measurement vector
        args : dict
            recovery algorithms arguments

        Returns
        -------
        ndarray
            the sparse reconstructed vector
        """
        return self._algo(self._mat_A, arr_b, **args)

    def pipeline(self, arr_x):
        """
        Transform to measurment

        This method takes a sparse vector and multiplies it with the dictionary
        to get the acutal signal from its sparse representation. Often this
        is called forward model. Then also the defined compression step is
        applied and we implemented the whole pipelein

        Parameters
        ----------
        arr_x : ndarray
            sparse vector

        Returns
        -------
        ndarray
            the compressed measurement
        """
        arr_y = self.to_signal(arr_x)
        arr_b = self.compress(arr_y)
        return self.recover(arr_b)

    def phase_trans_rec(self,
                        num_s,
                        fun_x,
                        args,
                        arr_snr,
                        fun_noise,
                        dct_fun_compare,
                        num_trials
                        ):
        """
            Calculate a phase transition

        Given the scenario we generate sparse vectors with a certain fixed
        sparsity level and according to a provided scheme. Then we apply the
        forward model and the compression and add noise to the compressed
        measurements, which is generated by a provided noise generating
        function. Then we aim a reconstructing the original signal from this
        compressed and noisy data. This is done several times for each
        level of SNR and after each reconstruction, we compare the
        reconstruction with respect to one or more provided error metrics.
        Finally everything is saved and returned in a dictionary where the keys
        are given by the SNR and the name of the applied error metric.

        Parameters
        ----------
        num_s : int
            sparsity level
        fun_x : method
            function, which takes the sparsity level as parameter to generate
            a sparse vector
        args : dict
            arguments for the recovery algorithm
        arr_snr : ndarray
            all level of SNR to go through
        fun_noise : ndarray
            noise generating function, taking the snr and the vector
            size as arguments
        dct_fun_compare : dict
            dictionary of comparison function
        num_trials : int
            number of trials to run at each snr level

        Returns
        -------
        ndarray
            the compressed measurement
        """

        # dictionary of result with keys from dct_fun_compare
        dct_res = {}
        for ii in dct_fun_compare.items():
            dct_res.update({ii[0]: np.zeros((
                len(arr_snr),            # for each snr level
                num_trials                # for each trial
            ))})

        # go through trials and SNR
        for ii, snr in enumerate(arr_snr):
            for jj in range(num_trials):

                # generate ground truth and noisy measurement
                arrX = fun_x(num_s)
                arrB = self.compress(arrX) + fun_noise(snr, self._num_c)

                # do reconstruction
                arrXEst = self.recover(arrB, args)

                # apply all the comparison functions and store results
                for kk in dct_res.items():
                    key = kk[0]
                    dct_res[key][ii, jj] = dct_fun_compare[key](arrX, arrXEst)

        # take the mean in each result across all trials
        for ii in dct_res.items():
            key = ii[0]
            dct_res[key] = np.mean(dct_res[key], axis=1)

        return dct_res
