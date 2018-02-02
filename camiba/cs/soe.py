"""
Here we implement a Class, which provides different methods for sparsity order
estimation from compressed measurements.

for EET see:
http://www.eurasip.org/Proceedings/Eusipco/Eusipco2014/HTML/papers/1569925343.pdf

for EFT see:

"""

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import os

from ..linalg.basic import proj_sphere
from ..linalg.basic import khatri_rao
from ..linalg.vand import *
from ..algs import omp
from .scenario import Scenario


class Soe(Scenario):

    """
    Here we implement a Class, which provides different methods for sparsity
    order estimation from compressed measurements. Many of them are very good.
    """

    def __init__(
            self,
            num_n,
            num_m,
            mos_method='eft',
            algorithm=omp.recover,
            **kwargs
    ):

        # save the MOS algorithm
        self._mos_method = mos_method

        if self._mos_method == 'lopes':

            # evenly divide the measurement
            self.num_m1 = int(num_m/2)
            self.num_m2 = num_m - self.num_m1
            self._num_gamma = kwargs['num_gamma']

            # init measurement matrix
            matMeasurement = np.zeros((num_m, num_n))

            # generate cauchy measurements
            matMeasurement[:self.num_m1, :] = npr.standard_cauchy(
                (self.num_m1, num_n)
            ) * self._num_gamma

            # generate gaussian measurements
            matMeasurement[self.num_m1:, :] = npr.randn(
                self.num_m2,
                num_n
            ) * self._num_gamma

            # set appropriate estimation function
            self._estimate_function = self._est_lopes

        elif mos_method == 'ravazzi':
            self._num_gamma = kwargs['num_gamma']
            matUniform = npr.uniform(0, 1, (num_m, num_n))
            matNormal = (1 / np.sqrt(self._num_gamma)) * \
                npr.randn(num_m, num_n)
            matSubSel = 1 * (matUniform < self._num_gamma)

            matMeasurement = matSubSel * matNormal

            self._estimate_function = self._est_ravazzi

        else:

            # path to store training data to
            self._buffer_path = kwargs['str_path']

            # overlap parameter
            self._num_p = kwargs['num_p']

            # error probability during training
            self._num_err_prob = kwargs['num_err_prob']

            # make scenario complex if we do overlap
            self._do_complex = ((num_p != 0) or (kwargs['do_complex']))

            if self._num_p == 0:

                # find dimensions most suitable
                self._num_l, self._num_k = self._largest_div(num_m)

                # create the matrices
                if self._do_complex:
                    mat_psi = npr.randn(self._num_l, num_n) + \
                        1j * npr.randn(self._num_l, num_n)

                    mat_phi = npr.randn(self._num_k, num_n) + \
                        1j * npr.randn(self._num_k, num_n)

                else:
                    mat_psi = npr.randn(self._num_l, num_n)
                    mat_phi = npr.randn(self._num_k, num_n)

                self._estimate_function = self._nope_overlap

            else:
                mat_psi = vand.draw(
                    int(math.ceil(float(num_m)/self._num_p)),
                    num_n,
                    self._buffer_path + "vander_c"
                )
                mat_phi = npr.randn(self._num_p, num_n) + \
                    1j * npr.randn(self._num_p, num_n)

                self._num_l = self._find_block_length(num_m, self._num_p)
                self._num_k = int((num_m - self._num_l) / self._num_p + 1)

                # set appropriate estimation function
                self._estimate_function = self._true_overlap

            # measurement is the KRP of scaled KRP of vandermonde
            # where we only take the first m rows
            matMeasurement = general.proj_sphere(
                krp.prod(mat_psi, mat_phi)[:num_m, :]
            )

            if mos_method == 'eft':
                # generate the array with the helper values
                self._num_q = min(self._num_l, self._num_k)
                self._arr_r = self._eft_fetch_arr_r(
                    self._num_q,
                    self._num_l,
                    str_path+"eft_arr_r_"+str(self._num_l)+"_"+str(self._num_k)
                )

                # fetch the thresholding coefficients
                self._arr_eta = self._eft_fetch_arr_eta(
                    self._num_l,
                    self._num_k,
                    num_err_prob,
                    self._do_complex,
                    str_path + "eft_arr_eta_" +
                    str(num_err_prob) + "_" +
                    int(self._do_complex) * "c_" +
                    str(self._num_l) + "_" + str(self._num_k)
                ) + 0.4

            elif mos_method == 'eet':
                self._arr_eta = self._eet_fetch_arr_eta(
                    self._num_l,
                    self._num_k,
                    self._num_err_prob,
                    self._do_complex,
                    str_path + "eet_arr_eta_" +
                    str(self._num_err_prob) + "_" +
                    int(self._do_complex) * "c_" +
                    str(self._num_l) + "_" + str(self._num_k)
                )
            elif mos_method == 'new':
                self._arr_eta = self._new_fetch_arr_eta(
                    self._num_l,
                    self._num_k,
                    self._num_err_prob,
                    self._do_complex,
                    str_path + "new_arr_eta_" +
                    str(self._num_err_prob) + "_" +
                    int(self._do_complex) * "c_" +
                    str(self._num_l) + "_" + str(self._num_k)
                )

        # now create the cs scenario
        sc.Scenario.__init__(
            self,						# yay!
            np.eye(num_n),				# during soe the dictionary is the
            # identity matrix, i.e. the vector
            # itself is sparse
            matMeasurement,
            algorithm					# recovery is done with OMP
        )

    def estimate(
        self,
        arr_b
    ):
        """
            actual interface function that allows this class
            to perform SOE given a single vector or an array of vectors
        """

        if len(arr_b.shape) > 1:
            return np.apply_along_axis(
                self._estimate_function, 0, arr_b
            )
        else:
            return self._estimate_function(arr_b)

    def _eft_fetch_arr_r(
        self,
        num_m,
        num_n,
        str_p
    ):
        """ Helper function to calculate r and store r helper array"""

        if os.path.isfile(str_p+'.npy'):
            return(np.load(str_p+'.npy'))
        else:
            arr_r = np.zeros(num_m)

            # calculate the array for every entry
            # starting at value 1
            for ii in range(1, num_m):

                num_M = float(ii+1)

                num_r_old = 1
                num_r_new = 0

                num_dim_frac = (num_M + float(num_n))/float(num_n*num_M)

                while abs(num_r_old - num_r_new) > 1e-15:
                    num_r_old = num_r_new

                    num_r_m = num_r_old**num_M
                    num_r_prod = (1.0 - num_r_m)*(1.0 + num_r_old)

                    num_r_new = -(num_dim_frac*num_r_prod -
                                  1.0 - num_r_m*(1.0 - num_r_old))

                arr_r[ii] = num_r_new

            np.save(str_p+'.npy', arr_r)
            return arr_r

    def _eft_fetch_arr_eta(
        self,
        num_N,
        num_M,
        num_err_prob,
        doComplex,
        str_p
    ):
        """Given dimensions and the desired probability of error, and
        whether to work complex, we generate training data on noise
        and save the trained coefficients in str_p"""

        if os.path.isfile(str_p+'.npy'):
            return(np.load(str_p+'.npy'))
        else:
            print('$SOE$    Starting with EFT training...')

            # dynamically generate enough trials
            num_trials = int(round(10.0*1.0/(num_err_prob)))

            # adapt to complex case
            if doComplex:
                ten_trials = (npr.randn(num_N, num_M, num_trials) +
                              1j*npr.randn(num_N, num_M, num_trials)
                              )/math.sqrt(2.0)
            else:
                ten_trials = npr.randn(num_N, num_M, num_trials)

            # generate values for eta as thresholding parameter
            res_eta = 10000
            grd_eta = np.linspace(-2, 2, res_eta)

            # matrix for storing number of false alarms
            mat_FA_prb = np.zeros((res_eta, self._num_q))

            # go through all trials
            for ii in range(0, num_trials):
                if ii % int(float(num_trials)*0.1) == 0:
                    print("$SOE$    %d%% " % int(float(ii)/num_trials*100))

                # reverse array of singular values
                arr_SV = np.flipud(
                    npl.svd(ten_trials[:, :, ii],
                            compute_uv=0)**2
                )

                # go through every dimension
                for jj in range(1, self._num_q+1):
                    jj = int(jj-1)
                    num_sig = np.mean(arr_SV[0:(jj+1)])
                    num_SV_est = (jj + 1) * \
                        ((1 - self._arr_r[jj])
                         / (1 - self._arr_r[jj]**(jj + 1))
                         )*num_sig

                    num_ratio = (arr_SV[jj] - num_SV_est)/num_SV_est
                    mat_FA_prb[num_ratio > grd_eta, jj - 1] += 1.0

            # normalize to probability
            mat_FA_prb /= float(num_trials)

            # invert the relation between eta and probability
            arr_eta = np.zeros(self._num_q)
            for ii in range(0, self._num_q):
                ind_eta = 0
                while mat_FA_prb[ind_eta, ii] > num_err_prob and ind_eta + 2 < res_eta:
                    ind_eta += 1

                # print(mat_FA_prb[ind_eta,ii])
                arr_eta[ii] = grd_eta[ind_eta]

            np.save(str_p+'.npy', arr_eta)
            print('$SOE$    Done with EFT training...')
            return arr_eta

    def _eet_fetch_arr_eta(
        self,
        num_N,
        num_M,
        num_err_prob,
        doComplex,
        str_p
    ):
        """Given dimensions and the desired probability of error, and
        whether to work complex, we generate training data on noise
        and save the trained coefficients in str_p"""

        if os.path.isfile(str_p+'.npy'):
            return(np.load(str_p+'.npy'))
        else:
            print('$SOE$    Starting with EET training...')

            # dynamically generate enough trials
            num_trials = int(round(500.0*1.0/(num_err_prob)))

            # adapt to complex case
            if doComplex:
                mat_trials = (npr.randn(self._numK, num_trials) +
                              1j*npr.randn(self._numK, num_trials)
                              )/math.sqrt(2.0)
            else:
                mat_trials = npr.randn(self._numK, num_trials)

            num_P = min(self._num_k, self._num_l)

            arr_sv = np.empty((num_P, num_trials))
            ten_trials = np.empty(
                (self._num_l, self._num_k, num_trials), dtype=mat_trials.dtype)

            if self._num_p == 0:
                for ii in range(num_trials):
                    ten_trials[:, :, ii] = mat_trials[:, ii].reshape(
                        self._num_l, self._num_k)
            else:
                for ii in range(num_trials):
                    ten_trials[:, :, ii] = self._reshape_measurement(
                        mat_trials[:, ii])

            for ii in range(num_trials):
                arr_sv[:, ii] = npl.svd(ten_trials[:, :, ii], compute_uv=0)

            arr_quot = arr_sv[:(num_P-1), :] / arr_sv[1:, :]

            num_Q = 20000
            num_max_ratio = np.max(arr_quot, axis=1)

            num_tau = num_max_ratio/num_Q

            arr_P = np.empty(num_Q)

            arr_eta = np.zeros(num_P - 1)

            lst_Q = list(range(num_Q))

            for ii in range(num_P-1):
                for qq in range(num_Q-1):
                    arr_P[qq] = np.sum(
                        (arr_quot[ii, :] >= num_tau[ii]*qq) *
                        (arr_quot[ii, :] < num_tau[ii]*(qq+1))
                    )/num_trials

                arr_abs = np.abs((1 - np.cumsum(arr_P) + self._num_err_prob))
                num_j_eta = np.argmin(arr_abs)
                arr_eta[ii] = num_j_eta*num_tau[ii]

            np.save(str_p+'.npy', arr_eta)
            print('$SOE$    Done with EET training...')
            return arr_eta

    def _new_fetch_arr_eta(
        self,
        num_N,
        num_M,
        num_err_prob,
        doComplex,
        str_p
    ):
        if os.path.isfile(str_p+'.npy'):
            return(np.load(str_p+'.npy'))
        else:
            print('$SOE$    Starting with NEW training...')

            # dynamically generate enough trials
            num_trials = int(round(500.0*1.0/(num_err_prob)))

            # adapt to complex case
            if doComplex:
                mat_trials = 2*(npr.randn(self._numK, num_trials) +
                                1j*npr.randn(self._numK, num_trials)
                                )/math.sqrt(2.0)
            else:
                mat_trials = 2*npr.randn(self._numK, num_trials)

            num_P = min(self._num_k, self._num_l)

            arr_sv = np.empty((num_P, num_trials))
            ten_trials = np.empty(
                (self._num_l, self._num_k, num_trials), dtype=mat_trials.dtype)

            if self._num_p == 0:
                for ii in range(num_trials):
                    ten_trials[:, :, ii] = mat_trials[:, ii].reshape(
                        self._num_l, self._num_k)
            else:
                for ii in range(num_trials):
                    ten_trials[:, :, ii] = self._reshape_measurement(
                        mat_trials[:, ii])

            for ii in range(num_trials):
                arr_sv[:, ii] = npl.svd(ten_trials[:, :, ii], compute_uv=0)**2

            arr_ratios = np.empty((num_P - 1, num_trials))
            for ii in range(num_P-1):
                arr_ratios[ii, :] = arr_sv[ii, :] / \
                    np.mean(arr_sv[(ii+1):, :], 0)

            arr_eta = np.zeros((num_P - 1, 2))
            for ii in range(num_P-1):
                arr_eta[ii, 0] = np.percentile(
                    arr_ratios[ii, :], 50.0*self._num_err_prob)
                arr_eta[ii, 1] = np.percentile(
                    arr_ratios[ii, :], 100. - 50.0*self._num_err_prob)

            np.save(str_p+'.npy', arr_eta)
            print('$SOE$    Done with NEW training...')
            return arr_eta

    def _reshape_measurement(
        self,
        vec_b
    ):
        # create a now empty matrix with the same datatype as into
        # where we shape the input into
        mat_B = np.empty(
            (self._num_l, self._num_k),
            dtype=vec_b.dtype
        )

        # do the reshapig with overlap
        for ii in range(0, self._num_k):
            mat_B[:, ii] = vec_b[(ii*self._num_p)                                 :(ii*self._num_p + self._num_l)]

        return mat_B

    def _do_eet(
        self,
        mat_B
    ):
        arr_sv = npl.svd(mat_B, compute_uv=0)
        N = len(arr_sv)
        for ii in range(N-1):
            #print((arr_sv[N-ii-2] / arr_sv[N - ii - 1]) , self._arr_eta[N-ii-2])
            if (arr_sv[N-ii-2] / arr_sv[N - ii - 1]) > self._arr_eta[N-ii-2]:
                return N - ii - 1
            # else:
                # print('BOOOM')
        return 1

    def _do_eft(
        self,
        mat_B
    ):

        # get the singular values
        arr_SV = npl.svd(mat_B, compute_uv=0)**2

        # do the threshold test
        for jj in range(1, self._num_q+1):
            jj = int(jj-1)
            num_sig = np.mean(arr_SV[0:(jj+1)])
            num_SV_est = num_SV_est = (jj + 1) * \
                ((1 - self._arr_r[jj])
                 / (1 - self._arr_r[jj]**(jj + 1))
                 )*num_sig

            num_ratio = (arr_SV[jj] - num_SV_est)/num_SV_est

            # use training data to distinguish between noise and actual data
            if num_ratio >= self._arr_eta[jj]:
                return (self._num_l - jj)

        return 0

    def _do_new(
        self,
        mat_B
    ):

        # get the singular values
        arr_SV = npl.svd(mat_B, compute_uv=0)**2

        num_P = min(self._num_k, self._num_l)

        for ii in range(num_P - 1):
            quot = (arr_SV[num_P - 2 - ii] /
                    np.mean(arr_SV[(num_P - 1 - ii):]))
            if quot > self._arr_eta[num_P - 2 - ii, 1] or quot < self._arr_eta[num_P - 2 - ii, 0]:
                return num_P - 1 - ii

        return 1

    def _nope_overlap(
        self,
        vec_b
    ):
        """	estimation routine that reshapes b without
            reuing any elements.
        """
        mat_B = vec_b.reshape(self._num_l, self._num_k)

        # initiate the model order selection
        if self._mos_method == 'eft':
            return self._do_eft(mat_B)
        elif self._mos_method == 'eet':
            return self._do_eet(mat_B)
        elif self._mos_method == 'new':
            return self._do_new(mat_B)

    def _true_overlap(
        self,
        vec_b		# vector to do SOE on
    ):
        """
            estimation routine that reshapes b by reusing some elements according to some overlap
        """

        mat_B = self._reshape_measurement(vec_b)

        # initiate the model order selection
        if self._mos_method == 'eft':
            return self._do_eft(mat_B)
        elif self._mos_method == 'eet':
            return self._do_eet(mat_B)
        elif self._mos_method == 'new':
            return self._do_new(mat_B)

    def _largest_div(
        self,
        num_n		# number to split into factors
    ):
        """
            Calculates the largest divisor of a natural number num_n
        """

        num_d = math.ceil(math.sqrt(num_n))
        while True:
            if num_n % num_d == 0:
                return (int(num_d), int(num_n/num_d))
            num_d -= 1

    def _find_block_length(
        self,
        num_m,		# the number of measurements we took
        num_p		# the block advance we chose
    ):
        """
            Finds the optimal block length num_l
            for given dimensions and block advance
        """

        num_l_init = int(math.ceil((float(num_m) + float(num_p))/(num_p+1.0)))
        num_l1 = num_l_init
        num_l2 = num_l_init - 1
        while True:
            if (num_m - num_l1) % num_p == 0:
                return int(num_l1)

            if (num_m - num_l2) % num_p == 0:
                return int(num_l2)

            num_l1 += 1
            num_l2 -= 1

    def _est_lopes(
        self,
        vec_b
    ):
        numT1 = np.median(np.abs(vec_b[: self.num_m1])) / self._num_gamma
        numT2 = np.mean(vec_b[self.num_m1:] ** 2) / (self._num_gamma ** 2)

        return int(np.round(numT1 ** 2 / numT2))

    def _est_ravazzi(
        self,
        vec_b
    ):
        num_s = 10
        num_k = 10
        arr_ka = np.zeros(num_s)
        arr_pi = np.zeros((self._numK, num_s))
        arr_al = np.zeros(num_s)
        arr_be = np.zeros(num_s)
        arr_pe = np.zeros(num_s)

        arr_pi[:, 0] = 0.5
        arr_al[0] = 500
        arr_pe[0] = 0.01
        arr_be[0] = 200

        for ii in range(num_s - 1):
            # E-step
            q1 = (arr_pe[ii])/np.sqrt(arr_al[ii])
            q2 = (1 - arr_pe[ii])/np.sqrt(arr_be[ii])
            e1 = np.exp(- vec_b ** 2 / (2 * arr_al[ii]))
            e2 = np.exp(- vec_b ** 2 / (2 * arr_be[ii]))
            arr_pi[:, ii + 1] = (q1 * e1)/(q1 * e1 + q2 * e2)

            # M-Step
            sum1 = np.sum(arr_pi[:, ii + 1])
            sum2 = np.sum(1 - arr_pi[:, ii + 1])
            arr_pe[ii + 1] = sum1 / self._numK
            arr_ka[ii + 1] = np.log(arr_pe[ii + 1]) \
                / np.log(1 - self._num_gamma)

            arr_al[ii + 1] = np.inner(arr_pi[:, ii + 1], vec_b ** 2) / sum1
            arr_be[ii + 1] = np.inner(1 - arr_pi[:, ii + 1], vec_b ** 2) / sum2
            print(arr_ka[ii+1])

        return int(arr_ka[num_s - 1])

    def phase_trans_est(self,
                        num_s,				# fixed sparsity order
                        fun_x,				# ground truth generation
                        arrSNR,				# array for SNR range
                        funNoise,			# noise generating function
                        dct_fun_compare,  # dictionary of comparison functions
                        numTrials			# number of trials for each SNR level
                        ):
        """
            calculate a phase transition where do SOE

            returns a dictionary with keys taken from dictionary of compare
            functions
        """
        dct_res = {}
        for ii in dct_fun_compare.items():
            dct_res.update({ii[0]: np.zeros((
                len(arrSNR),			# for each snr level
                numTrials				# for each trial
            ))})

        for ii, snr in enumerate(arrSNR):
            for jj in range(numTrials):

                # generate ground truth and noisy measurement
                arrX = fun_x(num_s)
                arrB = self.compress(arrX) + funNoise(snr, self._numK)

                # estimate the sparsity order
                num_s_est = self.estimate(arrB)

                for kk in dct_res.items():
                    key = kk[0]
                    dct_res[key][ii, jj] = dct_fun_compare[key](
                        arrX, num_s_est)

        for ii in dct_res.items():
            key = ii[0]
            dct_res[key] = np.mean(dct_res[key], axis=1)

        return dct_res

    def phase_trans_est_rec(self,
                            num_s,				# fixed sparsity order
                            fun_x,				# ground truth generation
                            args,				# preliminary arguments
                            arrSNR,				# array for SNR range
                            funNoise,			# noise generating function
                            dct_fun_compare,  # function to compare recovery to ground truth
                            numTrials			# number of trials for each SNR level
                            ):
        """
            calculate a phase transition where we first do SOE and then
            the reconstruction, which might get better with this knowledge

            returns a dictionary with keys taken from dictionary of compare
            functions
        """

        # dictionary with results
        dct_res = {}

        # initialize with keys from compare function
        for ii in dct_fun_compare.items():
            dct_res.update({ii[0]: np.zeros((
                len(arrSNR),			# for each snr level
                numTrials				# for each trial
            ))})

        # g through SNR and trials
        for ii, snr in enumerate(arrSNR):
            for jj in range(numTrials):

                # generate ground truth and noisy measurement
                arrX = fun_x(num_s)
                arrB = self.compress(arrX) + funNoise(snr, self._numK)

                # estimate the sparsity order
                num_s_est = self.estimate(arrB)

                # add the estimated order to the params of the recovery
                # TODO: update it if already present
                args.update({'num_k': num_s_est})

                # do recovery with estimated sparsity
                arr_x_est = self.recover(arrB, args)

                # write all the results into the appropriate place
                for kk in dct_res.items():
                    key = kk[0]
                    dct_res[key][ii, jj] = dct_fun_compare[key](
                        arrX, arr_x_est)

        # calculate
        for ii in dct_res.items():
            key = ii[0]
            dct_res[key] = np.mean(dct_res[key], axis=1)

        return dct_res
