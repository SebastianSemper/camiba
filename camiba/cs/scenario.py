import numpy as np
import numpy.random as npr


class Scenario:
    def __init__(self,
                 matD,
                 matC,
                 algo
                 ):
        self._numN, self._numM = matD.shape
        self._numK = matC.shape[0]

        self._matD = matD
        self._matC = matC

        self._matA = matC.dot(matD)

        self._algo = algo

    def gen_sparse(self,
                   num_s,
                   entries=[],
                   compl=False
                   ):
        dataType = ["float", "complex"][1*compl]

        arr_x = np.zeros(self._numM, dtype=dataType)
        if entries == []:
            if compl:
                arr_s = npr.randn(num_s) + 1j*npr.randn(num_s)
            else:
                arr_s = npr.randn(num_s)
        else:
            arr_s = npr.choice(entries, num_s, replace=True)

        arr_x[npr.choice(range(self._numM), num_s, replace=False)] = arr_s
        return arr_x

    def gen_sparse_sep(
        self,
        num_s,
        dist,
        entries=[],
        do_complex=False
    ):
        dataType = ["float", "complex"][1*do_complex]

        arr_x = np.zeros(self._numM, dtype=dataType)

        arr_possible = np.array(range(self._numM), dtype='int')

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

    def compress(self,
                 arr_y
                 ):
        return self._matC.dot(arr_y)

    def toSignal(self,
                 arrX
                 ):
        return self._matD.dot(arrX)

    def recover(self,
                arrB,
                args
                ):
        return self._algo(
            self._matA,
            arrB,
            **args
        )

    def pipeLine(self,
                 arrX
                 ):
        arrY = self.toSignal(arrX)
        arrB = self.compress(arrY)
        return self.recover(arrB)

    def phase_trans_rec(self,
                        num_s,                # fixed sparsity order
                        fun_x,                # function to generate ground truth
                        args,                # arguments for reconstruction
                        arrSNR,                #
                        funNoise,            # function that generates noise
                        dct_fun_compare,    # comparison functions
                        numTrials            # number of trials to run
                        ):
        """
            calculate a phase transition where do the reconstruction

            returns a dictionary with keys taken from dictionary of compare
            functions
        """

        # dictionary of result with keys from dct_fun_compare
        dct_res = {}
        for ii in dct_fun_compare.items():
            dct_res.update({ii[0]: np.zeros((
                len(arrSNR),            # for each snr level
                numTrials                # for each trial
            ))})

        # go through trials and SNR
        for ii, snr in enumerate(arrSNR):
            for jj in range(numTrials):

                # generate ground truth and noisy measurement
                arrX = fun_x(num_s)
                arrB = self.compress(arrX) + funNoise(snr, self._numK)

                # do reconstruction
                arrXEst = self.recover(arrB, args)

                # apply all the comparision functions and store results
                for kk in dct_res.items():
                    key = kk[0]
                    dct_res[key][ii, jj] = dct_fun_compare[key](arrX, arrXEst)

        # take the mean in each result across all trials
        for ii in dct_res.items():
            key = ii[0]
            dct_res[key] = np.mean(dct_res[key], axis=1)

        return dct_res
