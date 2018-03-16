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
"""
Here methods for data processing are provided, which mostly serve the purpose
to export data in such a way that it is easy to display in LaTeX using
PGFplots and TikZ.
"""

import numpy as np
import math
import json


def mat_to_heat(d, x, y, p):
    """
        Convert 2D Array to a heatmap

    This function takes a 2D ndarray and writes the data to disk such that
    it can be plotted as a 2D heatmap in pgfplots.

    Parameters
    ----------
    d : ndarray
        the array to plot
    x : ndarray
        x-axis range
    y : ndarray
        y-axis range
    p : string
        path to save to
    """
    num_n, num_m = d.shape

    with open(p, 'w') as o:
        for nn in range(num_n):
            for mm in range(num_m):
                o.write(' '.join('%.16f' %
                                 num for num in [y[nn], x[mm], d[nn, mm]]))
                o.write('\n')
            o.write('\n')


def dict_to_csv(d, p):
    """
        Write a Dictionary to CSV

    This routine takes a dictionary and uses the keys to use as header
    for the values writen column wise into a csv file, where we assume that
    the values are numpy arrays of the same length

    Parameters
    ----------
    d : dict
        dictionary of data to write
    p : string
        path to write to
    """

    # header list
    h = []

    # first data array
    f = list(d.items())[0]

    dtype = np.int
    for ii, tt in enumerate(list(d.items())):
        dtype = np.promote_types(dtype, tt[1].dtype)

    # create an empty array with same datatype as first entry in dictionary
    v = np.empty(
        (len(f[1]), len(list(d.items()))),
        dtype=dtype
    )

    # put everything into a header list and the defined empty array
    for ii, tt in enumerate(list(d.items())):
        h.append(tt[0])
        v[:, ii] = tt[1]

    # define the header string
    header = (',').join(h)

    with open(p, 'w') as o:
        o.write(header + '\n')
        for ii in range(v.shape[0]):
            o.write(','.join(['%.16f' % num for num in v[ii, :]]))
            o.write('\n')


def csv_to_dict(p):
    """
        Read from CSV to a Dictionary

    This routine reads in a csv file and returns a
    dictionary with column headers as keys and columns
    as numpy arrays.

    Parameters
    ----------
    p : string
        path to read from

    Returns
    -------
    dict
        the content of the csv as a dictionary of numpy ndarrays
    """

    # first count the rows
    num_lines = sum(1 for line in open(p, 'r'))

    # read the dictionary entries
    arr_keys = []
    with open(p, 'r') as f:
        for ii, ll in enumerate(f):
            if ii == 0:
                arr_keys = ll.replace('\n', '').split(',')
                arr_vals = np.empty(
                    (num_lines - 1, len(arr_keys))
                )
            else:
                arr_line = ll.replace('\n', '').split(',')
                arr_line = list(map(lambda c: float(c), arr_line))
                arr_vals[ii-1, :] = arr_line

    # create the dictionary
    dct_res = {}
    for ii, kk in enumerate(arr_keys):
        dct_res.update({kk: arr_vals[:, ii]})

    return dct_res


class Decoder(json.JSONDecoder):
    '''
    This class can be used to correctly decode integers in json files
    to Python integers right upon parsing of the json file.

    >>> json.load('file.json', cls=camiba.data.Decoder)
    '''
    def decode(self, s):
        result = super().decode(s)
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


def json_to_tex(in_path, out_path, verbose=False):
    """
        Save Dictionary to TeX Makros

    This function takes a path to a JSON file and outputs a file of TeX
    makros to make the values in the JSON file available to TeX. It may
    be very useful if one would like to run simulations parametrized by an
    external file and automatically include the used values in tex.

    If one had a file config.json looking like:

    >>> {
    >>>     "measurementDataPath": "museData.mat",
    >>>     "measurementDataName": "data",
    >>>     "dictionaryDataPath": "dictData%(d).mat",
    >>>     "dictionaryDataName": "data",
    >>>     "dataShift": [0,0,0],
    >>>     "dataStride": [1,1,1],
    >>>     "recoParams": {
    >>>         "numK": 50
    >>>     },
    >>>     "outputPath": "../data/",
    >>>     "logPath": "reconstruct.log"
    >>> }

    then issuing

    >>> json_to_tex(config.json, config.tex)

    would result in the file config.tex looking like:

    .. code-block:: none

      \\newcommand{\\measurementDataPath}{museData.mat}
      \\newcommand{\\measurementDataName}{data}
      \\newcommand{\\dictionaryDataName}{data}
      \\newcommand{\\dataShift1}{0}
      \\newcommand{\\dataShift2}{0}
      \\newcommand{\\dataShift3}{0}
      \\newcommand{\\dataStride1}{1}
      \\newcommand{\\dataStride2}{1}
      \\newcommand{\\dataStride3}{1}
      \\newcommand{\\recoParamsnumK}{50}
      \\newcommand{\\outputPath}{../data/}
      \\newcommand{\\logPath}{reconstruct.log}

    Parameters
    ----------
    in_path : str
        path to the json file
    out_path : str
        path to the tex file that will be created
    """
    json_file = open(in_path)
    json_dct = json.load(json_file)
    json_file.close()

    lstMakros = _dict_to_tex(json_dct, "")

    out_file = open(out_path, 'w')
    for mm in lstMakros:
        if verbose:
            print(mm)
        out_file.write(mm)
    out_file.close()


def dict_to_tex(in_dict, out_path, verbose=False):
    """
        Save Dictionary to TeX Makros

    This function takes a path to a JSON file and outputs a file of TeX
    makros to make the values in the JSON file available to TeX. It may
    be very useful if one would like to run simulations parametrized by an
    external file and automatically include the used values in tex.

    If one had a dictionary looking like:

    >>> {
    >>>     "measurementDataPath": "museData.mat",
    >>>     "measurementDataName": "data",
    >>>     "dictionaryDataPath": "dictData%(d).mat",
    >>>     "dictionaryDataName": "data",
    >>>     "dataShift": [0,0,0],
    >>>     "dataStride": [1,1,1],
    >>>     "recoParams": {
    >>>         "numK": 50
    >>>     },
    >>>     "outputPath": "../data/",
    >>>     "logPath": "reconstruct.log"
    >>> }

    then issuing

    >>> dict_to_tex(dictionary.json, dictionary.tex)

    would result in the file config.tex looking like:

    .. code-block:: none

      \\newcommand{\\measurementDataPath}{museData.mat}
      \\newcommand{\\measurementDataName}{data}
      \\newcommand{\\dictionaryDataName}{data}
      \\newcommand{\\dataShift1}{0}
      \\newcommand{\\dataShift2}{0}
      \\newcommand{\\dataShift3}{0}
      \\newcommand{\\dataStride1}{1}
      \\newcommand{\\dataStride2}{1}
      \\newcommand{\\dataStride3}{1}
      \\newcommand{\\recoParamsnumK}{50}
      \\newcommand{\\outputPath}{../data/}
      \\newcommand{\\logPath}{reconstruct.log}

    Parameters
    ----------
    in_dict : dict
        the dictionary to export
    out_path : str
        path to the tex file that will be created
    """

    lstMakros = _dict_to_tex(in_dict, "")

    out_file = open(out_path, 'w')
    for mm in lstMakros:
        if verbose:
            print(mm)
        out_file.write(mm)
    out_file.close()


def _dict_to_tex(dct, in_string):
    res = []
    if in_string == "":
        in_string = '\\newcommand{\\'

    for kk in dct.keys():
        # if we encounter a dictionary, we simply extend the definition of
        # then corresponding makro
        if type(dct[kk]) == dict:
            res.append(*_dict_to_tex(dct[kk], in_string + kk))
        elif type(dct[kk]) == list:
            for ii, ll in enumerate(dct[kk]):
                res.append(
                    in_string + kk + str(ii + 1) + '}{' + str(ll)+'}'
                )
        else:
            # leave out anything that contains a %, since we get all sorts
            # of problems in tex, there
            if (str(dct[kk])+kk).find("%") == -1:
                res.append(in_string + kk + '}{' + str(dct[kk])+'}')

    return res


def save_params(dct_vars, str_p):
    """
        Save Dictionary of Parameters to Disk

    This routine allows to store names and their values in a file, which
    can be read from LaTeX to dynamically update parameter values of
    simulations in a paper

    Parameters
    ----------
    dct_vars : dict
        dictionary of parameter values
    str_p : string
        path to save the file to
    """

    # open the file to write
    with open(str_p, 'w') as f:

        # write the header
        f.write("name,value \n")

        for pp in dct_vars.items():
            if type(pp[1]) == float:
                f.write(pp[0] + "," + "{0:.8f}".format(pp[1])+'\n')
            else:
                lg = int(math.ceil(math.log10(pp[1])))
                s = "{0:0"+str(lg)+"d}"
                f.write(pp[0] + "," + s.format(pp[1])+'\n')
