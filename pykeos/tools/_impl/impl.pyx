cimport cython

import numpy as np
cimport numpy as np

BOOLTYPE = np.uint8
INTTYPE = np.int
INT8TYPE = np.int8
INT16TYPE = np.int16
INT32TYPE = np.int32
FLOATTYPE = np.float
FLOAT32TYPE = np.float32
FLOAT64TYPE = np.float64
ctypedef np.uint8_t BOOLTYPE_t
ctypedef np.int_t INTTYPE_t
ctypedef np.int8_t INT8TYPE_t
ctypedef np.int16_t INT16TYPE_t
ctypedef np.int32_t INT32TYPE_t
ctypedef np.float_t FLOATTYPE_t
ctypedef np.float32_t FLOAT32TYPE_t
ctypedef np.float64_t FLOAT64TYPE_t


def _localized_diagline_histogram(
    int n_time, np.ndarray[INT8TYPE_t, ndim=2] recmat):

    cdef int i, j, k, i_start, j_start = 0
    cdef list diagline_list = []
    for i in range(n_time):
        if k != 0:
            if k > 1:
                diagline_list.append((i_start, j_start, k))
            k = 0
        for j in range(i+1):
            if recmat[n_time-1-i+j, j] == 1:
                if k == 0:
                    i_start = i
                    j_start = j
                k += 1
            elif k != 0:
                if k > 1:
                    diagline_list.append((i_start, j_start, k))
                k = 0

    return diagline_list




def _localized_vertline_histogram(int n_time, np.ndarray[INT8TYPE_t, ndim=2] recmat):
    cdef int i, j, k, i_start, j_start = 0
    cdef list vertline_list = []

    for i in range(n_time):
        if k != 0:
            if k > 1:
                vertline_list.append((i_start, j_start, k))
            k = 0

        for j in range(n_time):
            if recmat[i, j] != 0:
                if k == 0:
                    i_start = i
                    j_start = j
                k += 1
            elif k != 0:
                if k > 1:
                    vertline_list.append((i_start, j_start, k))
                k = 0
    return vertline_list


def _localized_white_vertline_histogram(
    int n_time, np.ndarray[INT8TYPE_t, ndim=2] recmat):

    cdef int i, j, k, i_start, j_start = 0
    cdef list white_vertline_list = []

    for i in range(n_time):
        if k != 0:
            if k > 1:
                white_vertline_list.append((i_start, j_start, k))
            k = 0

        for j in range(n_time):
            if recmat[i, j] == 0:
                k += 1
                if k == 0:
                    i_start = i
                    j_start = j
            elif k != 0:
                if k > 1:
                    white_vertline_list.append((i_start, j_start, k))
                k = 0

    return white_vertline_list