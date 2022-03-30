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



def _weight_function(int di, int dj, int scale, int order):
    cdef float dist = (abs(di) + abs(dj)) / scale
    if dist < 1:
        return np.exp(1. - 1. / (1 - dist ** order))
    else:
        return 0.


def _localized_diagline_histogram(
    int n_time, np.ndarray[INT8TYPE_t, ndim=2] recmat):

    cdef int i, j, k
    cdef list diagline_list = []
    cdef np.ndarray[INT32TYPE_t, ndim=1] start_index = np.zeros([n_time - 1], dtype=np.int32)
    cdef np.ndarray[INT32TYPE_t, ndim=1] diag_length = np.zeros([n_time - 1], dtype=np.int32)

    i, j, k = 0, 0, 0

    # Iterate over rows
    for i in range(n_time):
        # Iterate over diagonals 
        for k in range(0, n_time-i-1):
            # If (i, i + diag) is recurrent
            if recmat[i, i + k + 1] == 1:
                # If the diag was not a 1-diagonal, change the start index
                if diag_length[k] == 0: 
                    start_index[k] = i

                diag_length[k] += 1
            # Else if (i, i + k) is not recurrent and we were in a 1-diagonal
            elif diag_length[k] > 0:
                # Store diagonal 
                diagline_list.append((start_index[k], start_index[k] + k, diag_length[k])) 
                diag_length[k] = 0

    return diagline_list    


def _localized_vertline_histogram(int n_time, np.ndarray[INT8TYPE_t, ndim=2] recmat):
    cdef int i, j, k, i_start, j_start
    i, j, k, i_start, j_start = 0, 0, 0, 0, 0
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

    cdef int i, j, k, i_start, j_start
    i, j, k, i_start, j_start = 0, 0, 0, 0, 0
    cdef list white_vertline_list = []

    for i in range(n_time):
        if k != 0:
            if k > 1:
                white_vertline_list.append((i_start, j_start, k))
            k = 0

        for j in range(n_time):
            if recmat[i, j] == 0 and i != j:
                if k == 0:
                    i_start = i
                    j_start = j
                k += 1
            elif k != 0:
                if k > 1:
                    white_vertline_list.append((i_start, j_start, k))
                k = 0

    return white_vertline_list



def _weighted_diagline_histogram(
    int n_time, list diaglines, np.ndarray[FLOATTYPE_t, ndim=2] diagline_series,int scale,  int order):

    cdef int i, k, i_start, j_start, i_mid, j_mid, di, dj
    i, k, i_start, j_start, i_mid, j_mid, di, dj = 0, 0, 0, 0, 0, 0, 0, 0

    cdef np.ndarray[FLOATTYPE_t, ndim=2] weights = np.zeros([2 * scale + 1, 2 * scale + 1], dtype=np.float)
    cdef float w = 0

    # Compute the weight matrix on its finite support [-weight, weight]
    weights[scale, scale] = _weight_function(0, 0, scale, order)
    for i in range(scale): 
        for j in range(scale):
            w = _weight_function(i + 1, j + 1, scale, order)

            weights[scale + i + 1, scale + j + 1] = w
            weights[scale + i + 1, scale - j - 1] = w
            weights[scale - i - 1, scale + j + 1] = w
            weights[scale - i - 1, scale - j - 1] = w

    # Iterate over diagonal
    for (i_start, j_start, k) in diaglines:
        # Find the center of the diagonal
        i_mid = i_start + int(round(k / 2.))
        j_mid = j_start + int(round(k / 2.))

        # Iterate over affected points
        for i in range( max(i_mid - scale, j_mid - scale, 0) , min(i_mid + scale, j_mid + scale, n_time)):
            di = i - i_mid
            dj = i - j_mid

            if abs(di) + abs(dj) < scale:
                # Count the weighted diagonal for timeseries i
                diagline_series[i , k] += weights[scale + di, scale + dj]


def _weighted_vertline_histogram(
    int n_time, list vertlines, np.ndarray[FLOATTYPE_t, ndim=2] vertline_series, int scale,  int order):

    cdef int i, k, i_start, j_start, i_mid, j_mid, di, dj
    i, k, i_start, j_start, i_mid, j_mid, di, dj = 0, 0, 0, 0, 0, 0, 0, 0

    cdef np.ndarray[FLOATTYPE_t, ndim=2] weights = np.zeros([2 * scale + 1, 2 * scale + 1], dtype=np.float)
    cdef float w = 0

    # Compute the weight matrix on its finite support [-weight, weight]
    weights[scale, scale] = _weight_function(0, 0, scale, order)
    for i in range(scale): 
        for j in range(scale):
            w = _weight_function(i + 1, j + 1, scale, order)

            weights[scale + i + 1, scale + j + 1] = w
            weights[scale + i + 1, scale - j - 1] = w
            weights[scale - i - 1, scale + j + 1] = w
            weights[scale - i - 1, scale - j - 1] = w

    # Iterate over vertical
    for (i_start, j_start, k) in vertlines:
        # Find the center of the vertical
        i_mid = i_start
        j_mid = j_start + int(round(k / 2.))

        # Iterate over affected points
        for i in range( max(i_mid - scale, j_mid - scale, 0) , min(i_mid + scale, j_mid + scale, n_time)):
            di = i - i_mid
            dj = i - j_mid

            if abs(di) + abs(dj) < scale:
                # Count the weighted vertical for timeseries i
                vertline_series[i , k] += weights[scale + di, scale + dj]


def _weighted_white_vertline_histogram(
    int n_time, list white_vertlines, np.ndarray[FLOATTYPE_t, ndim=2] white_vertline_series, int scale,  int order):

    cdef int i, k, i_start, j_start, i_mid, j_mid, di, dj
    i, k, i_start, j_start, i_mid, j_mid, di, dj = 0, 0, 0, 0, 0, 0, 0, 0

    cdef np.ndarray[FLOATTYPE_t, ndim=2] weights = np.zeros([2 * scale + 1, 2 * scale + 1], dtype=np.float)
    cdef float w = 0

    # Compute the weight matrix on its finite support [-weight, weight]
    weights[scale, scale] = _weight_function(0, 0, scale, order)
    for i in range(scale): 
        for j in range(scale):
            w = _weight_function(i + 1, j + 1, scale, order)

            weights[scale + i + 1, scale + j + 1] = w
            weights[scale + i + 1, scale - j - 1] = w
            weights[scale - i - 1, scale + j + 1] = w
            weights[scale - i - 1, scale - j - 1] = w

    # Iterate over vertical
    for (i_start, j_start, k) in white_vertlines:
        # Find the center of the vertical
        i_mid = i_start
        j_mid = j_start + int(round(k / 2.))

        # Iterate over affected points
        for i in range( max(i_mid - scale, j_mid - scale, 0) , min(i_mid + scale, j_mid + scale, n_time)):
            di = i - i_mid
            dj = i - j_mid

            if abs(di) + abs(dj) < scale:
                # Count the weighted vertical for timeseries i
                white_vertline_series[i , k] += weights[scale + di, scale + dj]