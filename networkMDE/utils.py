import numpy as np
import warnings


def compact_indexes(sparseM):
    # again: definitely not the most elegant way to do it
    # but it's late night and I am tired (ah ah, just an excuse)
    sparseM = np.array(sparseM)
    indexes = sparseM.transpose()[:2].flatten()
    indexes = set(indexes)
    indexes = np.sort(list(indexes))
    translate = dict([couple for couple in zip(indexes, np.arange(len(indexes)))])
    result = np.zeros(sparseM.shape)

    for row_index in range(len(sparseM)):
        result[row_index] = [
            translate[sparseM[row_index, 0]],
            translate[sparseM[row_index, 1]],
            sparseM[row_index, 2],
        ]

    return result


def matrix_to_sparse(matrix):
    N = len(matrix)

    matrix = np.array(matrix).flatten()
    delete_indexes = matrix == 0.0

    i = np.arange(N)
    j = np.arange(N)

    i = np.repeat(i, N)
    j = np.tile(j, N)

    # null values are deleted
    i = np.delete(i, delete_indexes)
    j = np.delete(j, delete_indexes)
    matrix = np.delete(matrix, delete_indexes)

    sparse = np.vstack((i, j, matrix)).transpose()
    sparse = compact_indexes(sparse)

    return sparse
