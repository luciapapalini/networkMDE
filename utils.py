import numpy as np

def matrix_to_sparse(matrix):
    N = len(matrix)

    matrix = np.array(matrix).flatten()
    delete_indexes = (matrix == 0.)

    i = np.arange(N)
    j = np.arange(N)

    i = np.repeat(i, N)
    j = np.tile(j, N)

    # null values are deleted
    i = np.delete(i, delete_indexes)
    j = np.delete(j, delete_indexes)
    matrix = np.delete(matrix, delete_indexes)

    sparse = np.vstack((i,j,matrix)).transpose()

    return sparse
