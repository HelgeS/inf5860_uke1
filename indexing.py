import numpy as np
from matplotlib import pyplot as plt


def index1(mat):
    """
    Return second row
    """
    return mat[1,:]


def index2(mat):
    """
    Return second column
    """
    return mat[:,1]


def index3(mat):
    """
    Return 2x2 square, starting at index (1, 1)
    """
    return mat[1:3,1:3]


def index4(mat):
    """
    Return second and forth column
    """
    return mat[:,[1,3]]


def index5(mat):
    """
    Return second and forth column, but in reversed order
    """
    return mat[:,[3,1]]


def index6(mat):
    """
    Return every other row in mat. Row with index 0, 2, 4, 6, 8...
    """
    return mat[0::2,:]


def index7(mat):
    """
    Return true where mat-values are higher than 3
    """
    return mat > 3


def index8(mat):
    """
    Return array of all values higher than 3
    """
    return mat[mat > 3]


def index9(mat):
    """
    Return the number of values higher than 3
    """
    return np.count_nonzero(mat > 3)


def index10(cow_names, ages):
    """
    Return an array of cow names, sorted by age, from youngest to oldest.
    Cow_names and ages are two numpy arrays, corresponding to cow names and age.
    """
    return cow_names[np.argsort(ages)]

if __name__ == '__main__':
    index1(np.arange(16).reshape((4,4)))
