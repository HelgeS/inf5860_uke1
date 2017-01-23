import numpy as np

def math1(mat):
  """
  Square each value in mat separatly
  """
  return mat**2


def math2(mat):
  """
  Return the sum of all the values in mat
  """
  return mat.sum()


def math3(mat):
  """
  Return the sum of each row in mat
  """
  return mat.sum(axis=0)


def math4(mat):
  """
  Return a scaled version of mat, so that it sums to 1
  """
  return mat/mat.sum()


def math5(mat, v):
  """
  Element-wise multiply each column in mat with vector v
  """
  return mat * v[:, np.newaxis]


def math6(mat, v):
  """
  Element-wise multiply each row in mat with vector v
  """
  return mat * v.transpose()


def math7(mat, v):
  """
  Matrix multiply matrix m with vector v
  """
  return mat.dot(v)


def math8(mat, v):
  """
  Element-wise multiply each column in mat with vector v
  """
  return mat * v


def math9(mat):
  """
  Return the inverse matrix of mat
  """
  return np.linalg.inv(mat)


def math10(mat, v):
  """
  Return the dot-product between each row in mat and v
  """
  return mat.dot(v)


def math11(mat, v):
  """
  Return the dot-product between each column in mat and v
  """
  return mat.transpose().dot(v)


def math12(mat1, mat2):
  """
  Return the dot-products between each corresponding (same index) column in mat1 and mat2
  """
  return (mat1*mat2).sum(1)
