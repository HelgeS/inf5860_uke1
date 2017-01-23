import numpy as np
import scipy.signal as sp
from matplotlib import pyplot as plt
import time

def main():
  img = plt.imread('lena.png')
  plt.imshow(img)

  start = time.time()
  out1 = sobel_filter(img)
  out2 = blur_filter(img)
  print('Calculation time: %.2f sec' % time.time()-start)
  plt.figure()
  plt.imshow(out1.mean(2), vmin=out1.min(), vmax=out1.max(), cmap='gray')
  plt.figure()
  plt.imshow(out2, vmin=out2.min(), vmax=out2.max())
  plt.show()


def convolution(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  """
  data = np.zeros(shape=image.shape)
  kernel = kernel[::-1, ::-1]

  ir, ic, irgb = image.shape
  kr, kc = kernel.shape
  kr_center = kr // 2
  kc_center = kc // 2

  for r in range(kr_center, ir - kr_center):
      for c in range(kc_center, ic - kc_center):
          for rgb in range(irgb):
              data[r, c, rgb] = np.sum((image[r - kr_center:r+kr_center+1, c-kc_center:c+kc_center+1, rgb]*kernel))

  return data

def convolution2(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  """
  kCenterX = kernel.shape[1] / 2;
  kCenterY = kernel.shape[0] / 2;

  out = np.zeros(shape=image.shape)

  for r in range(image.shape[0]):
      for c in range(image.shape[1]):
          for kr in range(kernel.shape[0]):
              mm = kernel.shape[0] - 1 - kr

              for kc in range(kernel.shape[1]):
                  nn = kernel.shape[1] - 1 - kc

                  ii = r + (kernel.shape[0] - kCenterY)
                  jj = c + (kernel.shape[1] - kCenterX)
                  
                  if ii >= 0 and ii < image.shape[0] and jj >= 0 and jj < image.shape[1]:
                      out[r][c] += image[ii][jj] * kernel[mm][nn]

  return out


def blur_filter(img):
  """
  Use your convolution function to filter your image with an average filter (box filter)
  with kernal size of 11.
  """
  k_size = 11
  kernel = np.ones((k_size, k_size))/k_size
  return convolution(img, kernel)


def sobel_filter(img):
  """
  Use your convolution function to filter your image with a sobel operator
  """
  x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#  g_x = convolution(img, x_kernel)
  g_y = convolution(img, y_kernel)
#  g = np.sqrt((g_x**2+g_y**2))
  return g_y


if __name__ == '__main__':
  main()
