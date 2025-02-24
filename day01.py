import numpy as np

v = np.array([1, 3, -9, 2])
print(v, v.shape)
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b, b.shape)
c = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
print(c, c.shape)
print(c.strides)
