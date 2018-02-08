import numpy

a = [1, 2, 3]
b = numpy.zeros((4,3))
b[0,:] = a
print(b)
