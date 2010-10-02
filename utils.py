from pylab import *
import scipy.signal as signal
import numpy.fft as dft
import math

def next_pow_2(n):
    i = 0
    while pow(2,i) < n:
        i += 1
    return pow(2,i)

def mmul(A,x):
    dim = x.shape[0]
    result = x.copy()
    for i in range(dim):
        result[i] = (A[i]*x).sum(axis=0)
    return result

def kgrid(shape, dx): #Returns a kgrid, given a shape and a lattice spacing
    if type(dx) is tuple or type(dx) is list:
        dk = [ 2*math.pi/(dx[i]*shape[i]) for i in range(2)]
    else:
        dk = [ 2*math.pi/(dx*shape[i]) for i in range(2)]
    k = indices(shape,dtype=Float32) + 0.000001
    for i in range(2):
        k[i,:] *= dk[i]
        k[i,where(k[i,:] > dk[i]*(shape[i]-1)/2,True,False)] -= shape[i]*dk[i]
    return k

def blur_image(im, sigma, dx):
    """Blurs the image in the Fourier domain."""
    kim = dft.fft2(im)
    kx, ky = kgrid(kim.shape, dx)
    kim *= exp(-1*(kx**2+ky**2)/(sigma*sigma))
    xim = dft.ifft2(kim)
    if abs(xim).max() == 0:
        xim = 0
    else:
        xim /= xim.max()
    return xim


