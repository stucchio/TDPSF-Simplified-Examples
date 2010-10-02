from pylab import *
import numpy.fft as dft
import scipy.signal as signal
from utils import *


class phase_space_filter:
    def __init__(self,dx,sigma,tol, npoints, side, N, dfunc, top_or_bottom=0):
        self.dx = dx
        self.dk = 2*math.pi/(dx*npoints)
        self.side = side
        self.tol = tol
        self.top_or_bottom = top_or_bottom
        self.trans_width = sigma*sqrt(-1*log(tol)+2.0*abs(log(2.0*sigma/sqrt(math.pi))))
        self.trans_width_lattice = int(self.trans_width/self.dx)
        self.sigma = sigma
        self.buffer_width = next_pow_2(3*self.trans_width_lattice)
        self.kbuff = log(tol)/sigma
        if self.side==0:
            self.shape = (2*self.buffer_width, npoints)
            self.chi = empty(shape=self.shape, dtype=Complex32)
            self.chi[:] = 0.0
            self.chi[self.trans_width_lattice:self.buffer_width-1*self.trans_width_lattice, self.trans_width_lattice:-1*self.trans_width_lattice] = 1.0
        if self.side==1:
            self.shape = (npoints, 2*self.buffer_width)
            self.chi = empty(shape=self.shape, dtype=Complex32)
            self.chi[:] = 0.0
            self.chi[self.trans_width_lattice:-1*self.trans_width_lattice, self.trans_width_lattice:self.buffer_width-1*self.trans_width_lattice] = 1.0
        self.chi = blur_image(self.chi, self.sigma, self.dx) #This is the x window

        k = kgrid(self.shape, dx)
        self.P = empty(shape=(N,) + self.shape, dtype=Complex32)
        self.N = N
        self.P[:] = 1.0

        self.buff = empty(shape=(N,) + self.shape, dtype=Complex32)
        self.D, self.DI = dfunc(k)

    def blur_k_filter(self):
        dk = [2*math.pi/(self.dx*self.P[0].shape[i]) for i in range(2)]
        for i in range(self.N):
            self.P[i] = blur_image(self.P[i], 1.0/self.sigma, dk)

    def width(self):
        return self.buffer_width*self.dx

    def kgrid(self):
        return kgrid(self.shape, self.dx)

    def __get_data(self,pos):
        if self.top_or_bottom == 0 and self.side == 0:
            return pos[:,0:self.buffer_width, :]
        if self.top_or_bottom == 1 and self.side == 0:
            return pos[:,-1*self.buffer_width:, :]
        if self.top_or_bottom == 0 and self.side == 1:
            return pos[:,:, 0:self.buffer_width]
        if self.top_or_bottom == 1 and self.side == 1:
            return pos[:,:, -1*self.buffer_width:]


    def __call__(self,pos):
        for i in range(self.N):
            if self.side == 0:
                self.buff[i,:] = 0
                self.buff[i,0:self.buffer_width,:] = self.__get_data(pos)[i]
                self.buff[i,:]*self.chi
                self.buff[i,:] = dft.fft2(self.buff[i,:])
            if self.side == 1:
                self.buff[i,:] = 0
                self.buff[i,:,0:self.buffer_width] = self.__get_data(pos)[i]
                self.buff[i,:]*self.chi
                self.buff[i,:] = dft.fft2(self.buff[i,:])

        self.buff = mmul(self.D, self.buff)
        self.buff *= self.P
        self.buff = mmul(self.DI, self.buff)
        for i in range(self.N):
            self.buff[i,:] = dft.ifft2(self.buff[i,:])
        self.buff *= self.chi
        if self.side == 0:
            self.__get_data(pos)[:] -= self.buff[:,0:self.buffer_width,:]
        if self.side == 1:
            self.__get_data(pos)[:] -= self.buff[:,:,0:self.buffer_width]
