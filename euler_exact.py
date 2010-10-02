import os
from pylab import *
import numpy.fft as dft
import scipy.signal as signal
from utils import *
from tdpsf import *

freq = float(sys.argv[1])

simname = "euler_subsonic_exact_" + str(freq)
simdir = simname+"_plots"


smallwidth = 128

N = 3 #Number of components of the wavefield
npoints = 2048
shape=(npoints, npoints)
dx = 0.125
arrow_stride = npoints/24
width=npoints*dx/2
dt = 2.0
tmax = 50
M = 0.5 #Mach number
x = indices(shape)*dx
x[0] -= npoints*dx/2
x[1] -= npoints*dx/2

def diagonalizing_pair(k): #Returns the diagonalizing pair, given a frequency coordinate array
    shape = k.shape[1::]
    root_k = sqrt(k[0]*k[0]+k[1]*k[1])
    D = empty(shape=(N,N) + shape, dtype=Complex64) #Diagonalizes the Hamiltonian
    D[0,0] = -1*root_k / k[1]
    D[0,1] = k[0]/k[1]
    D[0,2] = 1
    D[1,0] = root_k / k[1]
    D[1,1] = k[0]/k[1]
    D[1,2] = 1
    D[2,0] = 0
    D[2,1] = -k[1]/k[0]
    D[2,2] = 1

    DI = empty(shape=(N,N) + shape, dtype=Complex64) #Diagonalizes the Hamiltonian
    DI[0,0] = -0.5*k[1]/root_k
    DI[0,1] = 0.5*k[1]/root_k
    DI[0,2] = 0
    DI[1,0] = 0.5*k[1]*k[0]/(root_k*root_k)
    DI[1,1] = 0.5*k[1]*k[0]/(root_k*root_k)
    DI[1,2] = -1*k[1]*k[0]/(root_k*root_k)
    DI[2,0] = 0.5*k[1]*k[1]/(root_k*root_k)
    DI[2,1] = 0.5*k[1]*k[1]/(root_k*root_k)
    DI[2,2] = 1*k[0]*k[0]/(root_k*root_k)
    return D, DI

print "Running a simulation of the euler equations in a subsonic flow."
print "Mach number: " + str(M)
print "Lattice of " + str(shape) + ", with dx=" + str(dx) + ", dt="+str(dt)

#root_k[0,0] = 1.0

k = kgrid((npoints,npoints), dx)
root_k = sqrt(k[0]*k[0]+k[1]*k[1])

u = empty(shape=(N,) + shape, dtype=Complex64) #The wavefield itself
u[:] = 0
u[0] = exp((-1.0/9.0)*((x[0]+8)**2+x[1]**2))*cos(freq*sqrt((x[0]+8)**2+x[1]**2+1))*((x[0]+8)**2+x[1]**2)

D, DI = diagonalizing_pair(k)

H = empty(shape=(N,) + shape, dtype=Complex64) #Hamiltonian
H[0,:] = complex(0,1)*(M*k[0]+root_k)
H[1,:] = complex(0,1)*(M*k[0]-root_k)
H[2,:] = complex(0,1)*M*k[0]
H[:,0,0] = 0.0

os.system("rm -rf " + simdir + "; mkdir " + simdir)

for i in range(int(tmax/dt)):
    u[0,npoints/2-smallwidth:npoints/2+smallwidth,npoints/2-smallwidth:npoints/2+smallwidth].real.tofile(os.path.join(simdir, "frame" + str(i).rjust(6,"0")+".dat"))
    time = i*dt
    print "Time = " + str(time)
    for n in range(N): #FFT the wavefield
        u[n] = dft.fft2(u[n])
    u[:] = mmul(D,u)
    u *= exp(H*dt)
    u[:] = mmul(DI,u)
    for n in range(N): #Inverse FFT the wavefield
        u[n] = dft.ifft2(u[n])



