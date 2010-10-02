import os
from pylab import *
import numpy.fft as dft
from utils import *

simname = "wave"
simdir = simname+"_plots"

npoints = 256
shape=(npoints, npoints)
dx = 0.125
dt = 0.25
tmax = 24
x = indices(shape)*dx
x[0] -= npoints*dx/2
x[1] -= npoints*dx/2

dk = 2*math.pi/(dx*npoints)
kmax = math.pi/dx
k = kgrid(shape,dx)

root_k = complex(0,1)*sqrt(k[0]*k[0]+k[1]*k[1])
root_k[0,0] = complex(0,1)

N = 2 #Number of components of the wavefield

u = empty(shape=(N,) + shape, dtype=Complex32) #The wavefield itself
u[:] = 0
u[0] = exp(-1*(x[0]*x[0]+x[1]*x[1])/3.0)*exp((x[0]+x[1])*complex(0,-0.5)*math.pi/(dx*npoints))

D = empty(shape=(N,N) + shape, dtype=Complex32) #Diagonalizes the Hamiltonian
D[0,0] = 1
D[0,1] = 1/root_k
D[1,0] = 1
D[1,1] = -1/root_k

H = empty(shape=(N,) + shape, dtype=Complex32) #Hamiltonian
H[0,:] = root_k
H[1,:] = -1*root_k
H[:,0,0] = 0.0

DI = empty(shape=(N,N) + shape, dtype=Complex32) #Diagonalizes the Hamiltonian
DI[0,0] = 0.5
DI[0,1] = 0.5
DI[1,0] = 0.5*root_k
DI[1,1] = -0.5*root_k

def mmul(A,x):
    dim = x.shape[0]
    result = x.copy()
    for i in range(dim):
        result[i] = (A[i]*x).sum(axis=0)
    return result

hot() #Choose the color scheme
os.system("rm -rf " + simdir + "; mkdir " + simdir)
for i in range(int(tmax/dt)):
    time = i*dt
    print "Time = " + str(time)
    for n in range(N): #FFT the wavefield
        u[n] = dft.fft2(u[n])
    u[:] = mmul(D,u)
    u *= exp(H*dt)
    u[:] = mmul(DI,u)
    for n in range(N): #Inverse FFT the wavefield
        u[n] = dft.ifft2(u[n])

    clf()#Now plot things
    subplot(221)
    imshow(u[0].real)
    title("$u_0(x,"+str(time)+")$")
    colorbar()
    subplot(222)
    imshow(u[1].real)
    title("$u_1(x,"+str(time)+")$")
    colorbar()
    savefig(os.path.join(simdir, "frame" + str(i).rjust(6,"0")+".png"))

movie_command_line = "mencoder -ovc lavc -lavcopts vcodec=mpeg1video:vbitrate=1500 -mf type=png:fps=12 -nosound -of mpeg -o " + simdir + "/simulation.mpg mf://" + simdir + "/\*.png"
print "Making movie with command line: "
print movie_command_line
os.system(movie_command_line)

