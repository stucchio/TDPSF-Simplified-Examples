import os
from pylab import *
import numpy.fft as dft
import scipy.signal as signal
from utils import *
from tdpsf import *

freq = float(sys.argv[1])

simname = "euler_subsonic_"+str(freq)
simdir = simname+"_plots"


N = 3 #Number of components of the wavefield
npoints = 512
shape=(npoints, npoints)
dx = 0.125
arrow_stride = npoints/24
width=npoints*dx/2
dt = 0.25
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

sdev = 1.0
top   = phase_space_filter(dx,sdev,1e-4, npoints, 0, N, diagonalizing_pair)
bottom   = phase_space_filter(dx,sdev,1e-4, npoints, 0, N, diagonalizing_pair, 1)
print "Buff width: " + str(top.buffer_width)
left  = phase_space_filter(dx,sdev,1e-6, npoints, 1, N, diagonalizing_pair)
right = phase_space_filter(dx,sdev,1e-6, npoints, 1, N, diagonalizing_pair, 1)
kmin = log(1e-6)/sdev

print "Buffer widths: " + str(top.buffer_width) + "," + str(left.buffer_width)
buffwidth = top.buffer_width

kf = top.kgrid()
root_kf = sqrt(kf[0]**2+kf[1]**2)
top.P[:] = 0.0
top.P[0][where( M + kf[0]/root_kf > 0, True, False) ] = 1.0
top.P[1][where( M - kf[0]/root_kf > 0, True, False) ] = 1.0
top.P[2] = 1.0
top.blur_k_filter()

kf = bottom.kgrid()
root_kf = sqrt(kf[0]**2+kf[1]**2)
bottom.P[:] = 0.0
bottom.P[0][where( M + kf[0]/root_kf < 0, True, False) ] = 1.0
bottom.P[1][where( M - kf[0]/root_kf < 0, True, False) ] = 1.0
bottom.P[2] = 1.0
bottom.blur_k_filter()


kf = left.kgrid()
root_kf = sqrt(kf[0]**2+kf[1]**2)
left.P[:] = 0.0
left.P[0][where( 1*kf[1]/root_kf > 0, True, False) ] = 1.0
left.P[1][where( -1*kf[1]/root_kf > 0, True, False) ] = 1.0
left.P[2] = 0.0
left.blur_k_filter()

kf = right.kgrid()
root_kf = sqrt(kf[0]**2+kf[1]**2)
right.P[:] = 0.0
right.P[0][where( -1*kf[1]/root_kf > 0, True, False) ] = 1.0
right.P[1][where( 1*kf[1]/root_kf > 0, True, False) ] = 1.0
right.P[2] = 0.0
right.blur_k_filter()




hot() #Choose the color scheme
os.system("rm -rf " + simdir + "; mkdir " + simdir)

for i in range(int(tmax/dt)):
    time = i*dt

    if i % 8 == 0:
        print "Saving data file...t="+str(time)
        u[0,buffwidth:-buffwidth,buffwidth:-buffwidth].real.tofile(os.path.join(simdir, "frame" + str(int(i/8)).rjust(6,"0")+".dat"))

    print "Time = " + str(time)
    for n in range(N): #FFT the wavefield
        u[n] = dft.fft2(u[n])
    u[:] = mmul(D,u)
    u *= exp(H*dt)
    u[:] = mmul(DI,u)
    for n in range(N): #Inverse FFT the wavefield
        u[n] = dft.ifft2(u[n])
    if i % int(top.width()/(12.0*dt)) == 0: #Apply phase space filters before
        top(u)                           #waves can cross half the buffer
        left(u)
        right(u)
        bottom(u)
        print "Applying filter..."
    clf()#Now plot things
    imshow(u[0].real,extent=(width,-width,width+M*time,-width+M*time),vmin=-1.0,vmax=1.0)
    title("$p(x,"+str(time).ljust(5,'0')+")$")
    colorbar()
    #hold(True)
    #quiver(x[1,::arrow_stride,::arrow_stride],x[0,::arrow_stride,::arrow_stride]+M*time,u[2,::arrow_stride,::arrow_stride].real, u[1,::arrow_stride,::arrow_stride].real, scale=5)
    savefig(os.path.join(simdir, "frame" + str(i).rjust(6,"0")+".png"), dpi=120)


movie_command_line =  "mencoder -ovc lavc -lavcopts vcodec=mpeg1video:vbitrate=1500 -mf type=png:fps=15 -nosound -of mpeg -o " + simdir + "/simulation.mpg mf://" + simdir + "/\*.png"
print "Making movie with command line: "
print movie_command_line
os.system(movie_command_line)

mp4_movie_command_line = "ffmpeg -r 25 -b 1800 -i " + simdir + "/frame%06d.png " + simdir + "/simulation.mp4"
print mp4_movie_command_line
os.system(mp4_movie_command_line)

