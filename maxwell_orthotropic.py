import os
from pylab import *
import numpy.fft as dft
import scipy.signal as signal
from utils import *
from tdpsf import *

freq = float(sys.argv[1])

simname = "maxwell_orthotropic"+str(freq)
simdir = simname+"_plots"

#This simulation solves the u[3,4,5] parts of the wavefield.
N = 3 #Number of components of the wavefield
npoints = 512
shape=(npoints, npoints)
dx = 0.125
arrow_stride = npoints/24
width=npoints*dx/2
dt = 0.25
tmax = 100
x = indices(shape)*dx
x[0] -= npoints*dx/2
x[1] -= npoints*dx/2

b = 0.25 #Anisotropy parameter
f = (sqrt(1.0+b)+sqrt(1.0-b))/2.0
g = (-1*sqrt(1.0+b)+sqrt(1.0-b))/2.0

def diagonalizing_pair(k): #Returns the diagonalizing pair, given a frequency coordinate array
    shape = k.shape[1::]
    E = sqrt( (f*f+g*g)*(k[0]*k[0]+k[1]*k[1])-4*f*g*k[0]*k[1])
    E[0,0] = 1.0
    Er2 = sqrt(2.0)*E #E root 2
    D = empty(shape=(N,N) + shape, dtype=Complex64) #Diagonalizes the Hamiltonian
    D[0,0] = -1.0 / sqrt(2.0)# Er2
    D[0,1] = -1.0*(f*k[1]-g*k[0]) / Er2
    D[0,2] = (f*k[0]-g*k[1]) / Er2
    D[1,0] = 1.0/ sqrt(2.0) # Er2
    D[1,1] = -1.0*(f*k[1]-g*k[0]) / Er2
    D[1,2] = (f*k[0]-g*k[1]) / Er2
    D[2,0] = 0
    D[2,1] = (f*k[0]-g*k[1]) / E
    D[2,2] = (f*k[1]-g*k[0]) / E

    DI = empty(shape=(N,N) + shape, dtype=Complex64) #Diagonalizes the Hamiltonian
    DI[0,0] = -1.0 / sqrt(2.0) #Er2
    DI[1,0] = -1.0*(f*k[1]-g*k[0]) / Er2
    DI[2,0] = (f*k[0]-g*k[1]) / Er2
    DI[0,1] = 1.0/ sqrt(2.0) #Er2
    DI[1,1] = -1.0*(f*k[1]-g*k[0]) / Er2
    DI[2,1] = (f*k[0]-g*k[1]) / Er2
    DI[0,2] = 0
    DI[1,2] = (f*k[0]-g*k[1]) / E
    DI[2,2] = (f*k[1]-g*k[0]) / E
    return D, DI

print "Running a simulation of the maxwell equations in an orthotropic medium."
print "b = " + str(b)
print "f = " + str(f)
print "g = " + str(g)
print "Lattice of " + str(shape) + ", with dx=" + str(dx) + ", dt="+str(dt)

#root_k[0,0] = 1.0

k = kgrid((npoints,npoints), dx)
E = sqrt( (f**2+g**2)*(k[0]*k[0]+k[1]*k[1])-4*f*g*k[0]*k[1])


u = empty(shape=(N,) + shape, dtype=Complex64) #The wavefield itself
u[:] = 0
u[0] = exp((-1.0/9.0)*((x[0])**2+x[1]**2))*cos(freq*sqrt((x[0])**2+x[1]**2+1))*((x[0])**2+x[1]**2)

D, DI = diagonalizing_pair(k)

H = empty(shape=(N,) + shape, dtype=Complex64) #Hamiltonian
H[0,:] = complex(0,1)*E
H[1,:] = complex(0,-1)*E
H[2,:] = 0.0
#H[:,0,0] = 0.0

sdev = 1.0
top    = phase_space_filter(dx,sdev,1e-4, npoints, 0, N, diagonalizing_pair)
bottom = phase_space_filter(dx,sdev,1e-4, npoints, 0, N, diagonalizing_pair, 1)
print "Buff width: " + str(top.buffer_width)
left  = phase_space_filter(dx,sdev,1e-4, npoints, 1, N, diagonalizing_pair)
right = phase_space_filter(dx,sdev,1e-4, npoints, 1, N, diagonalizing_pair, 1)
kmin = log(1e-6)/sdev

print "Buffer widths: " + str(top.buffer_width) + "," + str(left.buffer_width)
buffwidth = top.buffer_width

kf = top.kgrid()
top.P[:] = 0.0
top.P[0][where( 2*(f*f+g*g)*kf[0] -4*f*g*kf[1] > 0, True, False) ] = 1.0
top.P[1][where( 2*(f*f+g*g)*kf[0] -4*f*g*kf[1] < 0, True, False) ] = 1.0
top.P[2] = 1.0
top.blur_k_filter()

kf = left.kgrid()
left.P[:] = 0.0
left.P[0][where( 2*(f*f+g*g)*kf[1] -4*f*g*kf[0] > 0, True, False) ] = 1.0
left.P[1][where( 2*(f*f+g*g)*kf[1] -4*f*g*kf[0] < 0, True, False) ] = 1.0
left.P[2] = 0.0
left.blur_k_filter()

kf = right.kgrid()
right.P[:] = 0.0
right.P[0][where( 2*(f*f+g*g)*kf[1] -4*f*g*kf[0] < 0, True, False) ] = 1.0
right.P[1][where( 2*(f*f+g*g)*kf[1] -4*f*g*kf[0] > 0, True, False) ] = 1.0
right.P[2] = 0.0
right.blur_k_filter()

kf = bottom.kgrid()
bottom.P[:] = 0.0
bottom.P[0][where( 2*(f*f+g*g)*kf[0] -4*f*g*kf[1] < 0, True, False) ] = 1.0
bottom.P[1][where( 2*(f*f+g*g)*kf[0] -4*f*g*kf[1] > 0, True, False) ] = 1.0
bottom.P[2] = 1.0
bottom.blur_k_filter()



hsv() #Choose the color scheme
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
    u[2,:] = 0.0
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
    imshow(u[0].real,extent=(width,-width,width,-width), vmin=-1,vmax=1)
    title("$B_z (x,"+str(time).ljust(5,'0')+")$")
    colorbar()
    #quiver(x[1,::arrow_stride,::arrow_stride],x[0,::arrow_stride,::arrow_stride]+M*time,u[2,::arrow_stride,::arrow_stride].real, u[1,::arrow_stride,::arrow_stride].real, scale=5)
    savefig(os.path.join(simdir, "frame" + str(i).rjust(6,"0")+".png"), dpi=120)


movie_command_line =  "mencoder -ovc lavc -lavcopts vcodec=mpeg1video:vbitrate=1500 -mf type=png:fps=15 -nosound -of mpeg -o " + simdir + "/simulation.mpg mf://" + simdir + "/\*.png"
print "Making movie with command line: "
print movie_command_line
os.system(movie_command_line)

mp4_movie_command_line = "ffmpeg -qmax 2 -r 25 -b 1800 -i " + simdir + "/frame%06d.png " + simdir + "/simulation.mp4"
print mp4_movie_command_line
os.system(mp4_movie_command_line)

