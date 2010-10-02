from math import *
import os
from pylab import *
from scipy.special import *
import numpy.fft as dft
import scipy.signal as signal


npoints = 1024
npointsl = 1024*8
dx = 0.1

def xg(np,d_x):
    return (arange(np,dtype=float)-np/2.0)*d_x

def kg(np, d_x):
    k = arange(np,dtype=float)
    k[np/2:] -= np
    k *= (2*pi/(d_x*np))
    return k

x = xg(npoints,dx)
k = kg(npoints,dx)
xl = xg(npointsl,dx)
kl = kg(npointsl,dx)

kmax = pi/dx
L = npoints*dx/2.0

psi = zeros(npoints, dtype=complex)

psil = zeros(npointsl, dtype=complex)

print "X domain: [" + str(x[0]) + ", " + str(x[-1]) + "]"
print "K domain: [" + str(-kmax) + ", " + str(kmax) + "]"

def erfinv(x):
    return sqrt(log(1.0/x)+log(pi)/2-log(2))

class tdpsf:
    def __init__(self,npoints,dx, tol=1e-6):
        self.npoints = npoints
        self.dx = dx
        L = npoints*dx/2.0
        x = xg(npoints/4,dx)
        k = kg(npoints/4,dx)
        kmax = pi/dx
        bx = erfinv(tol)#sqrt(log(1.0/tol))
        if bx > L/8:
            print "Error, bx="+str(bx) + " > L="+str(L)
        self.chi = 0.5*(erf((x+bx)/sqrt(2.0))-erf((x-bx)/sqrt(2.0)))
        bk = kmax/2.0-1.5*erfinv(tol/L)
        if bk < kmax/4:
            print "Error, bk="+str(bk) + " < kmax/4="+str(kmax/4)
        bk = 0
        self.P = 0.5*erfc((bk-k)/2)

    def __call__(self,grid):
        tmp = self.chi*grid[3*self.npoints/4:]
        tmp[:] = dft.fft(tmp)
        tmp *= self.P
        tmp = self.chi*dft.ifft(tmp)
        grid[3*self.npoints/4:] -= tmp

def exact(x,v,t,sigma):
    i = complex(0,1)
    return (exp(i*v*(x-v*t/2.0))/(2.0*sqrt(sigma)*(1.0+i*t/(sigma*sigma))))*exp(-1*((x-v*t)**2)/(2.0*(sigma*sigma)*(1+i*t/(sigma*sigma))))

def error_test(vel,epsilon):
    sigma = 7.0
    tmax = 4*L/vel
    dt = ((L/2)/kmax)/18
    psi[:] = exact(x,vel,0.0,sigma)#exp(-1.0*(x*x/16.0)+complex(0,vel)*x)
    psil[:] = exact(xl,vel,0.0,sigma)
    m0 = sqrt(abs(psi[npoints/4:-npoints/4]**2).sum())

    prop = exp(complex(0,-dt/2.0)*k*k)
    propl = exp(complex(0,-dt/2.0)*kl*kl)
    fil = tdpsf(npoints,dx,epsilon)

    nsteps = int(tmax/dt)+1
    err = 0.0
    for i in range(nsteps):
        psi[:] = dft.ifft(prop*dft.fft(psi)) #Calculate TDPSF approximation
        fil(psi)

        psil[:] = dft.ifft(propl*dft.fft(psil)) #Calculate large box solution

        rem = abs(psil[npointsl/2-npoints/4:npointsl/2+npoints/4] - psi[npoints/2-npoints/4:npoints/2+npoints/4])

        local_err = sqrt((rem*rem).sum())/m0
        if local_err > err:
            err = local_err
            #print str(v) +", " + str(i*dt) + ", " + str(err)
        #print "T="+str(i*dt) + ", M = " + str(sqrt(abs(psi*psi).sum()*dx)/m0)
    print "k = " + str(vel) + ", epsilon = " + str(epsilon) + ", err = " + str(err)
    return err

if __name__=="__main__":
    velocities = arange(25,1,-1)
    eps_powers = [2,3,4,5, 6]
    err = [[error_test(float(v),pow(10,-eps_pow)) for v in velocities] for eps_pow in eps_powers]
    clf()
    for e in enumerate(eps_powers):
        semilogy(velocities,err[e[0]],label="$\delta=10^{-" + str(e[1]) + "}$")
    xlabel("Frequency ($k$)")
    ylabel("Relative Error in $L^2$")
    title("Errors for the Schrodinger Equation")
##     axvline(x=kmax/2.0, label="$k_{max}/2$")
##     axvline(x=kmax/4.0, label="$k_{max}/4$")
    legend()
    savefig("schrodinger_error.eps")
    legend()

