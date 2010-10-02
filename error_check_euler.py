from pylab import *
from numpy import *

def sixdig(n):
    return str(n).rjust(6,"0")

file_prefix = "euler_subsonic_"
file_prefix_exact = "euler_subsonic_exact_"

def error_from_sim(t,chirp):
    u0 = fromfile(file_prefix+str(float(chirp))+"_plots/frame000000.dat", dtype=float64)
    u = fromfile(file_prefix+str(float(chirp))+"_plots/frame" + sixdig(t)+".dat", dtype=float64)
    ue = fromfile(file_prefix_exact+str(float(chirp))+"_plots/frame" + sixdig(t)+".dat", dtype=float64)
    nrm = abs(ue*ue).sum()
    return (abs((u-ue)).sum() / abs(u0).sum()), abs(u-ue).max()/abs(u0).max(), sqrt(abs((u-ue)**2).sum() / ((abs(u0)**2).sum()))

ntimes = 24
nchirps = 20

err = zeros((ntimes,nchirps), dtype=float64)
err_energy = zeros((ntimes,nchirps), dtype=float64)
errmax = zeros((ntimes,nchirps), dtype=float64)

for t in range(ntimes):
    for chirp in range(1,nchirps+1):
        err[t,chirp-1], errmax[t,chirp-1], err_energy[t,chirp-1] = error_from_sim(t,chirp)

logerr=log(err_energy+0.0000000000001)/log(10)

## subplot(211)
## imshow(transpose(logerr[:,::-1]),extent=(0,ntimes*2,1,nchirps+1))
## xlabel("Time")
## ylabel("Frequency")
## title("$log_{10}(\mid u-u_d \mid_{L^2} \/ \/ / \mid u_0 \mid_{L^2})$")
## colorbar()
## clabel(contour(transpose(logerr),[-1,-2,-3,-4, -5],extent=(0,ntimes*2,1,nchirps+1)),fmt="$10^{%1.0f}$")

max_errs = [max(err[:,i-1]) for i in range(1,nchirps+1)]
max_errs_linfty = [max(errmax[:,i-1]) for i in range(1,nchirps+1)]
max_errs_energy = [max(err_energy[:,i-1]) for i in range(1,nchirps+1)]
## subplot(212)
## gray()

title("Errors for the Euler Equations")
semilogy(arange(nchirps)+1,max_errs,label="$sup_t \mid u-u_d \mid_{L^1} \/ / \mid u_0 \mid_{L^1}$")
semilogy(arange(nchirps)+1,max_errs_linfty, label="$sup_t \mid u-u_d \mid_{L^\infty} \/ \/ \/ / \mid u_0 \mid_{L^\infty}$")
semilogy(arange(nchirps)+1,max_errs_energy, label="$sup_t \mid u-u_d \mid_{L^2} \/ \/ \/ / \mid u_0 \mid_{L^2}$")
ylabel("Relative and Absolute errors")
xlabel("Frequency")
legend()

savefig("euler_errors.eps", dpi=120)
