==================================
Simple phase space filter examples
==================================

Introduction
============

In 2008, I (together with Avy Soffer) wrote a paper on Phase Space Filters,
`A stable absorbing boundary layer for anisotropic waves`_.

.. _version:
.. _A stable absorbing boundary layer for anisotropic waves: http://arxiv.org/abs/0805.2929


This git repo stores the source code for that paper.

Hint for possible TDPSF users
-----------------------------

I have developed two separate versions of the TDPSF. One version,
described in the paper `Open Boundaries for the Nonlinear Schrodinger Equation`_.

.. _Open Boundaries for the Nonlinear Schrodinger Equation: http://arxiv.org/abs/math/0609183

is based on the Windowed Fourier Transform. This version has provable
error bounds, which is why it was the first paper I published, and why
my Ph.D. thesis was based on it.

The other version_ (the one this repo is based on) uses a *simpler* and *faster*
phase space filter. The accuracy of this filter is not proven, however in practice
I have found it to be just as effective as the WFT based filter.

Additionally, the running time of the program is *vastly* superior, by a factor of 8-32x.

I *strongly* recommend that anyone who wishes to use phase space filters
should use the simplified version.


Guide
=====

Dependencies
------------

You need python (probably 2.6 or later, but NOT 3.0), numpy, matplotlib and scipy.

http://python.org/

http://numpy.scipy.org/

http://www.scipy.org/

http://matplotlib.sourceforge.net/


schrodinger_test.py
-------------------
The file `schrodinger_test.py` is the program I used to generate Figure 1 of Section 3.1.
It implements the TDPSF for the one-dimensional Schrodinger equation.

This example is 1-dimensional.

Euler Solvers
-------------
The Euler equations are solved using a standard FFT-based spectral propagator. This
example is 2-dimensional.

The file `euler_exact.py` solves the problem on a grid of size `npoints * dx = 2048 * 0.125 = 256`,
which is large enough that the waves will not reach the boundary before `t=50`.

The file `euler_subsonic.py` solves the problem with the phase space filters.

The file `error_check_euler.py` measures the error by comparing the TDPSF simulations to
the exact simulations. The simulations are stored in the relative directory `euler_subsonic_$K`
and `euler_subsonic_exact_$K`, where `$K` is the frequency of the initial condition.

The file `euler_subsonic_long.py` solves the problem for a long time to measure the stability.

Maxwell Solvers
---------------

The naming for the maxwell examples is the same as for the euler examples.

