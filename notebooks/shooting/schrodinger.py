# coding: utf-8
import numpy as np
from scipy.optimize import bisect
from .shooting import *

def get_energies(V, n=3, soft_edge=True):
    """Use shoot_for_eigenvalues to find energies.

    """
    args = (V,)
    ic = (0,1)
    if soft_edge:
        get_interval_endpoint = lambda x, y: turning_point(x, y, increment=5)
    else: # e.g., infinite square well
        get_interval_endpoint = turning_point
    energies = shoot_for_eigenvalues(schro_rhs, args, ic, get_interval_endpoint, n)
    return energies

def schro_rhs(y, x, E, V, m=1.0, hbar=1.0):
    """RHS for the time independent schrodinger equation.
    
    Parameters:
    -----------
    y : iterable of floats
        contains $\psi(x)$ and $\psi'(x)$
    x : float
        the position at which we're evaluating the equation
    E : float
        Energy of the quantum state
    V : function
        function for the potential
    m : float, optional
        mass; defaults to 1
    hbar : float, optional
           Planck's constant/2pi; defaults to 1
    
    Returns:
    --------
    float : the right hand side of the time independent schrodinger
            equation for the time independent Schrodinger equation:
            $$\frac{d^2\psi}{dx^2} = -\frac{2m}{\hbar^2}
            \left(E - V(x) \right)\psi.$$
    """
    psi, psiprime = y
    psidoubleprime = -(2*m/hbar**2)*(E-V(x))*psi
    
    return np.array([psiprime, psidoubleprime])

# rewrite as get_interval_endpoint
def turning_point(args, E, ubound=1000, increment=0):
    """find the classical turning point for the potential and energy"""
    
    V = args[0]
    ediff = lambda x: V(x) - E
    x_t = bisect(ediff, 0, ubound)
    
    return x_t + increment

####################
#  Potentials
####################

def harmonic(x, m=1, omega=1):
    """Potential for the harmonic oscillator"""
    
    return m*omega**2*x**2/2

@np.vectorize
def isw(x, a=1):
    """Infinite square well potential"""
    if np.abs(x) <= a/2:
        return 0
    else:
        return np.inf
