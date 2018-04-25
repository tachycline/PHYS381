# coding: utf-8
import numpy as np
from numpy.testing import assert_allclose

def test_get_energies_sho(n=3):
    """Test that find_energies gives correct results for the harmonic oscillator."""
    from .schrodinger import get_energies
    
    def sho_test(x):
        """Potential for the harmonic oscillator, with m=1, omega=1"""    
        return x**2/2

    desired_energies = np.array([i+0.5 for i in range(n)])
    assert_allclose(get_energies(sho_test, n), desired_energies)

def test_harmonic(n=3):
    """Test that the harmonic potential is the SHO."""
    from .schrodinger import get_energies, harmonic
    
    
    desired_energies = np.array([i+0.5 for i in range(n)])
    assert_allclose(get_energies(harmonic, n), desired_energies)    
    
def test_turning_point_isw():
    """Test that we get a/2 as the turning point for ISW"""
    from .schrodinger import turning_point, isw
    
    assert_close(0.5, turning_point((isw, ), 1.0))
    
def test_get_energies_isw(n=3):
    """Test that we get the right results for the infinite square well"""
    from .schrodinger import get_energies, isw
    
    desired_energies = np.array([np.pi**2*i**2/2 for i in range(n)])
    assert_allclose(get_energies(isw, n, soft_edge=False), desired_energies)