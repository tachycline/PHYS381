# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# import the shooting stuff
from .schrodinger import schro_rhs, get_energies, turning_point

def plot_energy_diagram(V, n=3, soft_edge=True):
    """Plot an energy diagram for a potential well."""
    
    domain = np.linspace(-5, 5, 1000)
    potential = V(domain)
    minx = np.min(domain)
    maxx = np.max(domain)
    
    fig, ax = plt.subplots(1)
    fig.set_figwidth(10)
    fig.set_figheight(6)
    
    ax.plot(domain, potential)
    ax.set_axis_off()
    
    # plot energies and eigenfunctions
    energies = get_energies(V, n, soft_edge)
    even = True
    for energy in energies:
        ax.plot([minx, maxx], [energy, energy], ":", color="gray")
        y0 = (0.0, 1.0)
        if soft_edge:
            result = odeint(schro_rhs, y0, domain, args=(energy, V))
        else:
            tp = turning_point((V,), energy)-1.0e-6
            # still only works for symmetric potentials
            newdomain = np.linspace(-tp, tp, 1000)
            result = odeint(schro_rhs, y0, newdomain, args=(energy, V))
        amp = np.max(result[:,0]) - np.min(result[:,0])
        normalized = result[:,0]/amp
        # this needs to be made more general
        eigenfunction = 0.3*normalized + energy
        ax.plot(domain, eigenfunction)
        ax.text(4.5, energy,"$E={:5.2f}$".format(energy),  verticalalignment="bottom")
        even = not even
    deltae = energies[-1] - energies[-2]    
    ax.set_ylim(-0.1, energies[-1] + deltae)
