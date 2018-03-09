"""Some functions for playing around with Anscombe's quartet."""
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# These should be typed in by hand here, but I've already done that once.
DATAFILES = ["ans1.pkl", "ans2.pkl", "ans3.pkl", "ans4.pkl"]

DATADFS = []
for file in DATAFILES:
    DATADFS.append(pd.read_pickle("../data/"+file))

def linear(x, a, b):
    """A generic linear function."""
    return b + a*x

def make_fits(dflist):
    """Construct linear fits to the data sets in dflist.
    
    Parameters:
    -----------
    dflist : iterable
             the data sets to fit
    
    Returns:
    --------
    a list of tuples of fitting parameters.
    """
    fits = []
    
    for df in dflist:
        popt, pcov = curve_fit(linear, df.x, df.y)
        fits.append(popt)
    return fits

def plot_fits():
    """Make a plot showing all four data sets and their linear fits."""
    
    fits = make_fits(DATADFS)
    
    fig, ax = plt.subplots(2,2)
    fig.suptitle("Anscombe's Quartet")
    fig.set_figwidth(12)
    fig.set_figheight(9)
    
    for data, fit, axis in zip(DATADFS, fits, ax.flat):
        axis.plot(data.x, data.y, "o")
        axis.plot(data.x.sort_values(), linear(data.x.sort_values(), *fit))
    
    return fig, ax