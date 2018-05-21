import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATAFILE = "../data/sdss-trimmed.pkl"

def read_data(filename=DATAFILE):
    """Read the sample data set.

    You could pretty easily do this by hand, but having a function like
    this can be useful if you have to do any conversion or munging on
    import.

    Can be called without arguments.
    """
    df = pd.read_pickle(filename)
    return df


def make_tranche(df, selection_value, selector_col="DIST_ADOP", selection_width=0.01, quantiles=True):
    """Cut a tranche of df.

    Parameters
    ----------
    df : pandas DataFrame
        The full data set
    selection_value : float
        Value that sets the center point of the tranche.
    selector_col : str
        Name of the column on which we're going to select.
    selection_width: float
        Width of the selection
    quantiles : bool
        If true, selection_value and selection_width are percentile values, otherwise,
        they're data values.

    Returns
    -------
    pandas DataFrame
        A new pandas DataFrame containing the requested subset of df.
    """

    minval = selection_value - selection_width/2
    maxval = selection_value + selection_width/2

    if quantiles:
        minval = df[selector_col].quantile[minval]
        maxval = df[selector_col].quantile[maxval]

    selector = (df[selector_col] < maxval) & (df[selector_col] > minval)

    tranche = df[selector]
    return pd.DataFrame(tranche)

# The following functions are for you to fill in:

def single_hist(df, percentile, ax=None, large=True):
    """Plot a single histogram """
    
    minval = df['DIST_ADOP'].quantile(percentile-0.005)
    maxval = df['DIST_ADOP'].quantile(percentile+0.005)
    selector = (df['DIST_ADOP'] > minval) & (df['DIST_ADOP'] < maxval)
    subset = df[selector]

    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()
        

    if large:
        fig.set_figwidth(12)
        fig.set_figheight(9)
        vals, bins, patches = ax.hist(subset['TEFF_UNC_REL'], bins=50, rwidth=0.9)

        maxbin = bins[np.argmax(vals)+1]
        maxbinval = vals[np.argmax(vals)]    
    
        ax.set_xlabel("Relative uncertainty")
        ax.set_ylabel("Number of observations")
        ax.set_title(r"Histogram of Relative Uncertainties in $T_{eff}$ for " +
                     "{} stars in the {} percentile of distance".format(len(subset), percentile*100),fontsize=18 )

        ax.text(maxbin, maxbinval,"Max: {} at {:5.3e}".format(maxbinval, maxbin),
                verticalalignment="center", horizontalalignment="left", fontsize=14)
    else: # small
        vals, bins, patches = ax.hist(subset['TEFF_UNC_REL'], bins=50, histtype="step")
        ax.set_axis_off()
                 
                 
def hist_array(df):
    """Make an array of histograms."""
    
    fig, axes = plt.subplots(10,10)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    nplots = axes.size
    for plot in zip(axes.flat, np.linspace(0.1, 0.99, nplots)):
        ax = plot[0]
        single_hist(df, plot[1], ax, large=False)
