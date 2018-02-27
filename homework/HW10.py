import numpy as np
import pandas as pd

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

def single_hist():
    pass

def hist_array():
    pass
