def parameter_map(table, param, binMap):
    """
    Return a 2D map of the given parameter
    :param table: the BINTable in which the parameters are found
    :param param: The parameter we want to plot. See table.columns to see the list
    :param binMap: the 2D map of the binning scheme
    """
    def f(binNumber):
        return table[param][binNumber]
    return f(binMap)

def read(filename):
    hdu = fits.open(filename)
    bin_table = hdu[1].data
    fit_table = hdu[2].data
