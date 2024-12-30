import numpy as np

def calcSSE(n, sd):
    """
    Calculate the value of the Sum-of-Squares-of-Errors method

    Parameters
    ----------
    n :
        Population Size of each sample. Alternative use could be a weighting scheme
    sd :
        Standard Deviation of for each sample set
    """
    sse = 0
    for ii in range(0, len(n)):
        # Needed to get rid of the -1, because you're working with a population, not a sample
        # sse += (n[ii]-1) * sd[ii]**2
        sse += (n[ii]) * sd[ii]**2
    return sse

def calcSSG(n, mu, mu_tot):
    """
    Calculate the value fo the Sum-of-Squares-of-Groups method

    Parameters
    ----------
    n :
        Population Size of each sample. Alternative use could be a weighting scheme
    mu :
        Average value for each population
    mu_tot :
        Normalized sum of averaged values
    """
    ssg = 0
    for ii in range(0, len(n)):
        ssg += n[ii] * (mu[ii] - mu_tot)**2
    return ssg

def calcSST(n, mu, sd, mu_tot):
    """
    Calculate the value of the Sum-of-Squares-of-Total method

    Parameters
    ----------
    n :
        Population Size of each sample. Alternative use could be a weighting scheme
    mu :
        Average value for each population
    sd :
        Standard Deviation of for each sample set
    mu_tot :
        Normalized sum of averaged values
    """
    return calcSSE(n, sd) + calcSSG(n, mu, mu_tot)

def calcVarSST(n, mu, sd, mu_tot):
    """
    Calculate the variance of the Sum-of-Squares-of-Total method

    Parameters
    ----------
    n :
        Population Size of each sample. Alternative use could be a weighting scheme
    mu :
        Average value for each population
    sd :
        Standard Deviation of for each sample set
    mu_tot :
        Normalized sum of averaged values
    """
    return calcSST(n, mu, sd, mu_tot) / np.sum(n)
