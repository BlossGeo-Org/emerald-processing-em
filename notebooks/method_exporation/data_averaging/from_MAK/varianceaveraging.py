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


########################################################
# # First just try two sets #############################
# x = np.random.normal(loc=1, scale=2, size=50)
# y = np.random.normal(loc=2, scale=1, size=50)
# mu_actual = np.average(np.concatenate((x, y)))
# var_actual = np.var(np.concatenate((x, y)))
#
# # Using SST
# mu = (np.average(x) + np.average(y))/2
# sst = calcSST([50, 50], [np.average(x), np.average(y)],
#     [np.std(x), np.std(y)], mu)
# # var_est = sst / 99
# var_est = sst / 100
#
# # Using a direct method
# var_est2 = ((np.var(x)+np.var(y))/2) + ((np.average(x)-np.average(y))/2)**2
#
# print('Two sets of samples')
# print('---------------------------------')
# print(' \tmu \tvar')
# # print('Real \t'+str(mu_actual)+'\t'+str(var_actual))
# print(f'Real \t{mu_actual:1.4f} \t{var_actual:1.4f}')
# print(f'SST \t{mu:1.4f} \t{var_est:1.4f}')
# print(f'Dir \t \t{var_est2:1.4f}')
#
#
# # Try with many sets ##################################
# x = []
# for ii in range(0, 20):
#     x.append(np.random.normal(loc=np.random.random(1) * 10,
#                               scale=np.random.random(1) * 3 + 2,
#                               size=50))
# z = np.concatenate(x)
#
# mu_actual = np.average(z)
# var_actual = np.var(z)
#
# mu = 0
# mu_ind_iv = []
# std_ind_iv = []
# n = []
# for a in x:
#     mu += np.average(a)
#     mu_ind_iv.append(np.average(a))
#     std_ind_iv.append(np.std(a))
#     n.append(50)
# mu = mu / 20
# sst = calcSST(n, mu_ind_iv, std_ind_iv, mu)
# var_est = sst / (50 * 20)
# print('20 sets of samples')
# print('---------------------------------')
# print(' \tmu \tvar')
# print(f'Real \t{mu_actual:1.4f} \t{var_actual:1.4f}')
# print(f'SST \t{mu:1.4f} \t{var_est:1.4f}')
