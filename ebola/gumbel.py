import numpy as np


def truncated_gumbel(alpha, truncation):
    gumbel = np.random.gumbel() + np.log(alpha)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))


# Calculating the posterior using using A* sampling idea
def topdown(alphas, k):
    topgumbel = np.random.gumbel() + np.log(sum(alphas))
    gumbels = []
    for i in range(len(alphas)):
        if i == k:
            gumbel = topgumbel
        else:
            gumbel = truncated_gumbel(alphas[i], topgumbel)
        gumbels.append(gumbel)
    return gumbels


# calculating the posterior using rejection sampling
def rejection(alphas, k):
    log_alphas = np.log(alphas)
    gumbels = np.random.gumbel(size=len(alphas))
    while k != np.argmax(gumbels + log_alphas):
        gumbels = np.random.gumbel(size=len(alphas))
    return (gumbels).tolist()


# posterior of noise for i th event, using A*
def posterior_A_star(i, lambdas, lambda_max, indicators):
    p_1 = lambdas[i]/lambda_max
    p_0 = 1 - p_1
    if indicators[i] == True:
        gumbels = np.array(topdown(np.array([p_0, p_1]), 1))
        return gumbels - np.log(np.array([p_0, p_1]))
    else:
        gumbels = np.array(topdown(np.array([p_0, p_1]), 0))
        return gumbels - np.log(np.array([p_0, p_1]))


# posterior of noise for i th event, using rejection sampling
def posterior2(i, lambdas, lambda_max, indicators):
    p_1 = lambdas[i]/lambda_max
    p_0 = 1 - p_1
    if indicators[i] == True:
        gumbels = np.array(rejection(np.array([p_0, p_1]), 1))
        return gumbels 
    else:
        gumbels = np.array(rejection(np.array([p_0, p_1]), 0))
        return gumbels
