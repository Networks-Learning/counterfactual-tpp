import numpy as np
from numpy.random import random
from scipy import interpolate


# samples from poisson process with intensity mu.
def homogenous_poisson(mu, t_previous):
    u = np.random.uniform(0, 1)
    t = t_previous - (np.log(1 - u) / mu)
    return t


def thinning_T(start, intensity, lambda_max, T):
    n = 0
    indicators = []
    sample = []
    next_arrival_time = start
    while True:
        next_arrival_time += np.random.exponential(scale=1.0 / lambda_max)

        if next_arrival_time < T:
            n += 1
            d = np.random.rand()
            lambda_s = intensity(next_arrival_time)
            sample.append(next_arrival_time)
            if d <= lambda_s / lambda_max:
                indicators.append(True)
                # n += 1
            else:
                indicators.append(False)
        else:
            break  
    return sample, indicators

def thinning(lambdas, lambda_max, sample):
    indicators = []
    accepted = []
    for i in range(len(lambdas)):
        u = np.random.uniform(0, 1)
        if lambdas[i]/lambda_max > u:
            accepted.append(sample[i])
            indicators.append(True)
        else:
            indicators.append(False)
    accepted = np.array(accepted)
    return accepted, indicators

# INVERSION SAMPLING
def inverse_sample(g, mean):
    x = np.linspace(0, 2 * mean, 100000)
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = g(x[i])              # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y, x)    # this is a function
    return inverse_cdf


def return_samples(f, mean, N=1e6):
    # generate some samples according to the chosen pdf, f(x)
    uniform_samples = random(int(N))
    required_samples = inverse_sample(f, mean=mean)(uniform_samples)
    return required_samples
