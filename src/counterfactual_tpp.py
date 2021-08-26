import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from gumbel import posterior_A_star, posterior2
from sampling_utils import return_samples, thinning_T


def sample_counterfactual(sample, lambdas, lambda_max, indicators, new_intensity):
    """Samples from the counterfactual intensity given the following:
        - sample: h or the set of all events (i.e., t_is)
        - lambdas: the intensity of the events (i.e, lambda(t_i)s)
        - lambda_max
        - indicators: value of the u_is
        - new_intensity: lambda' (a python function)
    Returns: a sample from the counterfactual intensity
    """
    counterfactuals = []
    counterfactual_indicators = []
    k = 100
    for i in range(len(sample)):
        ups = []
        for j in range(k):
            post = posterior_A_star(i, lambdas, lambda_max, indicators)
            pp_1 = new_intensity(sample[i])/lambda_max
            pp_0 = 1 - pp_1
            up = np.argmax(np.log(np.array([pp_0, pp_1])) + post)
            ups.append(up)
        if sum(ups)/k > 0.5:
            counterfactuals.append(sample[i])
            counterfactual_indicators.append(True)
        else:
            counterfactual_indicators.append(False)
    return counterfactuals, counterfactual_indicators


def superposition(lambda_max, original_intensity, mean, number_of_samples):
    """Calculatetes a h_observed and h_rejected
    """
    h_observed = np.sort(return_samples(
        original_intensity, mean=mean, N=number_of_samples))
    lambda_observed = [original_intensity(i) for i in h_observed]
    lambda_bar = lambda x: lambda_max - original_intensity(x)
    h_rejected = np.sort(return_samples(
        lambda_bar, mean=mean, N=number_of_samples))
    lambda_bar_rejected = [lambda_bar(i) for i in h_rejected]
    return h_observed, lambda_observed, h_rejected, lambda_bar_rejected


def combine(h_observed, lambda_observed, h_rejected, original_intensity):
    # combining both observed and rejected
    sample = []
    lambdas = []
    indicators = []
    all = []
    for i in range(len(h_observed)):
        all.append((h_observed[i], lambda_observed[i], True))
    for i in range(len(h_rejected)):
        all.append((h_rejected[i], original_intensity(
            h_rejected[i]), False))  # IMPORTANT

    h = sorted(all, key=lambda x: x[0])
    for i in range(len(h)):
        sample.append(h[i][0])
        lambdas.append(h[i][1])
        indicators.append(h[i][2])

    sample = np.array(sample)
    lambdas = np.array(lambdas)
    return sample, lambdas, indicators

def check_monotonicity(sample, counterfactuals, original_intensity, intervened_intensity, accepted):
    monotonic = 1
    for s in sample:
        if intervened_intensity(s) >= original_intensity(s) and s in accepted:
            if s not in counterfactuals:
                print('NOT  MONOTONIC')
                monotonic = 0
    for s in sample:
        if intervened_intensity(s) < original_intensity(s) and s not in accepted:
            if s in counterfactuals:
                print('NOT  MONOTONIC')
                monotonic = 0
    if monotonic == 1:
        print('MONOTONIC')
    
def distance(accepted, counterfactuals, T):
    # Calculates the distance between oserved and counterfactual realizaitons
    k1 = len(accepted)
    k2 = len(counterfactuals)
    if k1 <= k2:
        d = np.sum(np.abs(accepted[0:k1] - counterfactuals[0:k1]))
        if k2 - k1 > 0:
            d += np.sum(np.abs(T - counterfactuals[k1:]))
    else:
        d = np.sum(np.abs(accepted[0:k2] - counterfactuals[0:k2]))
        if k1 - k2 > 0:
            d += np.sum(np.abs(T - accepted[k2:]))
    return d

def calculate_N(t, indicators, sample):
    count = 0
    for i in range(len(sample)):
        if sample[i] <= t and indicators[i] == True:
            count += 1
    return count

def covariance(sample, indicators, counterfactual_indicators, T, original_intensity, intervened_intensity, lambda_max):
    times  = np.linspace(0, T, 20)
    n_realizations = 10
    n_counter = 10
    all = np.zeros(len(times))
    Ns = np.zeros(len(times))
    Ms = np.zeros(len(times))
    for realization in range(n_realizations):
        sample, indicators = thinning_T(0, intensity=original_intensity, lambda_max=lambda_max, max_number_of_samples=100, T=T)
        sum_mul = np.zeros(len(times))
        N = np.array([calculate_N(times[i], indicators, sample) for i in range(len(times))])
        lambdas = original_intensity(np.asarray(sample))
        sample = np.asarray(sample)
        for counter in range(n_counter):
            counterfactuals, counterfactual_indicators = sample_counterfactual(sample, lambdas, lambda_max, indicators, intervened_intensity)
            M = np.array([calculate_N(times[i], counterfactual_indicators, sample) for i in range(len(times))])
            Ms += M
            sum_mul += M * N
        all += sum_mul/n_counter
        Ns += N
    expected_MN = all/n_realizations
    expected_N = Ns/n_realizations
    expected_M = Ms / (n_realizations * n_counter)
    cov  = expected_MN - expected_M * expected_M
    return cov