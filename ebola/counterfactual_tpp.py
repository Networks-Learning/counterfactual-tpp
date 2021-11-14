import numpy as np
import os
import sys
from gumbel import posterior_A_star
from sampling_utils import thinning_T


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
        pp_1 = new_intensity(sample[i])/lambda_max
        pp_0 = 1 - pp_1
        for j in range(k):
            post = posterior_A_star(i, lambdas, lambda_max, indicators)
            up = np.argmax(np.log(np.array([pp_0, pp_1])) + post)
            ups.append(up)
        if sum(ups)/k > np.random.uniform(0, 1):
            counterfactuals.append(sample[i])
            counterfactual_indicators.append(True)
        else:
            counterfactual_indicators.append(False)
    return np.array(counterfactuals), np.array(counterfactual_indicators)

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