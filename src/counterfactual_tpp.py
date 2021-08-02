import numpy as np
from src.gumbel import posterior_A_star
from src.sampling_utils import return_samples


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
            post = posterior_A_star(0, lambdas, lambda_max, indicators)
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
