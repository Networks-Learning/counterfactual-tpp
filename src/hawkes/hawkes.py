# some parts are taken from: https://github.com/Networks-Learning/jetbrains-seminar-2019/blob/hawkes-solution/hawkes/simPointProcess_solution.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath('..'))
from counterfactual_tpp import sample_counterfactual
from sampling_utils import thinning_T
sns.set_style('ticks')


def sampleHawkes(lambda_0, alpha_0, w, T, Nev, seed=None):
    """Generates samples from a Hawkes process with \lambda_0 and \alpha_0 until one of the following happens:
      - The next generated event is after T
      - Nev events have been generated.
    Returns: a tuple with the event times and the last generated time and their intensity values.
    """

    np.random.seed(seed)

    # First event is generated just as for a normal Poisson process.

    tev = np.zeros(Nev)
    n = 0
    lambda_star = lambda_0
    next_arrival_time = np.random.exponential(scale=1.0 / lambda_star)
    tev[n] = next_arrival_time

    # Generate the next events
    n += 1
    while n < Nev:
        lambda_star = lambda_star + alpha_0
        next_arrival_time += np.random.exponential(scale=1.0 / lambda_star)

        if next_arrival_time < T:
            d = np.random.rand()
            lambda_s = lambda_0 + alpha_0 * \
                np.sum(np.exp(-w * (next_arrival_time - tev[0:n])))

            if d <= lambda_s / lambda_star:
                tev[n] = next_arrival_time
                lambda_star = lambda_s
                n += 1
        else:
            break

    tev = tev[0:n - 1]

    if n == Nev:
        Tend = tev[-1]
    else:
        Tend = T
    lambdas = hawkes(tev, lambda_0, alpha_0, w)

    return tev, Tend, lambdas


def iterative_sampling(all_events, events, mu0, alpha, w, lambda_max, maxNev, T):
    """Generates samples from a Hawkes process with \mu_0 and \alpha_0 until one of the following happens:
      - The next generated event is after T
      - Nev events have been generated.
    This function uses the Superposition property to sample from the hawkes process.
    """
    if True not in events.values():
        return
    else:
        accepted_events = [k for k, v in events.items() if v == True]

        for t_i in accepted_events:
            def f(t): return alpha * np.exp(-w * (t - t_i))
            new_sample, indicators = thinning_T(t_i, f, lambda_max, maxNev, T)
            new_events = {new_sample[i]: indicators[i]
                          for i in range(len(new_sample))}
            all_events[t_i] = new_events
            iterative_sampling(all_events, new_events, mu0,
                               alpha, w, lambda_max, maxNev, T)


def extract_samples(all_events, sampled_events, mu0, alpha, w):
    # extracts all sampled events from all_events dictionary and also returns their intensity value.
    all_samples = []
    for events in list(all_events.values()):
        all_samples.extend(list(events.keys()))
    all_samples.sort()
    all_samples = np.array(all_samples)
    all_lambdas = [hawkes_calculate(
        x, sampled_events, mu0, alpha, w) for x in all_samples]
    all_lambdas = np.array(all_lambdas)
    return all_samples, all_lambdas


def sample_counterfactual_superposition(mu0, alpha, new_mu0, new_alpha, all_events, lambda_max, maxNev, w, T):
    """Generates samples from the counterfactual intensity, and return the counterfactuals.
    This is done in 3 steps:
        1. First we calculate the counterfactual basedon the history in each exponential (created by superposiiton.).
        2. Then we determine the events that were rejected in original intensity and accepted in intevened intensity (rej_acc).
        3. Then we create a new exponential and sample for each rej_acc event.
    """
    def constant1(x): return mu0
    def constant2(x): return new_mu0
    count = 0
    counterfactuals = {}
    for t_i, events in all_events.items():
        sample = list(events.keys())
        if count == 0:
            lambdas = [constant1(s) for s in sample]
            counterfactuals_events, counterfactual_indicators = sample_counterfactual(
                sample, lambdas, lambda_max, list(events.values()), constant2)
        else:
            def f(t): return alpha * np.exp(-w * (t - t_i))
            def g(t): return new_alpha * np.exp(-w * (t - t_i))
            lambdas = [f(s) for s in sample]
            counterfactuals_events, counterfactual_indicators = sample_counterfactual(
                sample, lambdas, lambda_max, list(events.values()), g)
        count += 1
        counterfactuals.update(
            {sample[s]: counterfactual_indicators[s] for s in range(len(sample))})

    rej_acc_events = {}
    for events in list(all_events.values()):
        for t_i in list(events.keys()):
            if events[t_i] == False and counterfactuals[t_i] == True:
                rej_acc_events[t_i] = True

    new_events = {}
    iterative_sampling(new_events, rej_acc_events, new_mu0,
                       new_alpha, w, lambda_max, maxNev, T)
    # These are the additional counterfactuals sampled from the new intensity.
    sampled_counterfactuals = list(new_events.keys())
    sampled_counterfactuals.sort()

    # Combine all counterfactuals
    real_counterfactuals = [k for k, v in counterfactuals.items() if v == True]
    real_counterfactuals.extend(sampled_counterfactuals)
    real_counterfactuals.sort()
    real_counterfactuals = list(dict.fromkeys(real_counterfactuals))
    return real_counterfactuals


def hawkes(tev, l_0, alpha_0, w):
    # calculates the value of intensity for a series of samples (i.e., tev)
    lambda_ti = np.zeros_like(tev, dtype=float)
    for i in range(len(tev)):
        lambda_ti[i] = l_0 + alpha_0 * np.sum(np.exp(-w * (tev[i] - tev[0:i])))
    return lambda_ti


def hawkes_calculate(x, tev, l_0, alpha_0, w):
    return l_0 + alpha_0 * np.sum(np.exp(-w * (x - tev[tev < x])))


def plotHawkes(tevs, l_0, alpha_0, w, T, resolution, label, color, legend):

    tvec = np.arange(0, T, step=T / resolution)

    n = -1
    l_t = np.zeros(len(tvec))
    for t in tvec:
        n += 1
        l_t[n] = l_0 + alpha_0 * np.sum(np.exp(-w * (t - tevs[tevs < t])))

    plt.plot(tvec, l_t, label=label)

    plt.plot(tevs, np.zeros(len(tevs)), color, label = legend)
    return tvec, l_t

def check_monotonicity_hawkes(mu0, alpha, new_mu0, new_alpha, all_events, sampled_events, real_counterfactuals, w):
    count = 0
    monotonic = 1
    def constant1(x): return mu0
    def constant2(x): return new_mu0
    for t_i, events in all_events.items():
        sample = list(events.keys())
        if count == 0:
            for s in sample: 
                if constant2(s) >= constant1(s) and s in sampled_events:
                    if s not in real_counterfactuals:
                        print('NOT  MONOTONIC')
                        monotonic = 0
                if constant2(s) < constant1(s) and s not in sampled_events:
                    if s in real_counterfactuals:
                        print('NOT  MONOTONIC')
                        monotonic = 0
        else:
            for s in sample:
                def f(t): return alpha * np.exp(-w * (t - t_i))
                def g(t): return new_alpha * np.exp(-w * (t - t_i))
                if g(s) >= f(s) and s in sampled_events:
                    if s not in real_counterfactuals:
                        print('NOT  MONOTONIC')
                        monotonic = 0
                if g(s) < f(s) and s not in sampled_events:
                    if s in real_counterfactuals:
                        print('NOT  MONOTONIC')
                        monotonic = 0
        count += 1
        
    if monotonic == 1:
            print('MONOTONIC')
##############################################################
# Simulation time
# T = 10

# # Maximum number of events per realization
# maxNev = 200

# # Base intensity
# lambda_0 = 1

# # Self excitation parameter
# alpha_0 = 0.5

# # Rate of decay
# w = 1

# tev, tend, lambdas = sampleHawkes(lambda_0, alpha_0, w, T, maxNev)
# plotHawkes(tev, lambda_0, alpha_0, w, T, 10000.0, label='test')
# plt.plot(tev, lambdas, 'r^')
# plt.ion()
# plt.show()  # Show the plot.
