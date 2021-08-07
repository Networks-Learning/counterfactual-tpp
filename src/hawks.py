# This code is taken from (with required modifications): https://github.com/Networks-Learning/jetbrains-seminar-2019/blob/hawkes-solution/hawkes/simPointProcess_solution.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def hawkes(tev, l_0, alpha_0, w):
    # calculates the value of intensity for a series of samples (i.e., tev)
    lambda_ti = np.zeros_like(tev, dtype=float)
    for i in range(len(tev)):
        lambda_ti[i] = l_0 + alpha_0 * np.sum(np.exp(-w * (tev[i] - tev[0:i])))
    return lambda_ti


def hawkes_calculate(x, tev, l_0, alpha_0, w):
    return l_0 + alpha_0 * np.sum(np.exp(-w * (x - tev[tev < x])))


def plotHawkes(tevs, l_0, alpha_0, w, T, resolution, label):

    tvec = np.arange(0, T, step=T / resolution)

    n = -1
    l_t = np.zeros(len(tvec))
    for t in tvec:
        n += 1
        l_t[n] = l_0 + alpha_0 * np.sum(np.exp(-w * (t - tevs[tevs < t])))

    plt.plot(tvec, l_t, label=label)

    plt.plot(tevs, np.zeros(len(tevs)), 'r+')
    return tvec, l_t


##############################################################
# Simulation time
T = 10

# Maximum number of events per realization
maxNev = 200

# Base intensity
lambda_0 = 1

# Self excitation parameter
alpha_0 = 0.5

# Rate of decay
w = 1

tev, tend, lambdas = sampleHawkes(lambda_0, alpha_0, w, T, maxNev)
plotHawkes(tev, lambda_0, alpha_0, w, T, 10000.0, label='test')
plt.plot(tev, lambdas, 'r^')
plt.ion()
plt.show()  # Show the plot.
