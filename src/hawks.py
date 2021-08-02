# This code is taken from: https://github.com/Networks-Learning/jetbrains-seminar-2019/blob/hawkes-solution/hawkes/simPointProcess_solution.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')


def sampleHawkes(lambda_0, alpha_0, w, T, Nev, seed=None):
    """Generates samples from a Hawkes process with \lambda_0 and \alpha_0 until one of the following happens:
      - The next generated event is after T
      - Nev events have been generated.
    Returns: a tuple with the event times and the last generated time.
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

    return tev, Tend


def preprocessEv(tev, T, w):
    lambda_ti = np.zeros_like(tev, dtype=float)
    survival = 0

    for i in range(len(tev)):
        lambda_ti[i] = np.sum(np.exp(-w * (tev[i] - tev[0:i])))
        survival += (1.0 / w) * (1.0 - np.exp(-w * (T - tev[i])))

    return lambda_ti, survival


def plotHawkes(tevs, l_0, alpha_0, w, T, resolution):
    tvec = np.arange(0, T, step=T / resolution)

    # Expected intensity given parameters
    lambda_t = (np.exp((alpha_0 - w) * tvec) + w * (1.0 / (alpha_0 - w)) *
                (np.exp((alpha_0 - w) * tvec) - 1)) * l_0

    # Empirical average of intensities
    lambda_t_emp = np.zeros(len(tvec))

    # Plot individual lambda(t) for each event sequence
    colorLambda = ['r--', 'k--', 'g--', 'm--', 'c--']
    colorEv = ['r+', 'k+', 'g+', 'm+', 'c+']

    for i in range(len(tevs)):
        n = -1
        l_t = np.zeros(len(tvec))

        for t in tvec:
            n += 1
            l_t[n] = l_0 + alpha_0 * \
                np.sum(np.exp(-w * (t - tevs[i][tevs[i] < t])))

        plt.plot(tvec, l_t, colorLambda[i % len(colorLambda)],
                 alpha=10 * 1. / len(tevs))

        plt.plot(tevs[i], np.zeros(len(tevs[i])), colorEv[i % len(colorEv)],
                 alpha=10 * 1. / len(tevs))

        lambda_t_emp += l_t

    # Take average of lambda_t at all time instances
    lambda_t_emp /= len(tevs)

    # Plot expected mean intensity
    plt.plot(tvec, lambda_t, 'b-', linewidth=1.5, alpha=0.75,
             label=r'$\mathbb{E}[\lambda(t)]$')

    # Plot empirical mean intensity
    plt.plot(tvec, lambda_t_emp, 'b:', linewidth=1.5, alpha=0.75,
             label=r'$\bar{\lambda}(t)$')

    # Labels
    plt.xlabel('Time ($t$)')
    plt.ylabel(r'$\lambda_0(t), \dots, \lambda_{%d}(t)$' % (len(tevs),))
    plt.legend()


##################################################


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

# Number of samples to take
Nsamples = 20

tev = [None] * Nsamples
Tend = [None] * Nsamples
lambda_ti = [None] * Nsamples
survival = np.zeros(Nsamples)

for i in range(Nsamples):
    tev[i], Tend[i] = sampleHawkes(lambda_0, alpha_0, w, T, maxNev)
    lambda_ti[i], survival[i] = preprocessEv(tev[i], Tend[i], w)

plotHawkes(tev, lambda_0, alpha_0, w, T, 10000.0)
plt.ion()   # Make the plot interactive
plt.show()  # Show the plot. May not be needed in IPython
