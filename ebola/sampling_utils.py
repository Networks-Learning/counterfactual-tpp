import numpy as np

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
    return np.array(sample)[indicators]