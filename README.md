# counterfactual-ttp
This is a repository containing code the Counterfactual Temporal Point Processes project.
## Pre-requisites

This code depends on the following packages:

 1. `numpy`
 2. `matplotlib`

## Code structure

 - `src/counterfactual_tpp.py` contains the code to sample rejected events using the superposition property and the algorithm to calculate the counterfactuals.
 - `src/gumbel.py` contains the utility functions for the Gumbel-Max SCM.
 - `src/sampling_utils.py` contains the code for the Lewis' thinning algorithm and some other sampling utilities.
 - `src/hawkes/hawkes.py` contains the code for sampling from the hawkes process using the Ogata's Algorithm, and also the using the superposition property of tpps. It also includes the algorithm for sampling a counterfactual sequence of events given a sequence of observed events for a Hawkes process.
----

# Experiments 

## Synthetic
`src/hawkes/hawkes.ipynb` and `src/inhomogeneous/gaussian.ipynb` contain code and result plots for the hawkes process and the gaussian intensities respectively.

## Epidemiological