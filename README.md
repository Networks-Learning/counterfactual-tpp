# counterfactual-ttp
This is a repository containing code the Counterfactual Temporal Point Processes.
## Pre-requisites

This code depends on the following packages:

 1. [`netwrokx`](https://networkx.org/)
 2. `numpy`
 3. `pandas`
 4. `matplotlib`
 
 
 to generate map plots:
 
 5. [`GeoPandas`](https://geopandas.org/)
 6. `geoplot`

## Code structure

 - [src/counterfactual_tpp.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/counterfactual_tpp.py) Contains the code to sample rejected events using the superposition property and the algorithm to calculate the counterfactuals.
 - [src/gumbel.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/gumbel.py) Contains the utility functions for the Gumbel-Max SCM.
 - [src/sampling_utils.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/sampling_utils.py) Contains the code for the Lewis' thinning algorithm (`thinning_T` function) and some other sampling utilities.
 - [src/hawkes/hawkes.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/hawkes/hawkes.py) Contains the code for sampling from the hawkes process using the superposition property of tpps. It also includes the algorithm for sampling a counterfactual sequence of events given a sequence of observed events for a Hawkes process.
 - [src/hawkes/hawkes_example.ipynb:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/hawkes/hawkes_example.ipynb) Contains an example of running algorithm 3 (in the paper) for both cases where we have (1) both observed and un-observed events, and (2) the case that we have only the observed events.
 - [ebola/graph_generation.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/ebola/graph_generation.py) Contains code to build the Ebola network based on the network of connected
    districts. This code is adopted from the [disease-control](https://github.com/Networks-Learning/disease-control) project. 
 - [ebola/dynamics.py:](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/ebola/dynamics.py) Contains code for sampling counterfactual sequence of infections given a sequence of observed infections from the SIR porcess (the ` calculate_counterfactual` function). The rest of the code is adopted from the [disease-control](https://github.com/Networks-Learning/disease-control) project, which simulates continuous-time SIR epidemics with exponentially distributed
    inter-event times.

The directory [ebola/data/ebola](https://github.com/Networks-Learning/counterfactual-ttp/tree/main/ebola/data/ebola) contains the information about the Ebola network adjanceny matrix and the cleaned ebola outbreak data adopted from the [disease-control](https://github.com/Networks-Learning/disease-control) project.

The directory [ebola/map/geojson](https://github.com/Networks-Learning/counterfactual-ttp/tree/main/ebola/map/geojson) contains the geographical information of the districts studied in the Ebola outbreak dataset. The geojson files are obtained from [Nominatim](https://nominatim.openstreetmap.org/ui/search.html).

The directory [ebola/map/overall_data](https://github.com/Networks-Learning/counterfactual-ttp/tree/main/ebola/map/overall_data) contains data for generating the geographical maps in the paper, and includs the overall number of infection under applying different interventions.

The directories [src/data_hawkes](https://github.com/Networks-Learning/counterfactual-ttp/tree/main/src/data_hawkes) and [src/data_inhomogeneous](https://github.com/Networks-Learning/counterfactual-ttp/tree/main/src/data_inhomogeneous) contain observational data used to generate Synthetic plots in the paper. You can use this data to re-generate paper's plots. Otherwise, you can simply generate new random samples by the code.

## Experiments 

### Synthetic
 - [Inhomogeneous Poisson Processes](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/inhomogeneous_experiments.ipynb)
 - [Hawkes Processes](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/src/hawkes_experiments.ipynb)

### Epidemiological
- [Ebola Epidemic Simulation and Counterfactual Calculations](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/ebola/ebola_experiments.ipynb)
- [Generate Geographical Distribution of infections](https://github.com/Networks-Learning/counterfactual-ttp/blob/main/ebola/map/generate_geopands_data.ipynb)

## Citation
If you use parts of the code in this repository for your own research, please consider citing.