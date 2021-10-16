import itertools
import json

import pandas as pd
import numpy as np
import networkx as nx

from settings import EBOLA_BASE_GRAPH_FILE, EBOLA_SCALED_GRAPH_FILE


def make_ebola_network(n_nodes, p_in, p_out, seed=None):
    """
    Build the EBOLA network with `n_nodes` based on the network of connected
    districts. Each district is mapped into a cluster of size proportional to
    the population of the district.

    Arguments:
    ==========
    n_nodes : int
        Desired number of nodes. Note: the resulting graph may have one node
        more or less than this number due to clique size approximation.
    p_in : float
        Intra-cluster edge probability
    p_out : dict
        Inter-country edge probability. It is a dict of float keyed by country for the
        inter-cluster edge probability between clusters inside a country, with an extra key
        'inter-country' for the probability of inter-cluster edge probability in different
        countries.

    Return:
    =======
    graph : networkx.Graph
        Undirected propagation network
    """
    # Load base graph
    with open(EBOLA_BASE_GRAPH_FILE, 'r') as f:
        base_graph_data = json.load(f)
    base_graph = nx.readwrite.json_graph.node_link_graph(base_graph_data)
    # Add inter-cluster edge probabilities
    for u, v, d in base_graph.edges(data=True):
        # If same country
        if base_graph.node[u]['country'] == base_graph.node[v]['country']:
            d['weight'] = p_out[base_graph.node[u]['country']]
        # If different country
        else:
            d['weight'] = p_out['inter-country']
    # Add intra-cluster edge-probability
    for u in base_graph.nodes():
        base_graph.add_edge(u, u, weight=p_in)
    # Replicate the base graph attributes to each cluster
    cluster_names = list(base_graph.nodes())
    country_names = [d['country'] for n, d in base_graph.nodes(data=True)]
    cluster_sizes = [int(np.ceil(n_nodes * base_graph.node[u]['size'])) for u in cluster_names]
    nodes_district_name = np.repeat(cluster_names, cluster_sizes)
    nodes_country_name = np.repeat(country_names, cluster_sizes)
    n_nodes = sum(cluster_sizes)
    # Build the intra/inter cluster probability matrix
    base_adj = nx.adjacency_matrix(base_graph, weight='weight').toarray().astype(float)
    # Generate stoch block model graph
    graph = nx.generators.stochastic_block_model(cluster_sizes, base_adj, seed=seed)
    # Assign district attribute to each node
    for u, district, country in zip(graph.nodes(), nodes_district_name, nodes_country_name):
        graph.node[u]['district'] = district
        graph.node[u]['country'] = country
    # Sanity check for name assignment of each cluster
    num_unique_block_district = len(set([(node_data['block'], node_data['district']) for u, node_data in graph.nodes(data=True)]))
    assert num_unique_block_district == len(cluster_names)
    # Extract the giant component
    graph = max(nx.connected_component_subgraphs(graph), key=len)
    return graph
