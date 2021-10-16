import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

EBOLA_BASE_GRAPH_FILE = os.path.join(DATA_DIR, 'ebola', 'ebola_base_graph.json')
EBOLA_SCALED_GRAPH_FILE = os.path.join(DATA_DIR, 'ebola', 'ebola_scaled_graph.json')
