from random import random

import networkx as nx

from code.utils import utils_graph


def create_non_weighted_graph(n, m):
    G = nx.gnm_random_graph(n, m)
    return G


def create_weighted_graph(n, m, weight_range=(1, 10)):
    G = utils_graph.create_weighted_graph(nx.barabasi_albert_graph, n, m, weight_range=(1, 10))
    return G