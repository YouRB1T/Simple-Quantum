import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import cut_max
import algorithms_optimization
from utils_graph import visualize_solution


def run_algorithms(edges, num_vertices):
    person_of_population, time = algorithms_optimization.genetic_algorithm(
        cut_max.objective_function,
        cut_max.initial_solution,
        cut_max.mutate,
        cut_max.crossover
    )

    print(person_of_population)


graph_edges = [(0, 1, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.0), (1, 3, 1.0)]
num_vertices = 4

run_algorithms(graph_edges, num_vertices)