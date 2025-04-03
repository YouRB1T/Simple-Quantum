import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import cut_max
import algorithms_optimization


def visualize_solution(G, partition):
    layout = nx.spring_layout(G, seed=10)
    node_colors = ['red' if partition[i] == 0 else 'blue' for i in range(len(partition))]
    nx.draw(G, layout, node_color=node_colors, with_labels=True, edge_color="gray")
    labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    plt.show()


def run_algorithms(edges, num_vertices):
    best_solution_sa, best_value_sa, ans_time_sa, ans_value_sa = algorithms_optimization.simulated_annealing(
        lambda p: cut_max.create_objective_function(graph_edges, p),
        cut_max.initial_solution,
        cut_max.generate_neighbor_solution,
        num_vertices
    )
    print("Simulated Annealing:", best_solution_sa, best_value_sa, ans_time_sa, ans_value_sa)

    best_solution_ga, ga_time = algorithms_optimization.genetic_algorithm(
        lambda p: cut_max.create_objective_function(graph_edges, p),
        lambda: cut_max.initial_solution(num_vertices),
        cut_max.crossover,
        cut_max.generate_neighbor_solution
    )

    print("Genetic Algorithm:", best_solution_ga, cut_max.create_objective_function(best_solution_ga), ga_time)

    best_solution_tabu, tabu_time = algorithms_optimization.tabu_search(
        lambda p: cut_max.create_objective_function(graph_edges, p),
        lambda s: [cut_max.generate_neighbor_solution(s) for _ in range(10)],
        cut_max.initial_solution(num_vertices)
    )
    print("Tabu Search:", best_solution_tabu, cut_max.create_objective_function(best_solution_tabu), tabu_time)

    G = nx.Graph()
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)
    visualize_solution(G, best_solution_sa)

graph_edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.0), (1, 3, 1.0)]
num_vertices = 4

run_algorithms(graph_edges, num_vertices)