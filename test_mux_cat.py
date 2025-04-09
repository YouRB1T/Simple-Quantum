import statistics

import networkx as nx
from matplotlib import pyplot as plt
from qiskit_optimization.applications import Maxcut

from code.quantum import algorythms
from code.tasks import cut_max
from code.algs import algorithms_optimization
from code.utils import max_cut_generator_graph


all_time = [[], [], [], []]
all_results_func = [[], [], [], []]
num_graphs = 10
n = 20
m = 15

graphs = [max_cut_generator_graph.create_weighted_graph(n, m) for _ in range(num_graphs)]
i = 1
for G in graphs:
    print(i)
    i += 1
    edge_labels = {k: f'{float(v):.3f}' for k, v in nx.get_edge_attributes(G, 'weight').items()}
    edges = [(*key, int(float(value))) for key, value in edge_labels.items()]

    best_solution, best_value_1, time = algorithms_optimization.simulated_annealing(
        cut_max.objective_function,
        cut_max.initial_solution,
        cut_max.generate_neighbor_solution,
        n,
        edges
    )
    all_time[0].append(time)
    all_results_func[0].append(best_value_1)

    best_solution, time, best_value_2 = algorithms_optimization.genetic_algorithm(
        cut_max.objective_function,
        cut_max.initial_solution,
        cut_max.mutate,
        cut_max.crossover,
        n,
        edges
    )

    all_time[1].append(time)
    all_results_func[1].append(best_value_2)

    best_solution, time, best_value_3 = algorithms_optimization.tabu_search(
        cut_max.objective_function,
        cut_max.generate_neighbors,
        cut_max.initial_solution,
        edges,
        n
    )

    all_time[2].append(time)
    all_results_func[2].append(best_value_3)

    best_solution, time = algorythms.qaoa_solver(n, G, edges)
    all_time[3].append(time)
    all_results_func[3].append(best_solution)


print(statistics.mean(all_time[0]), statistics.mean(all_results_func[0]))
print(statistics.mean(all_time[1]), statistics.mean(all_results_func[1]))
print(statistics.mean(all_time[2]), statistics.mean(all_results_func[2]))
