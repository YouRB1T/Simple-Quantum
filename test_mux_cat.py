import json

import numpy as np
from json_tricks import dump
import statistics
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.circuit import ParameterVectorElement
from qiskit_optimization.applications import Maxcut

from code.quantum import algorythms
from code.tasks import cut_max
from code.algs import algorithms_optimization
from code.utils import max_cut_generator_graph

def make_json_serializable(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():

            if isinstance(k, ParameterVectorElement):
                k = str(k)
            elif not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)
            new_dict[k] = make_json_serializable(v)
        return new_dict
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    return obj


num_graphs = 100
n = 10
m = 3

graphs = [max_cut_generator_graph.create_weighted_graph(n, m) for _ in range(num_graphs)]

results = {
    "simulated_annealing": [],
    "genetic_algorithm": [],
    "tabu_search": [],
    "qaoa_reps_1": [],
    "qaoa_reps_2": []
}

for idx, G in enumerate(graphs, start=1):
    print(f"Graph #{idx}")

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edges = [(u, v, float(w)) for (u, v), w in edge_labels.items()]

    print("  Simulated Annealing…")
    sa_metrics = algorithms_optimization.simulated_annealing(
        objective_function=cut_max.objective_function,
        initial_solution=cut_max.initial_solution,
        generate_neighbor_solution=cut_max.generate_neighbor_solution,
        num_vertices=n,
        edges=edges
    )
    results["simulated_annealing"].append(sa_metrics)

    print("  Genetic Algorithm…")
    ga_metrics = algorithms_optimization.genetic_algorithm(
        objective_function=cut_max.objective_function,
        initial_solution_fn=cut_max.initial_solution,
        mutate=cut_max.mutate,
        crossover=cut_max.crossover,
        num_vertices=n,
        edges=edges
    )
    results["genetic_algorithm"].append(ga_metrics)

    print("  Tabu Search…")
    ts_metrics = algorithms_optimization.tabu_search(
        objective_function=cut_max.objective_function,
        generate_neighbors=cut_max.generate_neighbors,
        initial_solution=cut_max.initial_solution,
        edges=edges,
        num_vertices=n
    )
    results["tabu_search"].append(ts_metrics)

    print("  QAOA reps=1…")
    q1_metrics = algorythms.qaoa_solver(n, G, edges, reps=1)
    results["qaoa_reps_1"].append(q1_metrics)

    print("  QAOA reps=2…")
    q2_metrics = algorythms.qaoa_solver(n, G, edges, reps=2)
    results["qaoa_reps_2"].append(q2_metrics)

erializable_results = make_json_serializable(results)
with open("benchmark_results_10.json", "w") as f:
    dump(erializable_results, f, indent=2)

all_time = []
all_results = []
labels = ["SA", "GA", "TS", "QAOA1", "QAOA2"]
for key in labels:
    alg_key = {
        "SA": "simulated_annealing",
        "GA": "genetic_algorithm",
        "TS": "tabu_search",
        "QAOA1": "qaoa_reps_1",
        "QAOA2": "qaoa_reps_2"
    }[key]
    times = [m["total_time"] for m in results[alg_key] if "total_time" in m]
    vals = [m["best_value"] for m in results[alg_key] if "best_value" in m]
    all_time.append(times)
    all_results.append(vals)

fig, axes = plt.subplots(1,2, figsize=(12,5))

for i, label in enumerate(labels):
    axes[0].plot(all_time[i], label=label)
    axes[1].plot(all_results[i], label=label)

axes[0].set_title("Время работы")
axes[1].set_title("Значение функции")
for ax in axes:
    ax.legend()
plt.show()

print("Средние времена и значения:")
for i, label in enumerate(labels):
    print(f"{label}: time={statistics.mean(all_time[i]):.4f} ± {statistics.pstdev(all_time[i]):.4f},"
          f" value={statistics.mean(all_results[i]):.2f} ± {statistics.pstdev(all_results[i]):.2f}")

