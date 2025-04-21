import json
import numpy as np
from json_tricks import dump
import statistics
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.circuit import ParameterVectorElement

from code.quantum import algorythms
from code.tasks import cut_max
from code.algs import algorithms_optimization
from code.utils import max_cut_generator_graph


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    return obj


def load_best_params(filename, n):
    with open(filename, "r") as f:
        params_by_size = json.load(f)
    return params_by_size[str(n)]["best_params"]


# Размерности графов: от 5 до 10
for n in range(10, 16):
    print(f"\n==== Размер графа: {n} ====")

    # Загрузка параметров из файлов
    sa_params = load_best_params("C:\\Users\\Dont_use_user\\PycharmProjects\\Simple-Quantum\\code\\results\\simulated_annealing_optuna_results.json", n)
    ga_params = load_best_params("C:\\Users\\Dont_use_user\\PycharmProjects\\Simple-Quantum\\code\\results\\genetic_optuna_results.json", n)
    ts_params = load_best_params("C:\\Users\\Dont_use_user\\PycharmProjects\\Simple-Quantum\\code\\results\\tabu_search_optuna_results.json", n)
    qaoa_params = load_best_params("C:\\Users\\Dont_use_user\\PycharmProjects\\Simple-Quantum\\code\\results\\qaoa_optuna_results.json", n)

    results = {
        "simulated_annealing": [],
        "genetic_algorithm": [],
        "tabu_search": [],
        "qaoa": []
    }

    num_graphs = 100
    m = 3
    graphs = [max_cut_generator_graph.create_weighted_graph(n, m) for _ in range(num_graphs)]

    for idx, G in enumerate(graphs, start=1):
        print(f"  Граф #{idx}")
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edges = [(u, v, float(w)) for (u, v), w in edge_labels.items()]

        # Simulated Annealing
        sa_result = algorithms_optimization.simulated_annealing(
            objective_function=cut_max.objective_function,
            initial_solution=cut_max.initial_solution,
            generate_neighbor_solution=cut_max.generate_neighbor_solution,
            num_vertices=n,
            edges=edges,
            **sa_params
        )
        results["simulated_annealing"].append(sa_result)

        # Genetic Algorithm
        ga_result = algorithms_optimization.genetic_algorithm(
            objective_function=cut_max.objective_function,
            initial_solution_fn=cut_max.initial_solution,
            mutate=cut_max.mutate,
            crossover=cut_max.crossover,
            num_vertices=n,
            edges=edges,
            **ga_params
        )
        results["genetic_algorithm"].append(ga_result)

        # Tabu Search
        ts_result = algorithms_optimization.tabu_search(
            objective_function=cut_max.objective_function,
            generate_neighbors=cut_max.generate_neighbors,
            initial_solution=cut_max.initial_solution,
            edges=edges,
            num_vertices=n,
            **ts_params
        )
        results["tabu_search"].append(ts_result)

        # QAOA
        reps = qaoa_params["reps"]
        maxiter = qaoa_params.get("maxiter")
        optimizer_name = qaoa_params.get("optimizer")

        qaoa_result = algorythms.qaoa_solver(
            n=n,
            G=G,
            elist=edges,
            reps=reps,
            optimizer_type=optimizer_name,
            maxiter=maxiter
        )
        results["qaoa"].append(qaoa_result)

    # Сохраняем после всех 100 графов для текущего n
    serializable_results = make_json_serializable(results)
    with open(f"benchmark_results_{n}.json", "w") as f:
        dump(serializable_results, f, indent=2)


