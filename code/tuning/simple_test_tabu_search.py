import optuna
import numpy as np
from code.utils import max_cut_generator_graph
from code.tasks import cut_max
from code.algs import algorithms_optimization
from tqdm import tqdm
import json

results = {}

def make_objective(n, graph_list):
    def objective(trial):
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        tabu_size = trial.suggest_int("tabu_size", 5, 50)
        window_size = trial.suggest_int("window_size", 5, 20)

        values = []

        for G in graph_list:
            edges = [(u, v, int(float(data))) for u, v, data in G.edges(data="weight")]

            result = algorithms_optimization.tabu_search(
                objective_function=cut_max.objective_function,
                generate_neighbors=cut_max.generate_neighbors,
                initial_solution=cut_max.initial_solution,
                num_vertices=n,
                edges=edges,
                max_iter=max_iter,
                tabu_size=tabu_size,
                window_size=window_size
            )

            values.append(result["best_value"])

        return -np.mean(values)
    return objective

for n in tqdm(range(5, 21), desc="Размеры графа"):
    graph_list = [max_cut_generator_graph.create_weighted_graph(n, int(n * 0.5)) for _ in range(50)]

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(n, graph_list), n_trials=30)

    results[n] = {
        "best_params": study.best_params,
        "best_value": -study.best_value,
    }

with open("../results/tabu_search_optuna_results.json", "w") as f:
    json.dump(results, f, indent=4)

for n in sorted(results):
    print(f"\nГраф размером n={n}")
    print("Лучшие параметры:", results[n]["best_params"])
    print("Среднее значение целевой функции:", results[n]["best_value"])
